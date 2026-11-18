# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from tqdm import tqdm
import csv
import logging
import time
from pathlib import Path
import pandas as pd
import torch


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def t2v_generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None



def i2v_generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, 
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F-1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
                # 'cond_flag': True,
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
                # 'cond_flag': False,
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None



def teacache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
        
    if self.enable_teacache:
        modulated_inp = e0 if self.use_ref_steps else e
        # teacache
        if self.cnt%2==0: # even -> conditon
            self.is_even = True
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp-self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc_even = False
                else:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()

        else: # odd -> unconditon
            self.is_even = False
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            else: 
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp-self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

    if self.enable_teacache: 
        if self.is_even:
            if not should_calc_even:
                x += self.previous_residual_even
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_even = x - ori_x
        else:
            if not should_calc_odd:
                x += self.previous_residual_odd
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_odd = x - ori_x
    
    else:
        for block in self.blocks:
            x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
    return [u.float() for u in x]

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Higher speedup will cause to worse quality -- 0.1 for 2.0x speedup -- 0.2 for 3.0x speedup")
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        default=False,
        help="Using Retention Steps will result in faster generation speed and better generation quality.")
    parser.add_argument(
        "--prompt_csv",
        type=str,
        default=None,
        help="Path to VideoPhy-2 Prompts CSV for batch generation"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="gen_videos",
        help="Directory to save batch .mp4 files"
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default="eval_results_baseline.csv",
        help="Output CSV path for VideoPhy-2"
    )    

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt_csv is not None:
            logging.info("Creating WanT2V pipeline.")
            wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
            )

            # TeaCache
            wan_t2v.__class__.generate = t2v_generate
            wan_t2v.model.__class__.enable_teacache = True
            wan_t2v.model.__class__.forward = teacache_forward
            wan_t2v.model.__class__.cnt = 0
            wan_t2v.model.__class__.num_steps = args.sample_steps*2
            wan_t2v.model.__class__.teacache_thresh = args.teacache_thresh
            wan_t2v.model.__class__.accumulated_rel_l1_distance_even = 0
            wan_t2v.model.__class__.accumulated_rel_l1_distance_odd = 0
            wan_t2v.model.__class__.previous_e0_even = None
            wan_t2v.model.__class__.previous_e0_odd = None
            wan_t2v.model.__class__.previous_residual_even = None
            wan_t2v.model.__class__.previous_residual_odd = None
            wan_t2v.model.__class__.use_ref_steps = args.use_ret_steps
            if args.use_ret_steps:
                if '1.3B' in args.ckpt_dir:
                    wan_t2v.model.__class__.coefficients = [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]
                if '14B' in args.ckpt_dir:
                    wan_t2v.model.__class__.coefficients = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
                wan_t2v.model.__class__.ret_steps = 5*2
                wan_t2v.model.__class__.cutoff_steps = args.sample_steps*2
            else:
                if '1.3B' in args.ckpt_dir:
                    wan_t2v.model.__class__.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
                if '14B' in args.ckpt_dir:
                    wan_t2v.model.__class__.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
                wan_t2v.model.__class__.ret_steps = 1*2
                wan_t2v.model.__class__.cutoff_steps = args.sample_steps*2 - 2

            def _fmt_duration(seconds: float) -> str:
                hrs, rem = divmod(int(seconds), 3600)
                mins, secs = divmod(rem, 60)
                return f"{hrs}h {mins:02d}m {secs:02d}s" if hrs > 0 else f"{mins}m {secs:02d}s"

            df = pd.read_csv(args.prompt_csv)
            total_prompts = len(df)
            video_dir = Path(args.video_dir).resolve()
            eval_csv = Path(args.eval_csv).resolve()
            video_dir.mkdir(parents=True, exist_ok=True)

            completed = set()
            if eval_csv.exists() and eval_csv.stat().st_size > 0:
                completed = set(pd.read_csv(eval_csv)["videopath"].dropna().tolist())

            already_done = sum(
                1 for idx in range(total_prompts)
                if (video_dir / f"{idx:04d}.mp4").resolve().exists()
                and str((video_dir / f"{idx:04d}.mp4").resolve()) in completed
            )
            remaining_total = total_prompts - already_done

            if rank == 0:
                logging.info("=" * 70)
                logging.info(f"[VideoPhy-2 Batch] Prompt CSV : {args.prompt_csv}")
                logging.info(f"[VideoPhy-2 Batch] Output Dir : {video_dir}")
                logging.info(f"[VideoPhy-2 Batch] Eval CSV   : {eval_csv}")
                logging.info(
                    f"[VideoPhy-2 Batch] Config     : size={args.size}, frames={args.frame_num}, "
                    f"steps={args.sample_steps}, shift={args.sample_shift}, cfg={args.sample_guide_scale}"
                )
                logging.info(
                    f"[VideoPhy-2 Batch] Progress   : {total_prompts} total | "
                    f"{already_done} already completed | {remaining_total} remaining"
                )
                logging.info("=" * 70)

            extra_cols = [c for c in df.columns if c not in ("videopath", "caption")]
            fieldnames = ["videopath", "caption", "gen_time_sec"] + extra_cols
            write_header = not eval_csv.exists() or eval_csv.stat().st_size == 0

            session_times: list[float] = []
            batch_start_time = time.perf_counter()

            with eval_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    f.flush()

                for idx, row in df.iterrows():
                    video_name = f"{idx:04d}.mp4"
                    out_file = (video_dir / video_name).resolve()

                    if out_file.exists() and str(out_file) in completed:
                        continue

                    # Defensive reset so TeaCache / EasyCache step counters start clean on every prompt
                    if hasattr(wan_t2v.model, "cnt"):
                        wan_t2v.model.cnt = 0
                    if hasattr(wan_t2v.model.__class__, "cnt"):
                        wan_t2v.model.__class__.cnt = 0

                    current_seed = args.base_seed + idx
                    short_caption = str(row["caption"]).replace("\n", " ")[:70]

                    if rank == 0:
                        logging.info(
                            f"[{idx + 1}/{total_prompts}] Starting {video_name} "
                            f"(seed={current_seed}) | Caption: \"{short_caption}...\""
                        )

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    t0 = time.perf_counter()
                    try:
                        video = wan_t2v.generate(
                            str(row["upsampled_caption"]),
                            size=SIZE_CONFIGS[args.size],
                            frame_num=args.frame_num,
                            shift=args.sample_shift,
                            sample_solver=args.sample_solver,
                            sampling_steps=args.sample_steps,
                            guide_scale=args.sample_guide_scale,
                            seed=current_seed,
                            offload_model=args.offload_model,
                        )
                    except Exception as exc:
                        if rank == 0:
                            logging.error(
                                f"[{idx + 1}/{total_prompts}] FAILED on {video_name} "
                                f"(seed={current_seed}): {exc}"
                            )
                        raise

                    elapsed = round(time.perf_counter() - t0, 3)
                    session_times.append(elapsed)

                    vram_str = ""
                    if torch.cuda.is_available():
                        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        vram_str = f" | Peak VRAM: {peak_gb:.2f} GB"

                    if rank == 0:
                        cache_video(
                            tensor=video[None],
                            save_file=str(out_file),
                            fps=cfg.sample_fps,
                            normalize=True,
                            value_range=(-1, 1),
                        )
                        record = {col: row[col] for col in extra_cols}
                        record["videopath"] = str(out_file)
                        record["caption"] = row["caption"]
                        record["gen_time_sec"] = elapsed
                        writer.writerow(record)
                        f.flush()

                        avg_sec = sum(session_times) / len(session_times)
                        videos_left = remaining_total - len(session_times)
                        eta_str = _fmt_duration(avg_sec * videos_left)

                        logging.info(
                            f"[{idx + 1}/{total_prompts}] Saved {video_name} in {elapsed:.2f}s | "
                            f"Avg: {avg_sec:.2f}s/vid | ETA: {eta_str}{vram_str}"
                        )

            if rank == 0:
                total_elapsed = time.perf_counter() - batch_start_time
                avg_final = (sum(session_times) / len(session_times)) if session_times else 0.0
                logging.info("=" * 70)
                logging.info(
                    f"[VideoPhy-2 Batch] Finished! Generated {len(session_times)} videos in "
                    f"{_fmt_duration(total_elapsed)} (Avg: {avg_final:.2f}s/vid)"
                )
                logging.info(f"[VideoPhy-2 Batch] Saved manifest: {eval_csv}")
                logging.info("=" * 70)
        else:
            if args.prompt is None:
                args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
            logging.info(f"Input prompt: {args.prompt}")
            if args.use_prompt_extend:
                logging.info("Extending prompt ...")
                if rank == 0:
                    prompt_output = prompt_expander(
                        args.prompt,
                        tar_lang=args.prompt_extend_target_lang,
                        seed=args.base_seed)
                    if prompt_output.status == False:
                        logging.info(
                            f"Extending prompt failed: {prompt_output.message}")
                        logging.info("Falling back to original prompt.")
                        input_prompt = args.prompt
                    else:
                        input_prompt = prompt_output.prompt
                    input_prompt = [input_prompt]
                else:
                    input_prompt = [None]
                if dist.is_initialized():
                    dist.broadcast_object_list(input_prompt, src=0)
                args.prompt = input_prompt[0]
                logging.info(f"Extended prompt: {args.prompt}")

            logging.info("Creating WanT2V pipeline.")
            wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
            )

            # TeaCache
            wan_t2v.__class__.generate = t2v_generate
            wan_t2v.model.__class__.enable_teacache = True
            wan_t2v.model.__class__.forward = teacache_forward
            wan_t2v.model.__class__.cnt = 0
            wan_t2v.model.__class__.num_steps = args.sample_steps*2
            wan_t2v.model.__class__.teacache_thresh = args.teacache_thresh
            wan_t2v.model.__class__.accumulated_rel_l1_distance_even = 0
            wan_t2v.model.__class__.accumulated_rel_l1_distance_odd = 0
            wan_t2v.model.__class__.previous_e0_even = None
            wan_t2v.model.__class__.previous_e0_odd = None
            wan_t2v.model.__class__.previous_residual_even = None
            wan_t2v.model.__class__.previous_residual_odd = None
            wan_t2v.model.__class__.use_ref_steps = args.use_ret_steps
            if args.use_ret_steps:
                if '1.3B' in args.ckpt_dir:
                    wan_t2v.model.__class__.coefficients = [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]
                if '14B' in args.ckpt_dir:
                    wan_t2v.model.__class__.coefficients = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
                wan_t2v.model.__class__.ret_steps = 5*2
                wan_t2v.model.__class__.cutoff_steps = args.sample_steps*2
            else:
                if '1.3B' in args.ckpt_dir:
                    wan_t2v.model.__class__.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
                if '14B' in args.ckpt_dir:
                    wan_t2v.model.__class__.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
                wan_t2v.model.__class__.ret_steps = 1*2
                wan_t2v.model.__class__.cutoff_steps = args.sample_steps*2 - 2
            logging.info(
                f"Generating {'image' if 't2i' in args.task else 'video'} ...")
            video = wan_t2v.generate(
                args.prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model)

    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
        # TeaCache
        wan_i2v.__class__.generate = i2v_generate
        wan_i2v.model.__class__.enable_teacache = True
        wan_i2v.model.__class__.forward = teacache_forward
        wan_i2v.model.__class__.cnt = 0
        wan_i2v.model.__class__.num_steps = args.sample_steps*2
        wan_i2v.model.__class__.teacache_thresh = args.teacache_thresh
        wan_i2v.model.__class__.accumulated_rel_l1_distance_even = 0
        wan_i2v.model.__class__.accumulated_rel_l1_distance_odd = 0
        wan_i2v.model.__class__.previous_e0_even = None
        wan_i2v.model.__class__.previous_e0_odd = None
        wan_i2v.model.__class__.previous_residual_even = None
        wan_i2v.model.__class__.previous_residual_odd = None
        wan_i2v.model.__class__.use_ref_steps = args.use_ret_steps
        if args.use_ret_steps:
            if '480P' in args.ckpt_dir:
                wan_i2v.model.__class__.coefficients = [ 2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01]
            if '720P' in args.ckpt_dir:
                wan_i2v.model.__class__.coefficients = [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02]
            wan_i2v.model.__class__.ret_steps = 5*2
            wan_i2v.model.__class__.cutoff_steps = args.sample_steps*2
        else:
            if '480P' in args.ckpt_dir:
                wan_i2v.model.__class__.coefficients = [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01,  5.87348440e+00, -2.01973289e-01]
            if '720P' in args.ckpt_dir:
                wan_i2v.model.__class__.coefficients = [-114.36346466,   65.26524496,  -18.82220707,    4.91518089,   -0.23412683]
            wan_i2v.model.__class__.ret_steps = 1*2
            wan_i2v.model.__class__.cutoff_steps = args.sample_steps*2 - 2

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")    
    
    

if __name__ == "__main__":
    args = _parse_args()
    generate(args)