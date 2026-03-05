import inspect
from typing import List, Optional, Union, Tuple

import torch
import numpy as np
import PIL.Image
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

# Import Wan modules
# Using absolute imports ensures this works when installed as a package
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.modules.clip import CLIPModel
import torchvision.transforms.functional as TF
import torch.nn.functional as F

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class WanFLF2VPipeline(DiffusionPipeline):
    r"""
    Pipeline for First-Last-Frame-to-Video generation using Wan2.1.
    """
    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    
    def __init__(
        self,
        vae: WanVAE,
        text_encoder: T5EncoderModel,
        image_encoder: CLIPModel,
        transformer: WanModel,
        scheduler: UniPCMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_stride = [4, 8, 8] # hardcoded based on config
        self.patch_size = [1, 2, 2] # hardcoded based on config

    def check_inputs(
        self,
        prompt,
        first_frame,
        last_frame,
        height,
        width,
        callback_steps,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` must be an integer > 0 if provided, but is {callback_steps}."
            )

    def prepare_latents(
        self,
        batch_size,
        num_channels,
        num_frames,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels,
            (num_frames - 1) // 4 + 1,
            height // 8,
            width // 8,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        first_frame: Union[PIL.Image.Image, List[PIL.Image.Image]] = None,
        last_frame: Union[PIL.Image.Image, List[PIL.Image.Image]] = None,
        height: Optional[int] = 720,
        width: Optional[int] = 1280,
        num_frames: Optional[int] = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "np",
        callback: Optional[callable] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[dict] = None,
    ):
        # 1. Check inputs
        self.check_inputs(prompt, first_frame, last_frame, height, width, callback_steps)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        device = self._execution_device

        # 3. Encode input prompt
        if isinstance(prompt, str):
            prompt = [prompt]
        if negative_prompt is None:
            negative_prompt = [""] * len(prompt)
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        
        self.text_encoder.model.to(device)
        context = self.text_encoder(prompt, device)
        context_null = self.text_encoder(negative_prompt, device)
        
        # 4. Preprocess images
        if not isinstance(first_frame, list):
            first_frame_list = [first_frame]
            last_frame_list = [last_frame]
        else:
            first_frame_list = first_frame
            last_frame_list = last_frame
            
        processed_first = []
        processed_last = []
        
        for f, l in zip(first_frame_list, last_frame_list):
            f_tensor = TF.to_tensor(f).sub_(0.5).div_(0.5).to(device)
            l_tensor = TF.to_tensor(l).sub_(0.5).div_(0.5).to(device)
            f_tensor = F.interpolate(f_tensor.unsqueeze(0), size=(height, width), mode='bicubic', align_corners=False).squeeze(0)
            l_tensor = F.interpolate(l_tensor.unsqueeze(0), size=(height, width), mode='bicubic', align_corners=False).squeeze(0)
            processed_first.append(f_tensor)
            processed_last.append(l_tensor)
            
        # 5. Encode images with CLIP
        clip_inputs = []
        for pf, pl in zip(processed_first, processed_last):
             clip_inputs.append(pf.unsqueeze(1)) # [3, 1, H, W]
             clip_inputs.append(pl.unsqueeze(1))
             
        self.image_encoder.model.to(device)
        clip_context = self.image_encoder.visual(clip_inputs)
        
        # 6. Encode with VAE
        y_list = []
        for pf, pl in zip(processed_first, processed_last):
            pf_input = pf.unsqueeze(1) # [3, 1, H, W]
            pl_input = pl.unsqueeze(1)
            zeros = torch.zeros(3, num_frames - 2, height, width, device=device)
            vae_input = torch.cat([pf_input, zeros, pl_input], dim=1) # [3, F, H, W]
            y_list.append(vae_input)
            
        self.vae.model.to(device)
        y = self.vae.encode(y_list) # Returns list of [C, T, H, W] latents
        
        # 7. Create Mask and Concat
        lat_h = height // 8
        lat_w = width // 8
        
        msk = torch.ones(1, num_frames, lat_h, lat_w, device=device)
        msk[:, 1:-1] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2) # [1, 4, T_lat, H_lat, W_lat]
        
        y_masked = []
        for latent in y:
             y_masked.append(torch.cat([msk[0], latent], dim=0))
             
        # 8. Prepare Latents (Noise)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        noise_shape = (16, (num_frames - 1) // 4 + 1, lat_h, lat_w)
        # Use generator for reproducibility if provided
        latents = randn_tensor(shape=noise_shape, generator=generator, device=device, dtype=torch.float32)
        latents_list = [latents] * batch_size # List of latents
        
        # 9. Denoising Loop
        seq_len = ((num_frames - 1) // 4 + 1) * lat_h * lat_w // 4
        
        self.transformer.to(device)
        
        for i, t in enumerate(self.progress_bar(timesteps)):
            t_tensor = torch.stack([t] * batch_size).to(device)
            
            # Predict noise for conditional
            noise_pred_cond = self.transformer(
                latents_list, t=t_tensor, context=context, seq_len=seq_len, clip_fea=clip_context, y=y_masked
            )
            
            # Predict noise for unconditional
            if guidance_scale > 1.0:
                noise_pred_uncond = self.transformer(
                    latents_list, t=t_tensor, context=context_null, seq_len=seq_len, clip_fea=clip_context, y=y_masked
                )
                
                # Combine (CFG)
                noise_pred_list = []
                for cond, uncond in zip(noise_pred_cond, noise_pred_uncond):
                    noise_pred_list.append(uncond + guidance_scale * (cond - uncond))
            else:
                 noise_pred_list = noise_pred_cond

            # Step
            new_latents_list = []
            for latent, noise_pred in zip(latents_list, noise_pred_list):
                 # Scheduler step usually expects [1, C, T, H, W] or similar.
                 # noise_pred is [C, T, H, W]
                 step_output = self.scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False)[0]
                 new_latents_list.append(step_output.squeeze(0))
            latents_list = new_latents_list

        # 10. Decode
        # VAE decode expects list
        videos = self.vae.decode(latents_list)
        
        output_videos = []
        for vid in videos:
             # video tensor [3, F, H, W] value range [-1, 1]
             # Denormalize to [0, 1]
             vid = (vid * 0.5 + 0.5).clamp(0, 1)
             vid = vid.permute(1, 2, 3, 0).cpu().numpy() # [F, H, W, C]
             output_videos.append(vid)
             
        if output_type == "np":
             return ImagePipelineOutput(images=output_videos)
        
        return ImagePipelineOutput(images=output_videos)

if __name__ == "__main__":
    import argparse
    from wan.configs import WAN_CONFIGS
    from functools import partial
    import os

    # Re-import to ensure we are using the module definitions available in scope if needed
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE
    from wan.modules.clip import CLIPModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--first_frame", type=str, required=True)
    parser.add_argument("--last_frame", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    config = WAN_CONFIGS['flf2v-14B']
    device = torch.device(f"cuda:{args.device_id}")
    
    print(f"Loading models from {args.checkpoint_dir}...")
    
    # 1. Text Encoder
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.checkpoint_dir, config.t5_checkpoint),
        tokenizer_path=os.path.join(args.checkpoint_dir, config.t5_tokenizer),
    )
    
    # 2. VAE
    vae = WanVAE(
        vae_pth=os.path.join(args.checkpoint_dir, config.vae_checkpoint),
        device=device
    )
    
    # 3. CLIP
    image_encoder = CLIPModel(
        dtype=config.clip_dtype,
        device=device,
        checkpoint_path=os.path.join(args.checkpoint_dir, config.clip_checkpoint),
        tokenizer_path=os.path.join(args.checkpoint_dir, config.clip_tokenizer)
    )
    
    # 4. Transformer
    transformer = WanModel.from_pretrained(args.checkpoint_dir, model_type='flf2v')
    transformer.eval().requires_grad_(False)
    
    # 5. Scheduler
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction', 
        use_flow_sigmas=True, 
        num_train_timesteps=1000, 
        flow_shift=16.0 
    )

    pipe = WanFLF2VPipeline(
        vae=vae,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        transformer=transformer,
        scheduler=scheduler
    )
    pipe.to(device)

    print(f"Loading images...")
    first_img = PIL.Image.open(args.first_frame).convert("RGB")
    last_img = PIL.Image.open(args.last_frame).convert("RGB")
    
    print("Generating video...")
    output = pipe(
        prompt=args.prompt,
        first_frame=first_img,
        last_frame=last_img,
        height=720,
        width=1280,
        num_frames=81,
        guidance_scale=5.0
    )
    
    import imageio
    video = output.images[0] # [F, H, W, C]
    video = (video * 255).astype(np.uint8)
    imageio.mimsave(args.output, video, fps=16)
    print(f"Video saved to {args.output}")
