
from mmgp import offload
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch, json, os
import math

from diffusers.image_processor import VaeImageProcessor
from .transformer_qwenimage import QwenImageTransformer2DModel

from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, AutoTokenizer, Qwen2VLProcessor
from .autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers import FlowMatchEulerDiscreteScheduler
from .pipeline_qwenimage import QwenImagePipeline
from PIL import Image
from shared.utils.utils import calculate_new_dimensions, convert_tensor_to_image

def stitch_images(img1, img2):
    # Resize img2 to match img1's height
    width1, height1 = img1.size
    width2, height2 = img2.size
    new_width2 = int(width2 * height1 / height2)
    img2_resized = img2.resize((new_width2, height1), Image.Resampling.LANCZOS)
    
    stitched = Image.new('RGB', (width1 + new_width2, height1))
    stitched.paste(img1, (0, 0))
    stitched.paste(img2_resized, (width1, 0))
    return stitched

class model_factory():
    def __init__(
        self,
        checkpoint_dir,
        model_filename = None,
        model_type = None, 
        model_def = None,
        base_model_type = None,
        text_encoder_filename = None,
        quantizeTransformer = False,
        save_quantized = False,
        dtype = torch.bfloat16,
        VAE_dtype = torch.float32,
        mixed_precision_transformer = False,
    ):
    

        transformer_filename = model_filename[0]
        processor = None
        tokenizer = None
        if base_model_type in ["qwen_image_edit_20B", "qwen_image_edit_plus_20B"]:
            processor = Qwen2VLProcessor.from_pretrained(os.path.join(checkpoint_dir,"Qwen2.5-VL-7B-Instruct"))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir,"Qwen2.5-VL-7B-Instruct"))
        self.base_model_type = base_model_type

        base_config_file = "configs/qwen_image_20B.json" 
        with open(base_config_file, 'r', encoding='utf-8') as f:
            transformer_config = json.load(f)
        transformer_config.pop("_diffusers_version")
        transformer_config.pop("_class_name")
        transformer_config.pop("pooled_projection_dim")
        
        from accelerate import init_empty_weights
        with init_empty_weights():
            transformer = QwenImageTransformer2DModel(**transformer_config)
        source =  model_def.get("source", None)

        if source is not None:
            offload.load_model_data(transformer, source)
        else:
            offload.load_model_data(transformer, transformer_filename)
        # transformer = offload.fast_load_transformers_model("transformer_quanto.safetensors", writable_tensors= True , modelClass=QwenImageTransformer2DModel, defaultConfigPath="transformer_config.json")

        if not source is None:
            from wgp import save_model
            save_model(transformer, model_type, dtype, None)

        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, model_filename[0], dtype, base_config_file)

        text_encoder = offload.fast_load_transformers_model(text_encoder_filename,  writable_tensors= True , modelClass=Qwen2_5_VLForConditionalGeneration,  defaultConfigPath= os.path.join(checkpoint_dir, "Qwen2.5-VL-7B-Instruct", "config.json"))
        # text_encoder = offload.fast_load_transformers_model(text_encoder_filename, do_quantize=True,  writable_tensors= True , modelClass=Qwen2_5_VLForConditionalGeneration, defaultConfigPath="text_encoder_config.json", verboseLevel=2)
        # text_encoder.to(torch.float16)
        # offload.save_model(text_encoder, "text_encoder_quanto_fp16.safetensors", do_quantize= True)

        vae = offload.fast_load_transformers_model( os.path.join(checkpoint_dir,"qwen_vae.safetensors"), writable_tensors= True , modelClass=AutoencoderKLQwenImage, defaultConfigPath=os.path.join(checkpoint_dir,"qwen_vae_config.json"))
        
        self.pipeline = QwenImagePipeline(vae, text_encoder, tokenizer, transformer, processor)
        self.vae=vae
        self.text_encoder=text_encoder
        self.tokenizer=tokenizer
        self.transformer=transformer
        self.processor = processor

    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "replace the logo with the text 'Black Forest Labs'",
        n_prompt = None,
        sampling_steps: int = 20,
        input_ref_images = None,
        input_frames= None,
        input_masks= None,
        width= 832,
        height=480,
        guide_scale: float = 4,
        fit_into_canvas = None,
        callback = None,
        loras_slists = None,
        batch_size = 1,
        video_prompt_type = "",
        VAE_tile_size = None, 
        joint_pass = True,
        sample_solver='default',
        denoising_strength = 1.,
        model_mode = 0,
        outpainting_dims = None,
        **bbargs
    ):
        # Generate with different aspect ratios
        aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472)
        }
        

        if sample_solver =='lightning':
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),  # We use shift=3 in distillation
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),  # We use shift=3 in distillation
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,  # set shift_terminal to None
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
        else:
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": 0.5,
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": 0.9,
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": 0.02,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False
            }

        self.scheduler=FlowMatchEulerDiscreteScheduler(**scheduler_config)
        self.pipeline.scheduler = self.scheduler 
        if VAE_tile_size is not None:
            self.vae.use_tiling  = VAE_tile_size[0] 
            self.vae.tile_latent_min_height  = VAE_tile_size[1] 
            self.vae.tile_latent_min_width  = VAE_tile_size[1]

        qwen_edit_plus = self.base_model_type in ["qwen_image_edit_plus_20B"]
        self.vae.enable_slicing()
        # width, height = aspect_ratios["16:9"]

        if n_prompt is None or len(n_prompt) == 0:
            n_prompt=  "text, watermark, copyright, blurry, low resolution"

        image_mask = None if input_masks is None else convert_tensor_to_image(input_masks, mask_levels= True) 
        if input_frames is not None:
            input_ref_images = [convert_tensor_to_image(input_frames) ] +  ([] if input_ref_images  is None else input_ref_images )

        if input_ref_images is not None:
            # image stiching method
            stiched = input_ref_images[0]
            if "K" in video_prompt_type :
                w, h = input_ref_images[0].size
                height, width = calculate_new_dimensions(height, width, h, w, fit_into_canvas)

            if not qwen_edit_plus:
                for new_img in input_ref_images[1:]:
                    stiched = stitch_images(stiched, new_img)
                input_ref_images  = [stiched]

        image = self.pipeline(
            prompt=input_prompt,
            negative_prompt=n_prompt,
            image = input_ref_images,
            image_mask = image_mask,
            width=width,
            height=height,
            num_inference_steps=sampling_steps,
            num_images_per_prompt = batch_size,
            true_cfg_scale=guide_scale,
            callback = callback,
            pipeline=self,
            loras_slists=loras_slists,
            joint_pass = joint_pass,
            denoising_strength=denoising_strength,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            lora_inpaint = image_mask is not None and model_mode == 1,
            outpainting_dims = outpainting_dims,
            qwen_edit_plus = qwen_edit_plus,
        )      
        if image is None: return None
        return image.transpose(0, 1)

    def get_loras_transformer(self, get_model_recursive_prop, model_type, model_mode, **kwargs):
        if model_mode == 0: return [], []
        preloadURLs = get_model_recursive_prop(model_type,  "preload_URLs")
        if len(preloadURLs) == 0: return [], []
        return [os.path.join("ckpts", os.path.basename(preloadURLs[0]))] , [1]


