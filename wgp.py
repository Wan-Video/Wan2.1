import os
os.environ["GRADIO_LANG"] = "en"
# # os.environ.pop("TORCH_LOGS", None)  # make sure no env var is suppressing/overriding
# os.environ["TORCH_LOGS"]= "recompiles"
import torch._logging as tlog
# tlog.set_logs(recompiles=True, guards=True, graph_breaks=True)    
import time
import sys
import threading
import argparse
from mmgp import offload, safetensors2, profile_type 
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from datetime import datetime
import gradio as gr
import random
import json
import numpy as np
import importlib
from shared.utils import notification_sound
from shared.utils.loras_mutipliers import preparse_loras_multipliers, parse_loras_multipliers
from shared.utils.utils import convert_tensor_to_image, save_image, get_video_info, get_file_creation_date, convert_image_to_video, calculate_new_dimensions, convert_image_to_tensor, calculate_dimensions_and_resize_image, rescale_and_crop, get_video_frame, resize_and_remove_background, rgb_bw_to_rgba_mask
from shared.utils.utils import calculate_new_dimensions, get_outpainting_frame_location, get_outpainting_full_area_dimensions
from shared.utils.utils import has_video_file_extension, has_image_file_extension
from shared.utils.audio_video import extract_audio_tracks, combine_video_with_audio_tracks, combine_and_concatenate_video_with_audio_tracks, cleanup_temp_audio_files,  save_video, save_image
from shared.utils.audio_video import save_image_metadata, read_image_metadata
from shared.match_archi import match_nvidia_architecture
from shared.attention import get_attention_modes, get_supported_attention_modes
from huggingface_hub import hf_hub_download, snapshot_download    
import torch
import gc
import traceback
import math 
import typing
import asyncio
import inspect
from shared.utils import prompt_parser
import base64
import io
from PIL import Image
import zipfile
import tempfile
import atexit
import shutil
import glob
import cv2
from transformers.utils import logging
logging.set_verbosity_error
from preprocessing.matanyone  import app as matanyone_app
from tqdm import tqdm
import requests
from shared.gradio.gallery import AdvancedMediaGallery

# import torch._dynamo as dynamo
# dynamo.config.recompile_limit = 2000   # default is 256
# dynamo.config.accumulated_recompile_limit = 2000  # or whatever limit you want

global_queue_ref = []
AUTOSAVE_FILENAME = "queue.zip"
PROMPT_VARS_MAX = 10

target_mmgp_version = "3.6.0"
WanGP_version = "8.73"
settings_version = 2.36
max_source_video_frames = 3000
prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer = None, None, None, None

from importlib.metadata import version
mmgp_version = version("mmgp")
if mmgp_version != target_mmgp_version:
    print(f"Incorrect version of mmgp ({mmgp_version}), version {target_mmgp_version} is needed. Please upgrade with the command 'pip install -r requirements.txt'")
    exit()
lock = threading.Lock()
current_task_id = None
task_id = 0
vmc_event_handler = matanyone_app.get_vmc_event_handler()
unique_id = 0
unique_id_lock = threading.Lock()
gen_lock = threading.Lock()
offloadobj = enhancer_offloadobj = wan_model = None
reload_needed = True

def clear_gen_cache():
    if "_cache" in offload.shared_state:
        del offload.shared_state["_cache"]

def release_model():
    global wan_model, offloadobj, reload_needed
    wan_model = None    
    clear_gen_cache()
    offload.shared_state
    if offloadobj is not None:
        offloadobj.release()
        offloadobj = None
        torch.cuda.empty_cache()
        gc.collect()
        try:
            torch._C._host_emptyCache()
        except:
            pass
        reload_needed = True
    else:
        gc.collect()

def get_unique_id():
    global unique_id  
    with unique_id_lock:
        unique_id += 1
    return str(time.time()+unique_id)

def download_ffmpeg():
    if os.name != 'nt': return
    exes = ['ffmpeg.exe', 'ffprobe.exe', 'ffplay.exe']
    if all(os.path.exists(e) for e in exes): return
    api_url = 'https://api.github.com/repos/GyanD/codexffmpeg/releases/latest'
    r = requests.get(api_url, headers={'Accept': 'application/vnd.github+json'})
    assets = r.json().get('assets', [])
    zip_asset = next((a for a in assets if 'essentials_build.zip' in a['name']), None)
    if not zip_asset: return
    zip_url = zip_asset['browser_download_url']
    zip_name = zip_asset['name']
    with requests.get(zip_url, stream=True) as resp:
        total = int(resp.headers.get('Content-Length', 0))
        with open(zip_name, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    with zipfile.ZipFile(zip_name) as z:
        for f in z.namelist():
            if f.endswith(tuple(exes)) and '/bin/' in f:
                z.extract(f)
                os.rename(f, os.path.basename(f))
    os.remove(zip_name)


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif seconds >= 60:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{seconds:.1f}s"

def pil_to_base64_uri(pil_image, format="png", quality=75):
    if pil_image is None:
        return None

    if isinstance(pil_image, str):
        from shared.utils.utils import get_video_frame
        pil_image = get_video_frame(pil_image, 0)

    buffer = io.BytesIO()
    try:
        img_to_save = pil_image
        if format.lower() == 'jpeg' and pil_image.mode == 'RGBA':
            img_to_save = pil_image.convert('RGB')
        elif format.lower() == 'png' and pil_image.mode not in ['RGB', 'RGBA', 'L', 'P']:
             img_to_save = pil_image.convert('RGBA')
        elif pil_image.mode == 'P':
             img_to_save = pil_image.convert('RGBA' if 'transparency' in pil_image.info else 'RGB')
        if format.lower() == 'jpeg':
            img_to_save.save(buffer, format=format, quality=quality)
        else:
            img_to_save.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        encoded_string = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting PIL to base64: {e}")
        return None

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def compute_sliding_window_no(current_video_length, sliding_window_size, discard_last_frames, reuse_frames):
    left_after_first_window = current_video_length - sliding_window_size + discard_last_frames
    return 1 + math.ceil(left_after_first_window / (sliding_window_size - discard_last_frames - reuse_frames))


def clean_image_list(gradio_list):
    if not isinstance(gradio_list, list): gradio_list = [gradio_list]
    gradio_list = [ tup[0] if isinstance(tup, tuple) else tup for tup in gradio_list ]        

    if any( not isinstance(image, (Image.Image, str))  for image in gradio_list): return None
    if any( isinstance(image, str) and not has_image_file_extension(image) for image in gradio_list): return None
    gradio_list = [ convert_image( Image.open(img) if isinstance(img, str) else img  ) for img in gradio_list  ]        
    return gradio_list

def cancel_edit(state):
    state["editing_task_index"] = None
    gr.Info("Edit cancelled.")
    return gr.Tabs(selected="video_gen")

def edit_task_in_queue(
            lset_name,
            image_mode,
            prompt,
            negative_prompt,
            resolution,
            video_length,
            batch_size,
            seed,
            force_fps,
            num_inference_steps,
            guidance_scale,
            guidance2_scale,
            guidance3_scale,
            switch_threshold,
            switch_threshold2,
            guidance_phases,
            model_switch_phase,
            audio_guidance_scale,
            flow_shift,
            sample_solver,
            embedded_guidance_scale,
            repeat_generation,
            multi_prompts_gen_type,
            multi_images_gen_type,
            skip_steps_cache_type,
            skip_steps_multiplier,
            skip_steps_start_step_perc,    
            loras_choices,
            loras_multipliers,
            image_prompt_type,
            image_start,
            image_end,
            model_mode,
            video_source,
            keep_frames_video_source,
            video_guide_outpainting,
            video_prompt_type,
            image_refs,
            frames_positions,
            video_guide,
            image_guide,
            keep_frames_video_guide,
            denoising_strength,
            video_mask,
            image_mask,
            control_net_weight,
            control_net_weight2,
            mask_expand,
            audio_guide,
            audio_guide2,
            audio_source,            
            audio_prompt_type,
            speakers_locations,
            sliding_window_size,
            sliding_window_overlap,
            sliding_window_color_correction_strength,
            sliding_window_overlap_noise,
            sliding_window_discard_last_frames,
            image_refs_relative_size,
            remove_background_images_ref,
            temporal_upsampling,
            spatial_upsampling,
            film_grain_intensity,
            film_grain_saturation,
            MMAudio_setting,
            MMAudio_prompt,
            MMAudio_neg_prompt,            
            RIFLEx_setting,
            NAG_scale,
            NAG_tau,
            NAG_alpha,
            slg_switch, 
            slg_layers,
            slg_start_perc,
            slg_end_perc,
            apg_switch,
            cfg_star_switch,
            cfg_zero_step,
            prompt_enhancer,
            min_frames_if_references,
            override_profile,
            mode,
            state,
):
    new_inputs = get_function_arguments(edit_task_in_queue, locals())
    new_inputs.pop('lset_name', None)

    if 'loras_choices' in new_inputs:
        all_loras = state.get("loras", [])
        lora_indices = new_inputs.pop('loras_choices')
        activated_lora_filenames = [Path(all_loras[int(index)]).name for index in lora_indices]
        new_inputs['activated_loras'] = activated_lora_filenames

    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    editing_task_index = state.get("editing_task_index", None)

    if editing_task_index is None:
        gr.Warning("No task selected for editing.")
        return update_queue_data(queue), gr.Tabs(selected="video_gen")

    task_to_edit_index = editing_task_index + 1
    if task_to_edit_index >= len(queue):
        gr.Warning("Task index out of bounds. Cannot edit.")
        return update_queue_data(queue), gr.Tabs(selected="video_gen")

    task_to_edit = queue[task_to_edit_index]
    original_params = task_to_edit['params'].copy()
    media_keys = [
        "image_start", "image_end", "image_refs", "video_source", 
        "video_guide", "image_guide", "video_mask", "image_mask",
        "audio_guide", "audio_guide2", "audio_source"
    ]

    multi_image_keys = ["image_refs"]
    single_image_keys = ["image_start", "image_end"]

    for key, new_value in new_inputs.items():
        if key in media_keys:
            if new_value is not None:
                if key in multi_image_keys:
                    cleaned_value = clean_image_list(new_value)
                    original_params[key] = cleaned_value
                elif key in single_image_keys:
                    cleaned_list = clean_image_list(new_value)
                    original_params[key] = cleaned_list[0] if cleaned_list else None
                else:
                    original_params[key] = new_value
        else:
            original_params[key] = new_value
            
    task_to_edit['params'] = original_params
    task_to_edit['prompt'] = new_inputs.get('prompt')
    task_to_edit['length'] = new_inputs.get('video_length')
    task_to_edit['steps'] = new_inputs.get('num_inference_steps')
    update_task_thumbnails(task_to_edit, original_params)
    
    gr.Info(f"Task ID {task_to_edit['id']} has been updated successfully.")
    state["editing_task_index"] = None
    return update_queue_data(queue), gr.Tabs(selected="video_gen")

def process_prompt_and_add_tasks(state, model_choice):
    def ret():
        return gr.update(), gr.update()
    
    if state.get("validate_success",0) != 1:
        ret()
    
    state["validate_success"] = 0
    model_filename = state["model_filename"]
    model_type = state["model_type"]
    inputs = get_model_settings(state, model_type)

    if model_choice != model_type or inputs ==None:
        raise gr.Error("Webform can not be used as the App has been restarted since the form was displayed. Please refresh the page")
    
    inputs["state"] =  state
    gen = get_gen_info(state)
    inputs["model_type"] = model_type
    inputs.pop("lset_name")
    if inputs == None:
        gr.Warning("Internal state error: Could not retrieve inputs for the model.")
        queue = gen.get("queue", [])
        return ret()
    model_def = get_model_def(model_type)
    model_handler = get_model_handler(model_type)
    image_outputs = inputs["image_mode"] > 0
    any_steps_skipping = model_def.get("tea_cache", False) or model_def.get("mag_cache", False)
    model_type = get_base_model_type(model_type)
    inputs["model_filename"] = model_filename
    
    mode = inputs["mode"]
    if mode.startswith("edit_"):
        edit_video_source =gen.get("edit_video_source", None)
        edit_overrides =gen.get("edit_overrides", None)
        _ , _ , _, frames_count = get_video_info(edit_video_source)
        if frames_count > max_source_video_frames:
            gr.Info(f"Post processing is not supported on videos longer than {max_source_video_frames} frames. Output Video will be truncated")
            # return
        for k in ["image_start", "image_end", "image_refs", "video_guide", "audio_guide", "audio_guide2", "audio_source" , "video_mask", "image_mask"]:
            inputs[k] = None    
        inputs.update(edit_overrides)
        del gen["edit_video_source"], gen["edit_overrides"]
        inputs["video_source"]= edit_video_source 
        prompt = []

        spatial_upsampling = inputs.get("spatial_upsampling","")
        if len(spatial_upsampling) >0: prompt += ["Spatial Upsampling"]
        temporal_upsampling = inputs.get("temporal_upsampling","")
        if len(temporal_upsampling) >0: prompt += ["Temporal Upsampling"]
        if has_image_file_extension(edit_video_source)  and len(temporal_upsampling) > 0:
            gr.Info("Temporal Upsampling can not be used with an Image")
            return ret()
        film_grain_intensity  = inputs.get("film_grain_intensity",0)
        film_grain_saturation  = inputs.get("film_grain_saturation",0.5)        
        # if film_grain_intensity >0: prompt += [f"Film Grain: intensity={film_grain_intensity}, saturation={film_grain_saturation}"]
        if film_grain_intensity >0: prompt += ["Film Grain"]
        MMAudio_setting = inputs.get("MMAudio_setting",0)
        repeat_generation= inputs.get("repeat_generation",1)
        if mode =="edit_remux":
            audio_source = inputs["audio_source"]
            if  MMAudio_setting== 1:
                prompt += ["MMAudio"]
                audio_source = None 
                inputs["audio_source"] = audio_source
            else:
                if audio_source is None:
                    gr.Info("You must provide a custom Audio")
                    return ret()
                prompt += ["Custom Audio"]
                repeat_generation == 1

        seed = inputs.get("seed",None)
        if len(prompt) == 0:
            if mode=="edit_remux":
                gr.Info("You must choose at least one Remux Method")
            else:
                gr.Info("You must choose at least one Post Processing Method")
            return ret()
        inputs["prompt"] = ", ".join(prompt)
        add_video_task(**inputs)
        gen["prompts_max"] = 1 + gen.get("prompts_max",0)
        state["validate_success"] = 1
        queue= gen.get("queue", [])
        return ret()

    if hasattr(model_handler, "validate_generative_settings"):
        error = model_handler.validate_generative_settings(model_type, model_def, inputs)
        if error is not None and len(error) > 0:
            gr.Info(error)
            return ret()
    if inputs.get("cfg_star_switch", 0) != 0 and inputs.get("apg_switch", 0) != 0:
        gr.Info("Adaptive Progressive Guidance and Classifier Free Guidance Star can not be set at the same time")
        return ret()
    prompt = inputs["prompt"]
    if len(prompt) ==0:
        gr.Info("Prompt cannot be empty.")
        gen = get_gen_info(state)
        queue = gen.get("queue", [])
        return ret()
    prompt, errors = prompt_parser.process_template(prompt)
    if len(errors) > 0:
        gr.Info("Error processing prompt template: " + errors)
        return ret()
    model_filename = get_model_filename(model_type)  
    prompts = prompt.replace("\r", "").split("\n")
    prompts = [prompt.strip() for prompt in prompts if len(prompt.strip())>0 and not prompt.startswith("#")]
    if len(prompts) == 0:
        gr.Info("Prompt cannot be empty.")
        gen = get_gen_info(state)
        queue = gen.get("queue", [])
        return ret()

    resolution = inputs["resolution"]
    width, height = resolution.split("x")
    width, height = int(width), int(height)
    image_start = inputs["image_start"]
    image_end = inputs["image_end"]
    image_refs = inputs["image_refs"]
    image_prompt_type = inputs["image_prompt_type"]
    audio_prompt_type = inputs["audio_prompt_type"]
    if image_prompt_type == None: image_prompt_type = ""
    video_prompt_type = inputs["video_prompt_type"]
    if video_prompt_type == None: video_prompt_type = ""
    force_fps = inputs["force_fps"]
    audio_guide = inputs["audio_guide"]
    audio_guide2 = inputs["audio_guide2"]
    audio_source = inputs["audio_source"]
    video_guide = inputs["video_guide"]
    image_guide = inputs["image_guide"]
    video_mask = inputs["video_mask"]
    image_mask = inputs["image_mask"]
    speakers_locations = inputs["speakers_locations"]
    video_source = inputs["video_source"]
    frames_positions = inputs["frames_positions"]
    keep_frames_video_guide= inputs["keep_frames_video_guide"] 
    keep_frames_video_source = inputs["keep_frames_video_source"]
    denoising_strength= inputs["denoising_strength"]     
    sliding_window_size = inputs["sliding_window_size"]
    sliding_window_overlap = inputs["sliding_window_overlap"]
    sliding_window_discard_last_frames = inputs["sliding_window_discard_last_frames"]
    video_length = inputs["video_length"]
    num_inference_steps= inputs["num_inference_steps"]
    skip_steps_cache_type= inputs["skip_steps_cache_type"]
    MMAudio_setting = inputs["MMAudio_setting"]
    image_mode = inputs["image_mode"]
    switch_threshold = inputs["switch_threshold"]
    loras_multipliers = inputs["loras_multipliers"]
    activated_loras = inputs["activated_loras"]
    guidance_phases= inputs["guidance_phases"]
    model_switch_phase = inputs["model_switch_phase"]    
    switch_threshold = inputs["switch_threshold"]
    switch_threshold2 = inputs["switch_threshold2"]
    multi_prompts_gen_type = inputs["multi_prompts_gen_type"]
    video_guide_outpainting = inputs["video_guide_outpainting"]

    outpainting_dims = get_outpainting_dims(video_guide_outpainting)

    if server_config.get("fit_canvas", 0) == 2 and outpainting_dims is not None and any_letters(video_prompt_type, "VKF"):
        gr.Info("Output Resolution Cropping will be not used for this Generation as it is not compatible with Video Outpainting")

    if len(loras_multipliers) > 0:
        _, _, errors =  parse_loras_multipliers(loras_multipliers, len(activated_loras), num_inference_steps, nb_phases= guidance_phases)
        if len(errors) > 0: 
            gr.Info(f"Error parsing Loras Multipliers: {errors}")
            return ret()
    if guidance_phases == 3:
        if switch_threshold < switch_threshold2:
            gr.Info(f"Phase 1-2 Switch Noise Level ({switch_threshold}) should be Greater than Phase 2-3 Switch Noise Level ({switch_threshold2}). As a reminder, noise will gradually go down from 1000 to 0.")
            return ret()
    else:
        model_switch_phase = 1
        
    if not any_steps_skipping: skip_steps_cache_type = ""
    if not model_def.get("lock_inference_steps", False) and model_type in ["ltxv_13B"] and num_inference_steps < 20:
        gr.Info("The minimum number of steps should be 20") 
        return ret()
    if skip_steps_cache_type == "mag":
        if num_inference_steps > 50:
            gr.Info("Mag Cache maximum number of steps is 50")
            return ret()
        
    if image_mode > 0:
        audio_prompt_type = ""

    if "B" in audio_prompt_type or "X" in audio_prompt_type:
        from models.wan.multitalk.multitalk import parse_speakers_locations
        speakers_bboxes, error = parse_speakers_locations(speakers_locations)
        if len(error) > 0:
            gr.Info(error)
            return ret()

    if MMAudio_setting != 0 and server_config.get("mmaudio_enabled", 0) != 0 and video_length <16: #should depend on the architecture
        gr.Info("MMAudio can generate an Audio track only if the Video is at least 1s long")
    if "F" in video_prompt_type:
        if len(frames_positions.strip()) > 0:
            positions = frames_positions.replace(","," ").split(" ")
            for pos_str in positions:
                if not pos_str in ["L", "l"] and len(pos_str)>0: 
                    if not is_integer(pos_str):
                        gr.Info(f"Invalid Frame Position '{pos_str}'")
                        return ret()
                    pos = int(pos_str)
                    if pos <1 or pos > max_source_video_frames:
                        gr.Info(f"Invalid Frame Position Value'{pos_str}'")
                        return ret()
    else:
        frames_positions = None

    if audio_source is not None and MMAudio_setting != 0:
        gr.Info("MMAudio and Custom Audio Soundtrack can't not be used at the same time")
        return ret()
    if len(filter_letters(image_prompt_type, "VLG")) > 0 and len(keep_frames_video_source) > 0:
        if not is_integer(keep_frames_video_source) or int(keep_frames_video_source) == 0:
            gr.Info("The number of frames to keep must be a non null integer") 
            return ret()
    else:
        keep_frames_video_source = ""

    if image_outputs:
        image_prompt_type = image_prompt_type.replace("V", "").replace("L", "")

    if "V" in image_prompt_type:
        if video_source == None:
            gr.Info("You must provide a Source Video file to continue")
            return ret()
    else:
        video_source = None

    if "A" in audio_prompt_type:
        if audio_guide == None:
            gr.Info("You must provide an Audio Source")
            return ret()
        if "B" in audio_prompt_type:
            if audio_guide2 == None:
                gr.Info("You must provide a second Audio Source")
                return ret()
        else:
            audio_guide2 = None
    else:
        audio_guide = None
        audio_guide2 = None
        
    if model_type in ["vace_multitalk_14B"] and ("B" in audio_prompt_type or "X" in audio_prompt_type):
        if not "I" in video_prompt_type and not not "V" in video_prompt_type:
            gr.Info("To get good results with Multitalk and two people speaking, it is recommended to set a Reference Frame or a Control Video (potentially truncated) that contains the two people one on each side")

    if model_def.get("one_image_ref_needed", False):
        if image_refs  == None :
            gr.Info("You must provide an Image Reference") 
            return ret()
        if len(image_refs) > 1:
            gr.Info("Only one Image Reference (a person) is supported for the moment by this model") 
            return ret()
    if model_def.get("at_least_one_image_ref_needed", False):
        if image_refs  == None :
            gr.Info("You must provide at least one Image Reference") 
            return ret()
        
    if "I" in video_prompt_type:
        if image_refs == None or len(image_refs) == 0:
            gr.Info("You must provide at least one Reference Image")
            return ret()
        image_refs = clean_image_list(image_refs)
        if image_refs == None :
            gr.Info("A Reference Image should be an Image") 
            return ret()
    else:
        image_refs = None

    if "V" in video_prompt_type:
        if image_outputs:
            if image_guide is None:
                gr.Info("You must provide a Control Image")
                return ret()
        else:
            if video_guide is None:
                gr.Info("You must provide a Control Video")
                return ret()
        if "A" in video_prompt_type and not "U" in video_prompt_type:             
            if image_outputs:
                if image_mask is None:
                    gr.Info("You must provide a Image Mask")
                    return ret()
            else:
                if video_mask is None:
                    gr.Info("You must provide a Video Mask")
                    return ret()
        else:
            video_mask = None
            image_mask = None

        if "G" in video_prompt_type:
                if denoising_strength < 1.:
                    gr.Info(f"With Denoising Strength {denoising_strength:.1f}, denoising will start at Step no {int(round(num_inference_steps * (1. - denoising_strength),4))} ")
        else: 
            denoising_strength = 1.0
        if len(keep_frames_video_guide) > 0 and model_type in ["ltxv_13B"]:
            gr.Info("Keep Frames for Control Video is not supported with LTX Video")
            return ret()
        _, error = parse_keep_frames_video_guide(keep_frames_video_guide, video_length)
        if len(error) > 0:
            gr.Info(f"Invalid Keep Frames property: {error}")
            return ret()
    else:
        video_guide = None
        image_guide = None
        video_mask = None
        image_mask = None
        keep_frames_video_guide = ""
        denoising_strength = 1.0
    
    if image_outputs:
        video_guide = None
        video_mask = None
    else:
        image_guide = None
        image_mask = None


    if "S" in image_prompt_type:
        if image_start == None or isinstance(image_start, list) and len(image_start) == 0:
            gr.Info("You must provide a Start Image")
            return ret()
        image_start = clean_image_list(image_start)        
        if image_start == None :
            gr.Info("Start Image should be an Image") 
            return ret()
        if  multi_prompts_gen_type == 1 and len(image_start) > 1:
            gr.Info("Only one Start Image is supported") 
            return ret()       
    else:
        image_start = None

    if not any_letters(image_prompt_type, "SVL"):
        image_prompt_type = image_prompt_type.replace("E", "")
    if "E" in image_prompt_type:
        if image_end == None or isinstance(image_end, list) and len(image_end) == 0:
            gr.Info("You must provide an End Image") 
            return ret()
        image_end = clean_image_list(image_end)        
        if image_end == None :
            gr.Info("End Image should be an Image") 
            return ret()
        if multi_prompts_gen_type == 0:
            if video_source is not None:
                if len(image_end)> 1:
                    gr.Info("If a Video is to be continued and the option 'Each Text Prompt Will create a new generated Video' is set, there can be only one End Image")
                    return ret()        
            elif len(image_start or []) != len(image_end or []):
                gr.Info("The number of Start and End Images should be the same when the option 'Each Text Prompt Will create a new generated Video'")
                return ret()    
    else:        
        image_end = None


    if test_any_sliding_window(model_type) and image_mode == 0:
        if video_length > sliding_window_size:
            if model_type in ["t2v"] and not "G" in video_prompt_type :
                gr.Info(f"You have requested to Generate Sliding Windows with a Text to Video model. Unless you use the Video to Video feature this is useless as a t2v model doesn't see past frames and it will generate the same video in each new window.") 
                return ret()
            full_video_length = video_length if video_source is None else video_length +  sliding_window_overlap -1
            extra = "" if full_video_length == video_length else f" including {sliding_window_overlap} added for Video Continuation"
            no_windows = compute_sliding_window_no(full_video_length, sliding_window_size, sliding_window_discard_last_frames, sliding_window_overlap)
            gr.Info(f"The Number of Frames to generate ({video_length}{extra}) is greater than the Sliding Window Size ({sliding_window_size}), {no_windows} Windows will be generated")
    if "recam" in model_filename:
        if video_guide == None:
            gr.Info("You must provide a Control Video")
            return ret()
        computed_fps = get_computed_fps(force_fps, model_type , video_guide, video_source )
        frames = get_resampled_video(video_guide, 0, 81, computed_fps)
        if len(frames)<81:
            gr.Info(f"Recammaster Control video should be at least 81 frames once the resampling at {computed_fps} fps has been done")
            return ret()

    if "hunyuan_custom_custom_edit" in model_filename:
        if len(keep_frames_video_guide) > 0: 
            gr.Info("Filtering Frames with this model is not supported")
            return ret()

    if inputs["multi_prompts_gen_type"] != 0:
        if image_start != None and len(image_start) > 1:
            gr.Info("Only one Start Image must be provided if multiple prompts are used for different windows") 
            return ret()

        # if image_end != None and len(image_end) > 1:
        #     gr.Info("Only one End Image must be provided if multiple prompts are used for different windows") 
        #     return

    override_inputs = {
        "image_start": image_start[0] if image_start !=None and len(image_start) > 0 else None,
        "image_end": image_end, #[0] if image_end !=None and len(image_end) > 0 else None,
        "image_refs": image_refs,
        "audio_guide": audio_guide,
        "audio_guide2": audio_guide2,
        "audio_source": audio_source,
        "video_guide": video_guide,
        "image_guide": image_guide,
        "video_mask": video_mask,
        "image_mask": image_mask,
        "video_source": video_source,
        "frames_positions": frames_positions,
        "keep_frames_video_source": keep_frames_video_source,
        "keep_frames_video_guide": keep_frames_video_guide,
        "denoising_strength": denoising_strength,
        "image_prompt_type": image_prompt_type,
        "video_prompt_type": video_prompt_type,        
        "audio_prompt_type": audio_prompt_type,
        "skip_steps_cache_type": skip_steps_cache_type,
        "model_switch_phase": model_switch_phase,
    } 

    if inputs["multi_prompts_gen_type"] == 0:
        if image_start != None and len(image_start) > 0:
            if inputs["multi_images_gen_type"] == 0:
                new_prompts = []
                new_image_start = []
                new_image_end = []
                for i in range(len(prompts) * len(image_start) ):
                    new_prompts.append(  prompts[ i % len(prompts)] )
                    new_image_start.append(image_start[i // len(prompts)] )
                    if image_end != None:
                        new_image_end.append(image_end[i // len(prompts)] )
                prompts = new_prompts
                image_start = new_image_start 
                if image_end != None:
                    image_end = new_image_end 
            else:
                if len(prompts) >= len(image_start):
                    if len(prompts) % len(image_start) != 0:
                        gr.Info("If there are more text prompts than input images the number of text prompts should be dividable by the number of images")
                        return ret()
                    rep = len(prompts) // len(image_start)
                    new_image_start = []
                    new_image_end = []
                    for i, _ in enumerate(prompts):
                        new_image_start.append(image_start[i//rep] )
                        if image_end != None:
                            new_image_end.append(image_end[i//rep] )
                    image_start = new_image_start 
                    if image_end != None:
                        image_end = new_image_end 
                else: 
                    if len(image_start) % len(prompts)  !=0:
                        gr.Info("If there are more input images than text prompts the number of images should be dividable by the number of text prompts")
                        return ret()
                    rep = len(image_start) // len(prompts)  
                    new_prompts = []
                    for i, _ in enumerate(image_start):
                        new_prompts.append(  prompts[ i//rep] )
                    prompts = new_prompts
            if image_end == None or len(image_end) == 0:
                image_end = [None] * len(prompts)

            for single_prompt, start, end in zip(prompts, image_start, image_end) :
                override_inputs.update({
                    "prompt" : single_prompt,
                    "image_start": start,
                    "image_end" : end,
                })
                inputs.update(override_inputs) 
                add_video_task(**inputs)
        else:
            for single_prompt in prompts :
                override_inputs["prompt"] = single_prompt 
                inputs.update(override_inputs) 
                add_video_task(**inputs)
        new_prompts_count = len(prompts)
    else:
        new_prompts_count = 1
        override_inputs["prompt"] = "\n".join(prompts)
        inputs.update(override_inputs) 
        add_video_task(**inputs)
    new_prompts_count += gen.get("prompts_max",0)
    gen["prompts_max"] = new_prompts_count
    state["validate_success"] = 1
    queue= gen.get("queue", [])
    first_time_in_queue = state.get("first_time_in_queue", True)
    state["first_time_in_queue"] = True
    return update_queue_data(queue, first_time_in_queue), gr.update(open=True) if new_prompts_count > 1 else gr.update()

def get_preview_images(inputs):
    inputs_to_query = ["image_start", "video_source", "image_end",  "video_guide", "image_guide", "video_mask", "image_mask", "image_refs" ]
    labels = ["Start Image", "Video Source", "End Image", "Video Guide", "Image Guide", "Video Mask", "Image Mask", "Image Reference"]
    start_image_data = None
    start_image_labels = []
    end_image_data = None
    end_image_labels = []
    for label, name in  zip(labels,inputs_to_query):
        image= inputs.get(name, None)
        if image is not None:
            image= [image] if not isinstance(image, list) else image.copy()
            if start_image_data == None:
                start_image_data = image
                start_image_labels += [label] * len(image)
            else:
                if end_image_data == None:
                    end_image_data = image
                else:
                    end_image_data += image 
                end_image_labels += [label] * len(image)

    if start_image_data != None and len(start_image_data) > 1 and  end_image_data  == None:
        end_image_data = start_image_data [1:]
        end_image_labels = start_image_labels [1:]
        start_image_data = start_image_data [:1] 
        start_image_labels = start_image_labels [:1] 
    return start_image_data, end_image_data, start_image_labels, end_image_labels 

def add_video_task(**inputs):
    global task_id
    state = inputs["state"]
    gen = get_gen_info(state)
    queue = gen["queue"]
    task_id += 1
    current_task_id = task_id

    start_image_data, end_image_data, start_image_labels, end_image_labels = get_preview_images(inputs)

    queue.append({
        "id": current_task_id,
        "params": inputs.copy(),
        "repeats": inputs["repeat_generation"],
        "length": inputs["video_length"], # !!!
        "steps": inputs["num_inference_steps"],
        "prompt": inputs["prompt"],
        "start_image_labels": start_image_labels,
        "end_image_labels": end_image_labels,
        "start_image_data": start_image_data,
        "end_image_data": end_image_data,
        "start_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in start_image_data] if start_image_data != None else None,
        "end_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in end_image_data] if end_image_data != None else None
    })

def update_task_thumbnails(task,  inputs):
    start_image_data, end_image_data, start_labels, end_labels = get_preview_images(inputs)

    task.update({
        "start_image_labels": start_labels,
        "end_image_labels": end_labels,
        "start_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in start_image_data] if start_image_data != None else None,
        "end_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in end_image_data] if end_image_data != None else None
    })

def move_up(queue, selected_indices):
    if not selected_indices or len(selected_indices) == 0:
        return update_queue_data(queue)
    idx = selected_indices[0]
    if isinstance(idx, list):
        idx = idx[0]
    idx = int(idx)
    with lock:
        idx += 1
        if idx > 1:
            queue[idx], queue[idx-1] = queue[idx-1], queue[idx]
        elif idx == 1:
            queue[:] = queue[0:1] + queue[2:] + queue[1:2]

    return update_queue_data(queue)

def move_down(queue, selected_indices):
    if not selected_indices or len(selected_indices) == 0:
        return update_queue_data(queue)
    idx = selected_indices[0]
    if isinstance(idx, list):
        idx = idx[0]
    idx = int(idx)
    with lock:
        idx += 1
        if idx < len(queue)-1:
            queue[idx], queue[idx+1] = queue[idx+1], queue[idx]
        elif idx == len(queue)-1:
            queue[:] = queue[0:1] + queue[-1:] + queue[1:-1]

    return update_queue_data(queue)

def remove_task(queue, selected_indices):
    if not selected_indices or len(selected_indices) == 0:
        return update_queue_data(queue)
    idx = selected_indices[0]
    if isinstance(idx, list):
        idx = idx[0]
    idx = int(idx) + 1
    with lock:
        if idx < len(queue):
            if idx == 0:
                wan_model._interrupt = True
            del queue[idx]
    return update_queue_data(queue)

def update_global_queue_ref(queue):
    global global_queue_ref
    with lock:
        global_queue_ref = queue[:]

def save_queue_action(state):
    gen = get_gen_info(state)
    queue = gen.get("queue", [])

    if not queue or len(queue) <=1 :
        gr.Info("Queue is empty. Nothing to save.")
        return ""

    zip_buffer = io.BytesIO()

    with tempfile.TemporaryDirectory() as tmpdir:
        queue_manifest = []
        file_paths_in_zip = {}

        for task_index, task in enumerate(queue):
            if task is None or not isinstance(task, dict) or task.get('id') is None: continue

            params_copy = task.get('params', {}).copy()
            task_id_s = task.get('id', f"task_{task_index}")

            image_keys = ["image_start", "image_end", "image_refs", "image_guide", "image_mask"]
            video_keys = ["video_guide", "video_mask", "video_source", "audio_guide", "audio_guide2", "audio_source"]

            for key in image_keys:
                images_pil = params_copy.get(key)
                if images_pil is None:
                    continue

                is_originally_list = isinstance(images_pil, list)
                if not is_originally_list:
                    images_pil = [images_pil]

                image_filenames_for_json = []
                for img_index, pil_image in enumerate(images_pil):
                    if not isinstance(pil_image, Image.Image):
                         print(f"Warning: Expected PIL Image for key '{key}' in task {task_id_s}, got {type(pil_image)}. Skipping image.")
                         continue

                    img_id = id(pil_image)
                    if img_id in file_paths_in_zip:
                         image_filenames_for_json.append(file_paths_in_zip[img_id])
                         continue

                    img_filename_in_zip = f"task{task_id_s}_{key}_{img_index}.png"
                    img_save_path = os.path.join(tmpdir, img_filename_in_zip)

                    try:
                        pil_image.save(img_save_path, "PNG")
                        image_filenames_for_json.append(img_filename_in_zip)
                        file_paths_in_zip[img_id] = img_filename_in_zip
                        print(f"Saved image: {img_filename_in_zip}")
                    except Exception as e:
                        print(f"Error saving image {img_filename_in_zip} for task {task_id_s}: {e}")

                if image_filenames_for_json:
                     params_copy[key] = image_filenames_for_json if is_originally_list else image_filenames_for_json[0]
                else:
                     pass
                    #  params_copy.pop(key, None) #cant pop otherwise crash during reload

            for key in video_keys:
                video_path_orig = params_copy.get(key)
                if video_path_orig is None or not isinstance(video_path_orig, str):
                    continue

                if video_path_orig in file_paths_in_zip:
                    params_copy[key] = file_paths_in_zip[video_path_orig]
                    continue

                if not os.path.isfile(video_path_orig):
                    print(f"Warning: Video file not found for key '{key}' in task {task_id_s}: {video_path_orig}. Skipping video.")
                    params_copy.pop(key, None)
                    continue

                _, extension = os.path.splitext(video_path_orig)
                vid_filename_in_zip = f"task{task_id_s}_{key}{extension if extension else '.mp4'}"
                vid_save_path = os.path.join(tmpdir, vid_filename_in_zip)

                try:
                    shutil.copy2(video_path_orig, vid_save_path)
                    params_copy[key] = vid_filename_in_zip
                    file_paths_in_zip[video_path_orig] = vid_filename_in_zip
                    print(f"Copied video: {video_path_orig} -> {vid_filename_in_zip}")
                except Exception as e:
                    print(f"Error copying video {video_path_orig} to {vid_filename_in_zip} for task {task_id_s}: {e}")
                    params_copy.pop(key, None)


            params_copy.pop('state', None)
            params_copy.pop('start_image_labels', None)
            params_copy.pop('end_image_labels', None)
            params_copy.pop('start_image_data_base64', None)
            params_copy.pop('end_image_data_base64', None)
            params_copy.pop('start_image_data', None)
            params_copy.pop('end_image_data', None)
            task.pop('start_image_data', None)
            task.pop('end_image_data', None)

            manifest_entry = {
                "id": task.get('id'),
                "params": params_copy,
            }
            manifest_entry = {k: v for k, v in manifest_entry.items() if v is not None}
            queue_manifest.append(manifest_entry)

        manifest_path = os.path.join(tmpdir, "queue.json")
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(queue_manifest, f, indent=4)
        except Exception as e:
            print(f"Error writing queue.json: {e}")
            gr.Warning("Failed to create queue manifest.")
            return None

        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(manifest_path, arcname="queue.json")

                for file_id, saved_file_rel_path in file_paths_in_zip.items():
                    saved_file_abs_path = os.path.join(tmpdir, saved_file_rel_path)
                    if os.path.exists(saved_file_abs_path):
                        zf.write(saved_file_abs_path, arcname=saved_file_rel_path)
                        print(f"Adding to zip: {saved_file_rel_path}")
                    else:
                        print(f"Warning: File {saved_file_rel_path} (ID: {file_id}) not found during zipping.")

            zip_buffer.seek(0)
            zip_binary_content = zip_buffer.getvalue()
            zip_base64 = base64.b64encode(zip_binary_content).decode('utf-8')
            print(f"Queue successfully prepared as base64 string ({len(zip_base64)} chars).")
            return zip_base64

        except Exception as e:
            print(f"Error creating zip file in memory: {e}")
            gr.Warning("Failed to create zip data for download.")
            return None
        finally:
            zip_buffer.close()

def load_queue_action(filepath, state, evt:gr.EventData):
    global task_id

    gen = get_gen_info(state)
    original_queue = gen.get("queue", [])
    delete_autoqueue_file  = False 
    if evt.target == None:

        if original_queue or not Path(AUTOSAVE_FILENAME).is_file():
            return
        print(f"Autoloading queue from {AUTOSAVE_FILENAME}...")
        filename = AUTOSAVE_FILENAME
        delete_autoqueue_file = True
    else:
        if not filepath or not hasattr(filepath, 'name') or not Path(filepath.name).is_file():
            print("[load_queue_action] Warning: No valid file selected or file not found.")
            return update_queue_data(original_queue)
        filename = filepath.name


    save_path_base = server_config.get("save_path", "outputs")
    loaded_cache_dir = os.path.join(save_path_base, "_loaded_queue_cache")


    newly_loaded_queue = []
    max_id_in_file = 0
    error_message = ""
    local_queue_copy_for_global_ref = None

    try:
        print(f"[load_queue_action] Attempting to load queue from: {filename}")
        os.makedirs(loaded_cache_dir, exist_ok=True)
        print(f"[load_queue_action] Using cache directory: {loaded_cache_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(filename, 'r') as zf:
                if "queue.json" not in zf.namelist(): raise ValueError("queue.json not found in zip file")
                print(f"[load_queue_action] Extracting {filename} to {tmpdir}")
                zf.extractall(tmpdir)
                print(f"[load_queue_action] Extraction complete.")

            manifest_path = os.path.join(tmpdir, "queue.json")
            print(f"[load_queue_action] Reading manifest: {manifest_path}")
            with open(manifest_path, 'r', encoding='utf-8') as f:
                loaded_manifest = json.load(f)
            print(f"[load_queue_action] Manifest loaded. Processing {len(loaded_manifest)} tasks.")

            for task_index, task_data in enumerate(loaded_manifest):
                if task_data is None or not isinstance(task_data, dict):
                    print(f"[load_queue_action] Skipping invalid task data at index {task_index}")
                    continue

                params = task_data.get('params', {})
                task_id_loaded = task_data.get('id', 0)
                max_id_in_file = max(max_id_in_file, task_id_loaded)
                params['state'] = state

                image_keys = ["image_start", "image_end", "image_refs", "image_guide", "image_mask"]
                video_keys = ["video_guide", "video_mask", "video_source", "audio_guide", "audio_guide2", "audio_source"]

                loaded_pil_images = {}
                loaded_video_paths = {}

                for key in image_keys:
                    image_filenames = params.get(key)
                    if image_filenames is None: continue

                    is_list = isinstance(image_filenames, list)
                    if not is_list: image_filenames = [image_filenames]

                    loaded_pils = []
                    for img_filename_in_zip in image_filenames:
                         if not isinstance(img_filename_in_zip, str):
                             print(f"[load_queue_action] Warning: Non-string filename found for image key '{key}'. Skipping.")
                             continue
                         img_load_path = os.path.join(tmpdir, img_filename_in_zip)
                         if not os.path.exists(img_load_path):
                             print(f"[load_queue_action] Image file not found in extracted data: {img_load_path}. Skipping.")
                             continue
                         try:
                             pil_image = Image.open(img_load_path)
                             pil_image.load()
                             converted_image = convert_image(pil_image)
                             loaded_pils.append(converted_image)
                             pil_image.close()
                             print(f"Loaded image: {img_filename_in_zip} for key {key}")
                         except Exception as img_e:
                             print(f"[load_queue_action] Error loading image {img_filename_in_zip}: {img_e}")
                    if loaded_pils:
                        params[key] = loaded_pils if is_list else loaded_pils[0]
                        loaded_pil_images[key] = params[key]
                    else:
                        params.pop(key, None)

                for key in video_keys:
                    video_filename_in_zip = params.get(key)
                    if video_filename_in_zip is None or not isinstance(video_filename_in_zip, str):
                        continue

                    video_load_path = os.path.join(tmpdir, video_filename_in_zip)
                    if not os.path.exists(video_load_path):
                        print(f"[load_queue_action] Video file not found in extracted data: {video_load_path}. Skipping.")
                        params.pop(key, None)
                        continue

                    persistent_video_path = os.path.join(loaded_cache_dir, video_filename_in_zip)
                    try:
                        shutil.copy2(video_load_path, persistent_video_path)
                        params[key] = persistent_video_path
                        loaded_video_paths[key] = persistent_video_path
                        print(f"Loaded video: {video_filename_in_zip} -> {persistent_video_path}")
                    except Exception as vid_e:
                        print(f"[load_queue_action] Error copying video {video_filename_in_zip} to cache: {vid_e}")
                        params.pop(key, None)

                primary_preview_pil_list, secondary_preview_pil_list, primary_preview_pil_labels, secondary_preview_pil_labels  = get_preview_images(params)

                start_b64 = [pil_to_base64_uri(primary_preview_pil_list[0], format="jpeg", quality=70)] if isinstance(primary_preview_pil_list, list) and primary_preview_pil_list else None
                end_b64 = [pil_to_base64_uri(secondary_preview_pil_list[0], format="jpeg", quality=70)] if isinstance(secondary_preview_pil_list, list) and secondary_preview_pil_list else None

                top_level_start_image = params.get("image_start") or params.get("image_refs")
                top_level_end_image = params.get("image_end")

                runtime_task = {
                    "id": task_id_loaded,
                    "params": params.copy(),
                    "repeats": params.get('repeat_generation', 1),
                    "length": params.get('video_length'),
                    "steps": params.get('num_inference_steps'),
                    "prompt": params.get('prompt'),
                    "start_image_labels": primary_preview_pil_labels,
                    "end_image_labels": secondary_preview_pil_labels,
                    "start_image_data": top_level_start_image,
                    "end_image_data": top_level_end_image,
                    "start_image_data_base64": start_b64,
                    "end_image_data_base64": end_b64,
                }
                newly_loaded_queue.append(runtime_task)
                print(f"[load_queue_action] Reconstructed task {task_index+1}/{len(loaded_manifest)}, ID: {task_id_loaded}")

        with lock:
            print("[load_queue_action] Acquiring lock to update state...")
            gen["queue"] = newly_loaded_queue[:]
            local_queue_copy_for_global_ref = gen["queue"][:]

            current_max_id_in_new_queue = max([t['id'] for t in newly_loaded_queue if 'id' in t] + [0])
            if current_max_id_in_new_queue >= task_id:
                 new_task_id = current_max_id_in_new_queue + 1
                 print(f"[load_queue_action] Updating global task_id from {task_id} to {new_task_id}")
                 task_id = new_task_id
            else:
                 print(f"[load_queue_action] Global task_id ({task_id}) is > max in file ({current_max_id_in_new_queue}). Not changing task_id.")

            gen["prompts_max"] = len(newly_loaded_queue)
            print("[load_queue_action] State update complete. Releasing lock.")

        if local_queue_copy_for_global_ref is not None:
             print("[load_queue_action] Updating global queue reference...")
             update_global_queue_ref(local_queue_copy_for_global_ref)
        else:
             print("[load_queue_action] Warning: Skipping global ref update as local copy is None.")

        print(f"[load_queue_action] Queue load successful. Returning DataFrame update for {len(newly_loaded_queue)} tasks.")
        return update_queue_data(newly_loaded_queue)

    except (ValueError, zipfile.BadZipFile, FileNotFoundError, Exception) as e:
        error_message = f"Error during queue load: {e}"
        print(f"[load_queue_action] Caught error: {error_message}")
        traceback.print_exc()
        gr.Warning(f"Failed to load queue: {error_message[:200]}")

        print("[load_queue_action] Load failed. Returning DataFrame update for original queue.")
        return update_queue_data(original_queue)
    finally:
        if delete_autoqueue_file:
            if os.path.isfile(filename):
                os.remove(filename)
                print(f"Clear Queue: Deleted autosave file '{filename}'.")

        if filepath and hasattr(filepath, 'name') and filepath.name and os.path.exists(filepath.name):
             if tempfile.gettempdir() in os.path.abspath(filepath.name):
                 try:
                     os.remove(filepath.name)
                     print(f"[load_queue_action] Removed temporary upload file: {filepath.name}")
                 except OSError as e:
                     print(f"[load_queue_action] Info: Could not remove temp file {filepath.name}: {e}")
             else:
                  print(f"[load_queue_action] Info: Did not remove non-temporary file: {filepath.name}")

def clear_queue_action(state):
    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    aborted_current = False
    cleared_pending = False

    with lock:
        if "in_progress" in gen and gen["in_progress"]:
            print("Clear Queue: Signalling abort for in-progress task.")
            gen["abort"] = True
            gen["extra_orders"] = 0
            if wan_model is not None:
                wan_model._interrupt = True
            aborted_current = True

        if queue:
             if len(queue) > 1 or (len(queue) == 1 and queue[0] is not None and queue[0].get('id') is not None):
                 print(f"Clear Queue: Clearing {len(queue)} tasks from queue.")
                 queue.clear()
                 cleared_pending = True
             else:
                 pass

        if aborted_current or cleared_pending:
            gen["prompts_max"] = 0

    if cleared_pending:
        try:
            if os.path.isfile(AUTOSAVE_FILENAME):
                os.remove(AUTOSAVE_FILENAME)
                print(f"Clear Queue: Deleted autosave file '{AUTOSAVE_FILENAME}'.")
        except OSError as e:
            print(f"Clear Queue: Error deleting autosave file '{AUTOSAVE_FILENAME}': {e}")
            gr.Warning(f"Could not delete the autosave file '{AUTOSAVE_FILENAME}'. You may need to remove it manually.")

    if aborted_current and cleared_pending:
        gr.Info("Queue cleared and current generation aborted.")
    elif aborted_current:
        gr.Info("Current generation aborted.")
    elif cleared_pending:
        gr.Info("Queue cleared.")
    else:
        gr.Info("Queue is already empty or only contains the active task (which wasn't aborted now).")

    return update_queue_data([])

def quit_application():
    print("Save and Quit requested...")
    autosave_queue()
    import signal
    os.kill(os.getpid(), signal.SIGINT)

def start_quit_process():
    return 5, gr.update(visible=False), gr.update(visible=True)

def cancel_quit_process():
    return -1, gr.update(visible=True), gr.update(visible=False)

def show_countdown_info_from_state(current_value: int):
    if current_value > 0:
        gr.Info(f"Quitting in {current_value}...")
        return current_value - 1
    return current_value
quitting_app = False
def autosave_queue():
    global quitting_app
    quitting_app = True
    global global_queue_ref
    if not global_queue_ref:
        print("Autosave: Queue is empty, nothing to save.")
        return

    print(f"Autosaving queue ({len(global_queue_ref)} items) to {AUTOSAVE_FILENAME}...")
    temp_state_for_save = {"gen": {"queue": global_queue_ref}}
    zip_file_path = None
    try:

        def _save_queue_to_file(queue_to_save, output_filename):
             if not queue_to_save: return None

             with tempfile.TemporaryDirectory() as tmpdir:
                queue_manifest = []
                file_paths_in_zip = {}

                for task_index, task in enumerate(queue_to_save):
                    if task is None or not isinstance(task, dict) or task.get('id') is None: continue

                    params_copy = task.get('params', {}).copy()
                    task_id_s = task.get('id', f"task_{task_index}")

                    image_keys = ["image_start", "image_end", "image_refs", "image_guide", "image_mask"]
                    video_keys = ["video_guide", "video_mask", "video_source", "audio_guide", "audio_guide2", "audio_source" ]

                    for key in image_keys:
                        images_pil = params_copy.get(key)
                        if images_pil is None: continue
                        is_list = isinstance(images_pil, list)
                        if not is_list: images_pil = [images_pil]
                        image_filenames_for_json = []
                        for img_index, pil_image in enumerate(images_pil):
                            if not isinstance(pil_image, Image.Image): continue
                            img_id = id(pil_image)
                            if img_id in file_paths_in_zip:
                                image_filenames_for_json.append(file_paths_in_zip[img_id])
                                continue
                            img_filename_in_zip = f"task{task_id_s}_{key}_{img_index}.png"
                            img_save_path = os.path.join(tmpdir, img_filename_in_zip)
                            try:
                                pil_image.save(img_save_path, "PNG")
                                image_filenames_for_json.append(img_filename_in_zip)
                                file_paths_in_zip[img_id] = img_filename_in_zip
                            except Exception as e:
                                print(f"Autosave error saving image {img_filename_in_zip}: {e}")
                        if image_filenames_for_json:
                            params_copy[key] = image_filenames_for_json if is_list else image_filenames_for_json[0]
                        else:
                            params_copy.pop(key, None)

                    for key in video_keys:
                        video_path_orig = params_copy.get(key)
                        if video_path_orig is None or not isinstance(video_path_orig, str):
                            continue

                        if video_path_orig in file_paths_in_zip:
                            params_copy[key] = file_paths_in_zip[video_path_orig]
                            continue

                        if not os.path.isfile(video_path_orig):
                            print(f"Warning (Autosave): Video file not found for key '{key}' in task {task_id_s}: {video_path_orig}. Skipping.")
                            params_copy.pop(key, None)
                            continue

                        _, extension = os.path.splitext(video_path_orig)
                        vid_filename_in_zip = f"task{task_id_s}_{key}{extension if extension else '.mp4'}"
                        vid_save_path = os.path.join(tmpdir, vid_filename_in_zip)

                        try:
                            shutil.copy2(video_path_orig, vid_save_path)
                            params_copy[key] = vid_filename_in_zip
                            file_paths_in_zip[video_path_orig] = vid_filename_in_zip
                        except Exception as e:
                            print(f"Error (Autosave) copying video {video_path_orig} to {vid_filename_in_zip} for task {task_id_s}: {e}")
                            params_copy.pop(key, None)
                    params_copy.pop('state', None)
                    params_copy.pop('start_image_data_base64', None)
                    params_copy.pop('end_image_data_base64', None)
                    params_copy.pop('start_image_data', None)
                    params_copy.pop('end_image_data', None)

                    manifest_entry = {
                        "id": task.get('id'),
                        "params": params_copy,
                    }
                    manifest_entry = {k: v for k, v in manifest_entry.items() if v is not None}
                    queue_manifest.append(manifest_entry)

                manifest_path = os.path.join(tmpdir, "queue.json")
                with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(queue_manifest, f, indent=4)
                with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(manifest_path, arcname="queue.json")
                    for saved_file_rel_path in file_paths_in_zip.values():
                        saved_file_abs_path = os.path.join(tmpdir, saved_file_rel_path)
                        if os.path.exists(saved_file_abs_path):
                             zf.write(saved_file_abs_path, arcname=saved_file_rel_path)
                        else:
                             print(f"Warning (Autosave): File {saved_file_rel_path} not found during zipping.")
                return output_filename
             return None

        saved_path = _save_queue_to_file(global_queue_ref, AUTOSAVE_FILENAME)

        if saved_path:
            print(f"Queue autosaved successfully to {saved_path}")
        else:
            print("Autosave failed.")
    except Exception as e:
        print(f"Error during autosave: {e}")
        traceback.print_exc()

def finalize_generation_with_state(current_state):
     if not isinstance(current_state, dict) or 'gen' not in current_state:
         return gr.update(), gr.update(interactive=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, value=""), gr.update(), current_state

     gallery_update, abort_btn_update, gen_btn_update, add_queue_btn_update, current_gen_col_update, gen_info_update = finalize_generation(current_state)
     accordion_update = gr.Accordion(open=False) if len(get_gen_info(current_state).get("queue", [])) <= 1 else gr.update()
     return gallery_update, abort_btn_update, gen_btn_update, add_queue_btn_update, current_gen_col_update, gen_info_update, accordion_update, current_state

def generate_queue_html(queue):
    if len(queue) <= 1:
        return "<div style='text-align: center; color: grey; padding: 20px;'>Queue is empty.</div>"

    table_header = """
    <table>
        <thead>
            <tr>
                <th style="width:5%;" class="center-align">Qty</th>
                <th style="width:auto;" class="text-left">Prompt</th>
                <th style="width:7%;" class="center-align">Length</th>
                <th style="width:7%;" class="center-align">Steps</th>
                <th style="width:10%;" class="center-align">Start/Ref</th>
                <th style="width:10%;" class="center-align">End</th>
                <th style="width:4%;" class="center-align" title="Move Up"></th>
                <th style="width:4%;" class="center-align" title="Move Down"></th>
                <th style="width:4%;" class="center-align" title="Edit"></th>
                <th style="width:4%;" class="center-align" title="Remove"></th>
            </tr>
        </thead>
        <tbody>
    """
    
    table_rows = []
    for i, item in enumerate(queue):
        if i == 0:
            continue
        
        row_index = i - 1
        truncated_prompt = (item['prompt'][:97] + '...') if len(item['prompt']) > 100 else item['prompt']
        full_prompt = item['prompt'].replace('"', '&quot;')
        prompt_cell = f'<div class="prompt-cell" title="{full_prompt}">{truncated_prompt}</div>'
        
        start_img_data = item.get('start_image_data_base64') or [None]
        start_img_uri = start_img_data[0]
        start_img_labels = item.get('start_image_labels', [''])
        
        end_img_data = item.get('end_image_data_base64') or [None]
        end_img_uri = end_img_data[0]
        end_img_labels = item.get('end_image_labels', [''])
        
        num_steps = item.get('steps')
        length = item.get('length')
        
        start_img_md = ""
        if start_img_uri:
            start_img_md = f'<div class="hover-image" onclick="showImageModal(\'start_{row_index}\')"><img src="{start_img_uri}" alt="{start_img_labels[0]}" /></div>'
            
        end_img_md = ""
        if end_img_uri:
            end_img_md = f'<div class="hover-image" onclick="showImageModal(\'end_{row_index}\')"><img src="{end_img_uri}" alt="{end_img_labels[0]}" /></div>'
        
        up_btn = f"""<button onclick="updateAndTrigger('up_{row_index}')" class="action-button" title="Move Up">↑</button>"""
        down_btn = f"""<button onclick="updateAndTrigger('down_{row_index}')" class="action-button" title="Move Down">↓</button>"""
        edit_btn = f"""<button onclick="updateAndTrigger('edit_{row_index}')" class="action-button" title="Edit">✏️</button>"""
        remove_btn = f"""<button onclick="updateAndTrigger('remove_{row_index}')" class="action-button" title="Remove">✖️</button>"""

        row_html = f"""
        <tr>
            <td class="center-align">{item.get('repeats', "1")}</td>
            <td>{prompt_cell}</td>
            <td class="center-align">{length}</td>
            <td class="center-align">{num_steps}</td>
            <td class="center-align">{start_img_md}</td>
            <td class="center-align">{end_img_md}</td>
            <td class="center-align">{up_btn}</td>
            <td class="center-align">{down_btn}</td>
            <td class="center-align">{edit_btn}</td>
            <td class="center-align">{remove_btn}</td>
        </tr>
        """
        table_rows.append(row_html)
        
    table_footer = "</tbody></table>"
    
    return table_header + "".join(table_rows) + table_footer

def update_queue_data(queue, first_time_in_queue =False):
    update_global_queue_ref(queue)
    html_content = generate_queue_html(queue)
    return gr.HTML(value=html_content)


def create_html_progress_bar(percentage=0.0, text="Idle", is_idle=True):
    bar_class = "progress-bar-custom idle" if is_idle else "progress-bar-custom"
    bar_text_html = f'<div class="progress-bar-text">{text}</div>'

    html = f"""
    <div class="progress-container-custom">
        <div class="{bar_class}" style="width: {percentage:.1f}%;" role="progressbar" aria-valuenow="{percentage:.1f}" aria-valuemin="0" aria-valuemax="100">
           {bar_text_html}
        </div>
    </div>
    """
    return html

def update_generation_status(html_content):
    if(html_content):
        return gr.update(value=html_content)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt or image using Gradio")

    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="save proprocessed masks for debugging or editing"
    )

    parser.add_argument(
        "--save-speakers",
        action="store_true",
        help="save proprocessed audio track with extract speakers for debugging or editing"
    )

    parser.add_argument(
        "--debug-gen-form",
        action="store_true",
        help="View form generation / refresh time"
    )

    parser.add_argument(
        "--vram-safety-coefficient",
        type=float,
        default=0.8,
        help="max VRAM (between 0 and 1) that should be allocated to preloaded models"
    )


    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shared URL to access webserver remotely"
    )

    parser.add_argument(
        "--lock-config",
        action="store_true",
        help="Prevent modifying the configuration from the web interface"
    )

    parser.add_argument(
        "--lock-model",
        action="store_true",
        help="Prevent switch models"
    )

    parser.add_argument(
        "--save-quantized",
        action="store_true",
        help="Save a quantized version of the current model"
    )

    parser.add_argument(
        "--preload",
        type=str,
        default="0",
        help="Megabytes of the diffusion model to preload in VRAM"
    )

    parser.add_argument(
        "--multiple-images",
        action="store_true",
        help="Allow inputting multiple images with image to video"
    )


    parser.add_argument(
        "--lora-dir-i2v",
        type=str,
        default="",
        help="Path to a directory that contains Wan i2v Loras "
    )


    parser.add_argument(
        "--lora-dir",
        type=str,
        default="", 
        help="Path to a directory that contains Wan t2v Loras"
    )

    parser.add_argument(
        "--lora-dir-hunyuan",
        type=str,
        default="loras_hunyuan", 
        help="Path to a directory that contains Hunyuan Video t2v Loras"
    )

    parser.add_argument(
        "--lora-dir-hunyuan-i2v",
        type=str,
        default="loras_hunyuan_i2v", 
        help="Path to a directory that contains Hunyuan Video i2v Loras"
    )


    parser.add_argument(
        "--lora-dir-ltxv",
        type=str,
        default="loras_ltxv", 
        help="Path to a directory that contains LTX Videos Loras"
    )

    parser.add_argument(
        "--lora-dir-flux",
        type=str,
        default="loras_flux", 
        help="Path to a directory that contains flux images Loras"
    )

    parser.add_argument(
        "--lora-dir-qwen",
        type=str,
        default="loras_qwen", 
        help="Path to a directory that contains qwen images Loras"
    )

    parser.add_argument(
        "--check-loras",
        action="store_true",
        help="Filter Loras that are not valid"
    )


    parser.add_argument(
        "--lora-preset",
        type=str,
        default="",
        help="Lora preset to preload"
    )

    parser.add_argument(
        "--settings",
        type=str,
        default="settings",
        help="Path to settings folder"
    )


    # parser.add_argument(
    #     "--lora-preset-i2v",
    #     type=str,
    #     default="",
    #     help="Lora preset to preload for i2v"
    # )

    parser.add_argument(
        "--profile",
        type=str,
        default=-1,
        help="Profile No"
    )

    parser.add_argument(
        "--verbose",
        type=str,
        default=1,
        help="Verbose level"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="default denoising steps"
    )


    # parser.add_argument(
    #     "--teacache",
    #     type=float,
    #     default=-1,
    #     help="teacache speed multiplier"
    # )

    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="default number of frames"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="default generation seed"
    )

    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Access advanced options by default"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="For using fp16 transformer model"
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
        help="For using bf16 transformer model"
    )

    parser.add_argument(
        "--server-port",
        type=str,
        default=0,
        help="Server port"
    )

    parser.add_argument(
        "--theme",
        type=str,
        default="",
        help="set UI Theme"
    )

    parser.add_argument(
        "--perc-reserved-mem-max",
        type=float,
        default=0,
        help="percent of RAM allocated to Reserved RAM"
    )



    parser.add_argument(
        "--server-name",
        type=str,
        default="",
        help="Server name"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="",
        help="Default GPU Device"
    )

    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open browser"
    )

    parser.add_argument(
        "--t2v",
        action="store_true",
        help="text to video mode"
    )

    parser.add_argument(
        "--i2v",
        action="store_true",
        help="image to video mode"
    )

    parser.add_argument(
        "--t2v-14B",
        action="store_true",
        help="text to video mode 14B model"
    )

    parser.add_argument(
        "--t2v-1-3B",
        action="store_true",
        help="text to video mode 1.3B model"
    )

    parser.add_argument(
        "--vace-1-3B",
        action="store_true",
        help="Vace ControlNet 1.3B model"
    )    
    parser.add_argument(
        "--i2v-1-3B",
        action="store_true",
        help="Fun InP image to video mode 1.3B model"
    )

    parser.add_argument(
        "--i2v-14B",
        action="store_true",
        help="image to video mode 14B model"
    )


    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable pytorch compilation"
    )

    parser.add_argument(
        "--listen",
        action="store_true",
        help="Server accessible on local network"
    )

    # parser.add_argument(
    #     "--fast",
    #     action="store_true",
    #     help="use Fast model"
    # )

    # parser.add_argument(
    #     "--fastest",
    #     action="store_true",
    #     help="activate the best config"
    # )

    parser.add_argument(
    "--attention",
    type=str,
    default="",
    help="attention mode"
    )

    parser.add_argument(
    "--vae-config",
    type=str,
    default="",
    help="vae config mode"
    )    

    args = parser.parse_args()

    return args

def get_lora_dir(model_type):
    model_family = get_model_family(model_type)
    base_model_type = get_base_model_type(model_type)
    i2v = test_class_i2v(model_type) and not  base_model_type in ["i2v_2_2", "i2v_2_2_multitalk"]
    if model_family == "wan":
        lora_dir =args.lora_dir
        if i2v and len(lora_dir)==0:
            lora_dir =args.lora_dir_i2v
        if len(lora_dir) > 0:
            return lora_dir
        root_lora_dir = "loras_i2v" if i2v else "loras"

        if  "1.3B" in model_type :
            lora_dir_1_3B = os.path.join(root_lora_dir, "1.3B")
            if os.path.isdir(lora_dir_1_3B ):
                return lora_dir_1_3B
        elif base_model_type == "ti2v_2_2":
            lora_dir_5B = os.path.join(root_lora_dir, "5B")
            if os.path.isdir(lora_dir_5B ):
                return lora_dir_5B
        else:
            lora_dir_14B = os.path.join(root_lora_dir, "14B")
            if os.path.isdir(lora_dir_14B ):
                return lora_dir_14B
        return root_lora_dir    
    elif model_family == "ltxv":
            return args.lora_dir_ltxv
    elif model_family == "flux":
            return args.lora_dir_flux
    elif model_family =="hunyuan":
        if i2v:
            return args.lora_dir_hunyuan_i2v
        else:
            return args.lora_dir_hunyuan
    elif model_family =="qwen":
            return args.lora_dir_qwen
    else:
        raise Exception("loras unknown")

attention_modes_installed = get_attention_modes()
attention_modes_supported = get_supported_attention_modes()
args = _parse_args()

gpu_major, gpu_minor = torch.cuda.get_device_capability(args.gpu if len(args.gpu) > 0 else None)
if  gpu_major < 8:
    print("Switching to FP16 models when possible as GPU architecture doesn't support optimed BF16 Kernels")
    bfloat16_supported = False
else:
    bfloat16_supported = True

args.flow_reverse = True
processing_device = args.gpu
if len(processing_device) == 0:
    processing_device ="cuda"
# torch.backends.cuda.matmul.allow_fp16_accumulation = True
lock_ui_attention = False
lock_ui_transformer = False
lock_ui_compile = False

force_profile_no = int(args.profile)
verbose_level = int(args.verbose)
check_loras = args.check_loras ==1

server_config_filename = "wgp_config.json"
if not os.path.isdir("settings"):
    os.mkdir("settings") 
if os.path.isfile("t2v_settings.json"):
    for f in glob.glob(os.path.join(".", "*_settings.json*")):
        target_file = os.path.join("settings",  Path(f).parts[-1] )
        shutil.move(f, target_file) 

if not os.path.isfile(server_config_filename) and os.path.isfile("gradio_config.json"):
    shutil.move("gradio_config.json", server_config_filename) 

if not os.path.isdir("ckpts/umt5-xxl/"):
    os.makedirs("ckpts/umt5-xxl/")
src_move = [ "ckpts/models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors", "ckpts/models_t5_umt5-xxl-enc-bf16.safetensors", "ckpts/models_t5_umt5-xxl-enc-quanto_int8.safetensors" ]
tgt_move = [ "ckpts/xlm-roberta-large/", "ckpts/umt5-xxl/", "ckpts/umt5-xxl/"]
for src,tgt in zip(src_move,tgt_move):
    if os.path.isfile(src):
        try:
            if os.path.isfile(tgt):
                shutil.remove(src)
            else:
                shutil.move(src, tgt)
        except:
            pass
    

if not Path(server_config_filename).is_file():
    server_config = {
        "attention_mode" : "auto",  
        "transformer_types": [], 
        "transformer_quantization": "int8",
        "text_encoder_quantization" : "int8",
        "save_path": "outputs",  
        "image_save_path": "outputs",  
        "compile" : "",
        "metadata_type": "metadata",
        "boost" : 1,
        "clear_file_list" : 5,
        "vae_config": 0,
        "profile" : profile_type.LowRAM_LowVRAM,
        "preload_model_policy": [],
        "UI_theme": "default"
    }

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))
else:
    with open(server_config_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    server_config = json.loads(text)

#   Deprecated models
for path in  ["wan2.1_Vace_1.3B_preview_bf16.safetensors", "sky_reels2_diffusion_forcing_1.3B_bf16.safetensors","sky_reels2_diffusion_forcing_720p_14B_bf16.safetensors",
"sky_reels2_diffusion_forcing_720p_14B_quanto_int8.safetensors", "sky_reels2_diffusion_forcing_720p_14B_quanto_fp16_int8.safetensors", "wan2.1_image2video_480p_14B_bf16.safetensors", "wan2.1_image2video_480p_14B_quanto_int8.safetensors",
"wan2.1_image2video_720p_14B_quanto_int8.safetensors", "wan2.1_image2video_720p_14B_quanto_fp16_int8.safetensors", "wan2.1_image2video_720p_14B_bf16.safetensors",
"wan2.1_text2video_14B_bf16.safetensors", "wan2.1_text2video_14B_quanto_int8.safetensors",
"wan2.1_Vace_14B_mbf16.safetensors", "wan2.1_Vace_14B_quanto_mbf16_int8.safetensors", "wan2.1_FLF2V_720p_14B_quanto_int8.safetensors", "wan2.1_FLF2V_720p_14B_bf16.safetensors",  "wan2.1_FLF2V_720p_14B_fp16.safetensors", "wan2.1_Vace_1.3B_mbf16.safetensors", "wan2.1_text2video_1.3B_bf16.safetensors",
"ltxv_0.9.7_13B_dev_bf16.safetensors"
]:
    if Path(os.path.join("ckpts" , path)).is_file():
        print(f"Removing old version of model '{path}'. A new version of this model will be downloaded next time you use it.")
        os.remove( os.path.join("ckpts" , path))

for f, s in [("ckpts/Florence2/modeling_florence2.py", 127287)]:
    try:
        if os.path.isfile(f) and os.path.getsize(f) == s:
            print(f"Removing old version of model '{f}'. A new version of this model will be downloaded next time you use it.")
            os.remove(f)
    except: pass

models_def = {}
family_handlers = ["models.wan.wan_handler", "models.wan.df_handler", "models.hyvideo.hunyuan_handler", "models.ltx_video.ltxv_handler", "models.flux.flux_handler", "models.qwen.qwen_handler"]


# only needed for imported old settings files
model_signatures = {"t2v": "text2video_14B", "t2v_1.3B" : "text2video_1.3B",   "fun_inp_1.3B" : "Fun_InP_1.3B",  "fun_inp" :  "Fun_InP_14B", 
                    "i2v" : "image2video_480p", "i2v_720p" : "image2video_720p" , "vace_1.3B" : "Vace_1.3B", "vace_14B": "Vace_14B", "recam_1.3B": "recammaster_1.3B", 
                    "sky_df_1.3B" : "sky_reels2_diffusion_forcing_1.3B", "sky_df_14B" : "sky_reels2_diffusion_forcing_14B", 
                    "sky_df_720p_14B" : "sky_reels2_diffusion_forcing_720p_14B",
                    "phantom_1.3B" : "phantom_1.3B", "phantom_14B" : "phantom_14B", "ltxv_13B" : "ltxv_0.9.7_13B_dev", "ltxv_13B_distilled" : "ltxv_0.9.7_13B_distilled", 
                    "hunyuan" : "hunyuan_video_720", "hunyuan_i2v" : "hunyuan_video_i2v_720", "hunyuan_custom" : "hunyuan_video_custom_720", "hunyuan_custom_audio" : "hunyuan_video_custom_audio", "hunyuan_custom_edit" : "hunyuan_video_custom_edit",
                    "hunyuan_avatar" : "hunyuan_video_avatar"  }


def map_family_handlers(family_handlers):
    base_types_handlers, families_infos, models_eqv_map, models_comp_map = {}, {"unknown": (100, "Unknown")}, {}, {}
    for path in family_handlers:
        handler = importlib.import_module(path).family_handler
        for model_type in handler.query_supported_types():
            if model_type in base_types_handlers:
                prev = base_types_handlers[model_type].__name__
                raise Exception(f"Model type {model_type} supported by {prev} and {handler.__name__}")
            base_types_handlers[model_type] = handler
        families_infos.update(handler.query_family_infos())
        eq_map, comp_map = handler.query_family_maps()
        models_eqv_map.update(eq_map); models_comp_map.update(comp_map)
    return base_types_handlers, families_infos, models_eqv_map, models_comp_map

model_types_handlers, families_infos,  models_eqv_map, models_comp_map = map_family_handlers(family_handlers)

def get_base_model_type(model_type):
    model_def = get_model_def(model_type)
    if model_def == None:
        return model_type if model_type in model_types_handlers else None 
        # return model_type
    else:
        return model_def["architecture"]

def get_model_handler(model_type):
    base_model_type = get_base_model_type(model_type)
    if base_model_type is None:
        raise Exception(f"Unknown model type {model_type}")
    model_handler = model_types_handlers.get(base_model_type, None)
    if model_handler is None:
        raise Exception(f"No model handler found for base model type {base_model_type}")
    return model_handler

def are_model_types_compatible(imported_model_type, current_model_type):
    imported_base_model_type = get_base_model_type(imported_model_type)
    curent_base_model_type = get_base_model_type(current_model_type)
    if imported_base_model_type == curent_base_model_type:
        return True

    if imported_base_model_type in models_eqv_map:
        imported_base_model_type = models_eqv_map[imported_base_model_type]

    comp_list=  models_comp_map.get(imported_base_model_type, None)
    if comp_list == None: return False
    return curent_base_model_type in comp_list 

def get_model_def(model_type):
    return models_def.get(model_type, None )



def get_model_type(model_filename):
    for model_type, signature in model_signatures.items():
        if signature in model_filename:
            return model_type
    return None
    # raise Exception("Unknown model:" + model_filename)

def get_model_family(model_type, for_ui = False):
    base_model_type = get_base_model_type(model_type)
    if base_model_type is None:
        return "unknown"
    
    if for_ui : 
        model_def = get_model_def(model_type)
        model_family = model_def.get("group", None)
        if model_family is not None and model_family in families_infos:
            return model_family
    handler = model_types_handlers.get(base_model_type, None)
    if handler is None: 
        return "unknown"
    return handler.query_model_family()

def test_class_i2v(model_type):    
    model_def = get_model_def(model_type)
    return model_def.get("i2v_class", False)

def test_vace_module(model_type):
    model_def = get_model_def(model_type)
    return model_def.get("vace_class", False)

def test_any_sliding_window(model_type):
    model_def = get_model_def(model_type)
    return model_def.get("sliding_window", False)

def get_model_min_frames_and_step(model_type):
    mode_def = get_model_def(model_type)
    frames_minimum = mode_def.get("frames_minimum", 5)
    frames_steps = mode_def.get("frames_steps", 4)
    latent_size = mode_def.get("latent_size", frames_steps)
    return frames_minimum, frames_steps, latent_size 
    
def get_model_fps(model_type):
    mode_def = get_model_def(model_type)
    fps= mode_def.get("fps", 16)
    return fps

def get_computed_fps(force_fps, base_model_type , video_guide, video_source ):
    if force_fps == "auto":
        if video_source != None:
            fps,  _, _, _ = get_video_info(video_source)
        elif video_guide != None:
            fps,  _, _, _ = get_video_info(video_guide)
        else:
            fps = get_model_fps(base_model_type)
    elif force_fps == "control" and video_guide != None:
        fps,  _, _, _ = get_video_info(video_guide)
    elif force_fps == "source" and video_source != None:
        fps,  _, _, _ = get_video_info(video_source)
    elif len(force_fps) > 0 and is_integer(force_fps) :
        fps = int(force_fps)
    else:
        fps = get_model_fps(base_model_type)
    return fps

def get_model_name(model_type, description_container = [""]):
    model_def = get_model_def(model_type)
    if model_def == None: 
        return f"Unknown model {model_type}"
    model_name = model_def["name"]
    description = model_def["description"]
    description_container[0] = description
    return model_name

def get_model_record(model_name):
    return f"WanGP v{WanGP_version} by DeepBeepMeep - " +  model_name

def get_model_recursive_prop(model_type, prop = "URLs", sub_prop_name = None, return_list = True,  stack= []):
    model_def = models_def.get(model_type, None)
    if model_def != None: 
        prop_value = model_def.get(prop, None)
        if prop_value == None:
            return []
        if sub_prop_name is not None:
            if sub_prop_name == "_list":
                if not isinstance(prop_value,list) or len(prop_value) != 1:
                    raise Exception(f"Sub property value for property {prop} of model type {model_type} should be a list of size 1")
                prop_value = prop_value[0]
            else:
                if not isinstance(prop_value,dict) and not sub_prop_name in prop_value:
                    raise Exception(f"Invalid sub property value {sub_prop_name} for property {prop} of model type {model_type}")
                prop_value = prop_value[sub_prop_name]
        if isinstance(prop_value, str):
            if len(stack) > 10: raise Exception(f"Circular Reference in Model {prop} dependencies: {stack}")
            return get_model_recursive_prop(prop_value, prop = prop, sub_prop_name =sub_prop_name, stack = stack + [prop_value] )
        else:
            return prop_value
    else:
        if model_type in model_types:
            return [] if return_list else model_type 
        else:
            raise Exception(f"Unknown model type '{model_type}'")
        

def get_model_filename(model_type, quantization ="int8", dtype_policy = "", module_type = None, submodel_no = 1, URLs = None, stack=[]):
    if URLs is not None:
        pass
    elif module_type is not None:
        base_model_type = get_base_model_type(model_type) 
        # model_type_handler = model_types_handlers[base_model_type]
        # modules_files = model_type_handler.query_modules_files() if hasattr(model_type_handler, "query_modules_files") else {}
        if isinstance(module_type, list):
            URLs = module_type
        else:
            if "#" not in module_type:
                sub_prop_name = "_list"
            else:
                pos = module_type.rfind("#")
                sub_prop_name =  module_type[pos+1:]
                module_type = module_type[:pos]  
            URLs = get_model_recursive_prop(module_type, "modules", sub_prop_name =sub_prop_name, return_list= False)

        # choices = modules_files.get(module_type, None)
        # if choices == None: raise Exception(f"Invalid Module Id '{module_type}'")
    else:
        key_name = "URLs" if submodel_no  <= 1 else f"URLs{submodel_no}"

        model_def = models_def.get(model_type, None)
        if model_def == None: return ""
        URLs = model_def[key_name]
        if isinstance(URLs, str):
            if len(stack) > 10: raise Exception(f"Circular Reference in Model {key_name} dependencies: {stack}")
            return get_model_filename(URLs, quantization=quantization, dtype_policy=dtype_policy, submodel_no = submodel_no, stack = stack + [URLs])

    # choices = [ ("ckpts/" + os.path.basename(path) if path.startswith("http") else path)  for path in URLs ]
    choices = URLs
    if len(quantization) == 0:
        quantization = "bf16"

    model_family =  get_model_family(model_type) 
    dtype = get_transformer_dtype(model_family, dtype_policy)
    if len(choices) <= 1:
        raw_filename = choices[0]
    else:
        if quantization in ("int8", "fp8"):
            sub_choices = [ name for name in choices if quantization in os.path.basename(name) or quantization.upper() in os.path.basename(name)]
        else:
            sub_choices = [ name for name in choices if "quanto" not in os.path.basename(name)]

        if len(sub_choices) > 0:
            dtype_str = "fp16" if dtype == torch.float16 else "bf16"
            new_sub_choices = [ name for name in sub_choices if dtype_str in os.path.basename(name) or dtype_str.upper() in os.path.basename(name)]
            sub_choices = new_sub_choices if len(new_sub_choices) > 0 else sub_choices
            raw_filename = sub_choices[0]
        else:
            raw_filename = choices[0]

    return raw_filename

def get_transformer_dtype(model_family, transformer_dtype_policy):
    if not isinstance(transformer_dtype_policy, str):
        return transformer_dtype_policy
    if len(transformer_dtype_policy) == 0:
        if not bfloat16_supported:
            return torch.float16
        else:
            if model_family == "wan"and False:
                return torch.float16
            else: 
                return torch.bfloat16
        return transformer_dtype
    elif transformer_dtype_policy =="fp16":
        return torch.float16
    else:
        return torch.bfloat16

def get_settings_file_name(model_type):
    return  os.path.join(args.settings, model_type + "_settings.json")

def fix_settings(model_type, ui_defaults):
    if model_type is None: return

    settings_version =  ui_defaults.get("settings_version", 0)
    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)

    prompts = ui_defaults.get("prompts", "")
    if len(prompts) > 0:
        ui_defaults["prompt"] = prompts
    image_prompt_type = ui_defaults.get("image_prompt_type", None)
    if image_prompt_type != None :
        if not isinstance(image_prompt_type, str):
            image_prompt_type = "S" if image_prompt_type  == 0 else "SE"
        if settings_version <= 2:
            image_prompt_type = image_prompt_type.replace("G","")
        ui_defaults["image_prompt_type"] = image_prompt_type

    if "lset_name" in ui_defaults: del ui_defaults["lset_name"]

    audio_prompt_type = ui_defaults.get("audio_prompt_type", None)
    if settings_version < 2.2: 
        if not base_model_type in ["vace_1.3B","vace_14B", "sky_df_1.3B", "sky_df_14B", "ltxv_13B"]:
            for p in  ["sliding_window_size", "sliding_window_overlap", "sliding_window_overlap_noise", "sliding_window_discard_last_frames"]:
                if p in ui_defaults: del ui_defaults[p]

        if audio_prompt_type == None :
            if any_audio_track(base_model_type):
                audio_prompt_type ="A"
                ui_defaults["audio_prompt_type"] = audio_prompt_type

    if settings_version < 2.35 and any_audio_track(base_model_type): 
        audio_prompt_type = audio_prompt_type or ""
        audio_prompt_type += "V"
        ui_defaults["audio_prompt_type"] = audio_prompt_type

    video_prompt_type = ui_defaults.get("video_prompt_type", "")

    if base_model_type in ["hunyuan"]:
        video_prompt_type = video_prompt_type.replace("I", "")

    if base_model_type in ["flux"] and settings_version < 2.23:
        video_prompt_type = video_prompt_type.replace("K", "").replace("I", "KI")

    remove_background_images_ref = ui_defaults.get("remove_background_images_ref", None)
    if settings_version < 2.22:
        if "I" in video_prompt_type:
            if remove_background_images_ref == 2:
                video_prompt_type = video_prompt_type.replace("I", "KI")
        if remove_background_images_ref != 0:
            remove_background_images_ref = 1
    if base_model_type in ["hunyuan_avatar"]: 
        remove_background_images_ref = 0
        if settings_version < 2.26:
            if not "K" in video_prompt_type: video_prompt_type = video_prompt_type.replace("I", "KI")
    if remove_background_images_ref is not None:
        ui_defaults["remove_background_images_ref"] = remove_background_images_ref

    ui_defaults["video_prompt_type"] = video_prompt_type

    tea_cache_setting = ui_defaults.get("tea_cache_setting", None)
    tea_cache_start_step_perc = ui_defaults.get("tea_cache_start_step_perc", None)

    if tea_cache_setting != None:
        del ui_defaults["tea_cache_setting"]
        if tea_cache_setting > 0:
            ui_defaults["skip_steps_multiplier"] = tea_cache_setting
            ui_defaults["skip_steps_cache_type"] = "tea"
        else:
            ui_defaults["skip_steps_multiplier"] = 1.75
            ui_defaults["skip_steps_cache_type"] = ""

    if tea_cache_start_step_perc != None:
        del ui_defaults["tea_cache_start_step_perc"]
        ui_defaults["skip_steps_start_step_perc"] = tea_cache_start_step_perc

    image_prompt_type = ui_defaults.get("image_prompt_type", "")
    if len(image_prompt_type) > 0:
        image_prompt_types_allowed = model_def.get("image_prompt_types_allowed","")
        image_prompt_type = filter_letters(image_prompt_type, image_prompt_types_allowed)
    ui_defaults["image_prompt_type"] = image_prompt_type

    video_prompt_type = ui_defaults.get("video_prompt_type", "")
    image_ref_choices_list = model_def.get("image_ref_choices", {}).get("choices", [])
    if model_def.get("guide_custom_choices", None) is  None:
        if len(image_ref_choices_list)==0:
            video_prompt_type = del_in_sequence(video_prompt_type, "IK")
        else:
            first_choice = image_ref_choices_list[0][1]
            if "I" in first_choice and not "I" in video_prompt_type: video_prompt_type += "I"
            if len(image_ref_choices_list)==1 and "K" in first_choice and not "K" in video_prompt_type: video_prompt_type += "K"
        ui_defaults["video_prompt_type"] = video_prompt_type

    model_handler = get_model_handler(base_model_type)
    if hasattr(model_handler, "fix_settings"):
        model_handler.fix_settings(base_model_type, settings_version, model_def, ui_defaults)


def get_default_settings(model_type):
    def get_default_prompt(i2v):
        if i2v:
            return "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field."
        else:
            return "A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect."
    i2v = test_class_i2v(model_type)
    defaults_filename = get_settings_file_name(model_type)
    if not Path(defaults_filename).is_file():
        model_def = get_model_def(model_type)
        base_model_type = get_base_model_type(model_type)
        ui_defaults = {
            "settings_version" : settings_version,
            "prompt": get_default_prompt(i2v),
            "resolution": "1280x720" if "720" in base_model_type else "832x480",
            "video_length": 81,
            "num_inference_steps": 30,
            "seed": -1,
            "repeat_generation": 1,
            "multi_images_gen_type": 0,        
            "guidance_scale": 5.0,
            "flow_shift": 7.0 if not "720" in base_model_type and i2v else 5.0, 
            "negative_prompt": "",
            "activated_loras": [],
            "loras_multipliers": "",
            "skip_steps_multiplier": 1.5,
            "skip_steps_start_step_perc": 20,
            "RIFLEx_setting": 0,
            "slg_switch": 0,
            "slg_layers": [9],
            "slg_start_perc": 10,
            "slg_end_perc": 90,
            "audio_prompt_type": "V",
        }
        model_handler = get_model_handler(model_type)
        model_handler.update_default_settings(base_model_type, model_def, ui_defaults)

        ui_defaults_update = model_def.get("settings", None) 
        if ui_defaults_update is not None: ui_defaults.update(ui_defaults_update)

        if len(ui_defaults.get("prompt","")) == 0:
            ui_defaults["prompt"]= get_default_prompt(i2v)

        with open(defaults_filename, "w", encoding="utf-8") as f:
            json.dump(ui_defaults, f, indent=4)
    else:
        with open(defaults_filename, "r", encoding="utf-8") as f:
            ui_defaults = json.load(f)
        fix_settings(model_type, ui_defaults)            
    
    default_seed = args.seed
    if default_seed > -1:
        ui_defaults["seed"] = default_seed
    default_number_frames = args.frames
    if default_number_frames > 0:
        ui_defaults["video_length"] = default_number_frames
    default_number_steps = args.steps
    if default_number_steps > 0:
        ui_defaults["num_inference_steps"] = default_number_steps
    return ui_defaults


def init_model_def(model_type, model_def):
    base_model_type = get_base_model_type(model_type)
    family_handler = model_types_handlers.get(base_model_type, None)
    if family_handler is None:
        raise Exception(f"Unknown model type {model_type}")
    default_model_def = family_handler.query_model_def(base_model_type, model_def)
    if default_model_def is None: return model_def
    default_model_def.update(model_def)
    return default_model_def


models_def_paths =  glob.glob( os.path.join("defaults", "*.json") ) + glob.glob( os.path.join("finetunes", "*.json") ) 
models_def_paths.sort()
for file_path in models_def_paths:
    model_type = os.path.basename(file_path)[:-5]
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            json_def = json.load(f)
        except Exception as e:
            raise Exception(f"Error while parsing Model Definition File '{file_path}': {str(e)}")
    model_def = json_def["model"]
    model_def["path"] = file_path
    del json_def["model"]      
    settings = json_def   
    existing_model_def = models_def.get(model_type, None) 
    if existing_model_def is not None:
        existing_settings = models_def.get("settings", None)
        if existing_settings != None:
            existing_settings.update(settings)
        existing_model_def.update(model_def)
    else:
        models_def[model_type] = model_def # partial def
        model_def= init_model_def(model_type, model_def)
        models_def[model_type] = model_def # replace with full def
        model_def["settings"] = settings

model_types = models_def.keys()
displayed_model_types= []
for model_type in model_types:
    model_def = get_model_def(model_type)
    if not model_def is None and model_def.get("visible", True): 
        displayed_model_types.append(model_type)


transformer_types = server_config.get("transformer_types", [])
new_transformer_types = []
for model_type in transformer_types:
    if get_model_def(model_type) == None:
        print(f"Model '{model_type}' is missing. Either install it in the finetune folder or remove this model from ley 'transformer_types' in wgp_config.json")
    else:
        new_transformer_types.append(model_type)
transformer_types = new_transformer_types
transformer_type = server_config.get("last_model_type", None)
advanced = server_config.get("last_advanced_choice", False)
last_resolution = server_config.get("last_resolution_choice", None)
if args.advanced: advanced = True 

if transformer_type != None and not transformer_type in model_types and not transformer_type in models_def: transformer_type = None
if transformer_type == None:
    transformer_type = transformer_types[0] if len(transformer_types) > 0 else "t2v"

transformer_quantization =server_config.get("transformer_quantization", "int8")

transformer_dtype_policy = server_config.get("transformer_dtype_policy", "")
if args.fp16:
    transformer_dtype_policy = "fp16" 
if args.bf16:
    transformer_dtype_policy = "bf16" 
text_encoder_quantization =server_config.get("text_encoder_quantization", "int8")
attention_mode = server_config["attention_mode"]
if len(args.attention)> 0:
    if args.attention in ["auto", "sdpa", "sage", "sage2", "flash", "xformers"]:
        attention_mode = args.attention
        lock_ui_attention = True
    else:
        raise Exception(f"Unknown attention mode '{args.attention}'")

default_profile =  force_profile_no if force_profile_no >=0 else server_config["profile"]
loaded_profile = -1
compile = server_config.get("compile", "")
boost = server_config.get("boost", 1)
vae_config = server_config.get("vae_config", 0)
if len(args.vae_config) > 0:
    vae_config = int(args.vae_config)

reload_needed = False
save_path = server_config.get("save_path", os.path.join(os.getcwd(), "outputs"))
image_save_path = server_config.get("image_save_path", os.path.join(os.getcwd(), "outputs"))
if not "video_output_codec" in server_config: server_config["video_output_codec"]= "libx264_8"
if not "image_output_codec" in server_config: server_config["image_output_codec"]= "jpeg_95"

preload_model_policy = server_config.get("preload_model_policy", []) 


if args.t2v_14B or args.t2v: 
    transformer_type = "t2v"

if args.i2v_14B or args.i2v: 
    transformer_type = "i2v"

if args.t2v_1_3B:
    transformer_type = "t2v_1.3B"

if args.i2v_1_3B:
    transformer_type = "fun_inp_1.3B"

if args.vace_1_3B: 
    transformer_type = "vace_1.3B"

only_allow_edit_in_advanced = False
lora_preselected_preset = args.lora_preset
lora_preset_model = transformer_type

if  args.compile: #args.fastest or
    compile="transformer"
    lock_ui_compile = True


def save_model(model, model_type, dtype,  config_file,  submodel_no = 1,  is_module = False, filter = None, no_fp16_main_model = True, module_source_no = 1):
    model_def = get_model_def(model_type)
    # To save module and quantized modules
    # 1) set Transformer Model Quantization Type to 16 bits
    # 2) insert in def module_source : path and "model_fp16.safetensors in URLs"
    # 3) Generate (only quantized fp16 will be created)
    # 4) replace in def module_source : path and "model_bf16.safetensors in URLs"
    # 5) Generate (both bf16 and quantized bf16 will be created)
    if model_def == None: return
    if is_module:
        url_key = "modules"
        source_key = "module_source" if module_source_no <=1 else "module_source2"
    else:
        url_key = "URLs" if submodel_no <=1 else "URLs" + str(submodel_no)
        source_key = "source" if submodel_no <=1 else "source2"
    URLs= model_def.get(url_key, None)
    if URLs is None: return
    if isinstance(URLs, str):
        print("Unable to save model for a finetune that references external files")
        return
    from mmgp import offload    
    dtypestr= "bf16" if dtype == torch.bfloat16 else "fp16"
    if no_fp16_main_model: dtypestr = dtypestr.replace("fp16", "bf16")
    model_filename = None
    if is_module:
        if not isinstance(URLs,list) or len(URLs) != 1:
            print("Target Module files are missing")
            return 
        URLs= URLs[0]
    if isinstance(URLs, dict):
        url_dict_key = "URLs" if module_source_no ==1 else "URLs2"
        URLs = URLs[url_dict_key]
    for url in URLs:
        if "quanto" not in url and dtypestr in url:
            model_filename = os.path.basename(url)
            break
    if model_filename is None:
        print(f"No target filename with bf16 or fp16 in its name is mentioned in {url_key}")
        return

    finetune_file = os.path.join(os.path.dirname(model_def["path"]) , model_type + ".json")
    with open(finetune_file, 'r', encoding='utf-8') as reader:
        saved_finetune_def = json.load(reader)

    update_model_def = False
    model_filename = os.path.join("ckpts",model_filename)
    quanto_dtypestr= "bf16" if dtype == torch.bfloat16 else "fp16"
    if ("m" + dtypestr) in model_filename: 
        dtypestr = "m" + dtypestr 
        quanto_dtypestr = "m" + quanto_dtypestr 
    if not os.path.isfile(model_filename) and (not no_fp16_main_model or dtype == torch.bfloat16):
        offload.save_model(model, model_filename, config_file_path=config_file, filter_sd=filter)
        print(f"New model file '{model_filename}' had been created for finetune Id '{model_type}'.")
        del saved_finetune_def["model"][source_key]
        del model_def[source_key]
        print(f"The 'source' entry has been removed in the '{finetune_file}' definition file.")
        update_model_def = True

    if is_module:
        quanto_filename = model_filename.replace(dtypestr, "quanto_" + quanto_dtypestr + "_int8" )
        if hasattr(model, "_quanto_map"):
            print("unable to generate quantized module, the main model should at full 16 bits before quantization can be done")
        elif not os.path.isfile(quanto_filename):
            offload.save_model(model, quanto_filename, config_file_path=config_file, do_quantize= True, filter_sd=filter)
            print(f"New quantized file '{quanto_filename}' had been created for finetune Id '{model_type}'.")
            if isinstance(model_def[url_key][0],dict): 
                model_def[url_key][0][url_dict_key].append(quanto_filename) 
                saved_finetune_def["model"][url_key][0][url_dict_key].append(quanto_filename)
            else: 
                model_def[url_key][0].append(quanto_filename) 
                saved_finetune_def["model"][url_key][0].append(quanto_filename)
            update_model_def = True
    if update_model_def:
        with open(finetune_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(saved_finetune_def, indent=4))

def save_quantized_model(model, model_type, model_filename, dtype,  config_file, submodel_no = 1):
    if "quanto" in model_filename: return
    model_def = get_model_def(model_type)
    if model_def == None: return
    url_key = "URLs" if submodel_no <=1 else "URLs" + str(submodel_no)
    URLs= model_def.get(url_key, None)
    if URLs is None: return
    if isinstance(URLs, str):
        print("Unable to create a quantized model for a finetune that references external files")
        return
    from mmgp import offload
    if dtype == torch.bfloat16:
         model_filename =  model_filename.replace("fp16", "bf16").replace("FP16", "bf16")
    elif dtype == torch.float16:
         model_filename =  model_filename.replace("bf16", "fp16").replace("BF16", "bf16")

    for rep in ["mfp16", "fp16", "mbf16", "bf16"]:
        if "_" + rep in model_filename:
            model_filename = model_filename.replace("_" + rep, "_quanto_" + rep + "_int8")
            break
    if not "quanto" in model_filename:
        pos = model_filename.rfind(".")
        model_filename =  model_filename[:pos] + "_quanto_int8" + model_filename[pos+1:] 
    
    if os.path.isfile(model_filename):
        print(f"There isn't any model to quantize as quantized model '{model_filename}' aready exists")
    else:
        offload.save_model(model, model_filename, do_quantize= True, config_file_path=config_file)
        print(f"New quantized file '{model_filename}' had been created for finetune Id '{model_type}'.")
        if not model_filename in URLs:
            URLs.append(model_filename)
            finetune_file = os.path.join(os.path.dirname(model_def["path"]) , model_type + ".json")
            with open(finetune_file, 'r', encoding='utf-8') as reader:
                saved_finetune_def = json.load(reader)
            saved_finetune_def["model"][url_key] = URLs
            with open(finetune_file, "w", encoding="utf-8") as writer:
                writer.write(json.dumps(saved_finetune_def, indent=4))
            print(f"The '{finetune_file}' definition file has been automatically updated with the local path to the new quantized model.")

def get_loras_preprocessor(transformer, model_type):
    preprocessor =  getattr(transformer, "preprocess_loras", None)
    if preprocessor == None:
        return None
    
    def preprocessor_wrapper(sd):
        return preprocessor(model_type, sd)

    return preprocessor_wrapper

def get_local_model_filename(model_filename):
    if model_filename.startswith("http"):
        local_model_filename = os.path.join("ckpts", os.path.basename(model_filename))
    else:
        local_model_filename = model_filename
    return local_model_filename
    


def process_files_def(repoId, sourceFolderList, fileList):
    targetRoot = "ckpts/" 
    for sourceFolder, files in zip(sourceFolderList,fileList ):
        if len(files)==0:
            if not Path(targetRoot + sourceFolder).exists():
                snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= targetRoot)
        else:
            for onefile in files:     
                if len(sourceFolder) > 0: 
                    if not os.path.isfile(targetRoot + sourceFolder + "/" + onefile ):          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot, subfolder=sourceFolder)
                else:
                    if not os.path.isfile(targetRoot + onefile ):          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot)

def download_mmaudio():
    if server_config.get("mmaudio_enabled", 0) != 0:
        enhancer_def = {
            "repoId" : "DeepBeepMeep/Wan2.1",
            "sourceFolderList" : [ "mmaudio", "DFN5B-CLIP-ViT-H-14-378"  ],
            "fileList" : [ ["mmaudio_large_44k_v2.pth", "synchformer_state_dict.pth", "v1-44.pth"],["open_clip_config.json", "open_clip_pytorch_model.bin"]]
        }
        process_files_def(**enhancer_def)

def download_models(model_filename = None, model_type= None, module_type = False, submodel_no = 1):
    def computeList(filename):
        if filename == None:
            return []
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        return [filename]        



    from urllib.request import urlretrieve
    from shared.utils.download import create_progress_hook

    shared_def = {
        "repoId" : "DeepBeepMeep/Wan2.1",
        "sourceFolderList" : [ "pose", "scribble", "flow", "depth", "mask", "wav2vec", "chinese-wav2vec2-base", "roformer", "pyannote", "det_align", "" ],
        "fileList" : [ ["dw-ll_ucoco_384.onnx", "yolox_l.onnx"],["netG_A_latest.pth"],  ["raft-things.pth"], 
                      ["depth_anything_v2_vitl.pth","depth_anything_v2_vitb.pth"], ["sam_vit_h_4b8939_fp16.safetensors"], 
                      ["config.json", "feature_extractor_config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"],
                      ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
                      ["model_bs_roformer_ep_317_sdr_12.9755.ckpt", "model_bs_roformer_ep_317_sdr_12.9755.yaml", "download_checks.json"],
                      ["pyannote_model_wespeaker-voxceleb-resnet34-LM.bin", "pytorch_model_segmentation-3.0.bin"], ["detface.pt"], [ "flownet.pkl" ] ]
    }
    process_files_def(**shared_def)


    if server_config.get("enhancer_enabled", 0) == 1:
        enhancer_def = {
            "repoId" : "DeepBeepMeep/LTX_Video",
            "sourceFolderList" : [ "Florence2", "Llama3_2"  ],
            "fileList" : [ ["config.json", "configuration_florence2.py", "model.safetensors", "modeling_florence2.py", "preprocessor_config.json", "processing_florence2.py", "tokenizer.json", "tokenizer_config.json"],["config.json", "generation_config.json", "Llama3_2_quanto_bf16_int8.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]  ]
        }
        process_files_def(**enhancer_def)

    elif server_config.get("enhancer_enabled", 0) == 2:
        enhancer_def = {
            "repoId" : "DeepBeepMeep/LTX_Video",
            "sourceFolderList" : [ "Florence2", "llama-joycaption-beta-one-hf-llava"  ],
            "fileList" : [ ["config.json", "configuration_florence2.py", "model.safetensors", "modeling_florence2.py", "preprocessor_config.json", "processing_florence2.py", "tokenizer.json", "tokenizer_config.json"],["config.json", "llama_joycaption_quanto_bf16_int8.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]  ]
        }
        process_files_def(**enhancer_def)

    download_mmaudio()
    if model_filename is None: return

    def download_file(url,filename):
        if url.startswith("https://huggingface.co/") and "/resolve/main/" in url:
            base_dir = os.path.dirname(filename)
            url = url[len("https://huggingface.co/"):]
            url_parts = url.split("/resolve/main/")
            repoId = url_parts[0]
            onefile = os.path.basename(url_parts[-1])
            sourceFolder = os.path.dirname(url_parts[-1])
            if len(sourceFolder) == 0:
                hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = "ckpts/" if len(base_dir)==0 else base_dir)
            else:
                target_path = "ckpts/temp/" + sourceFolder
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = "ckpts/temp/", subfolder=sourceFolder)
                shutil.move(os.path.join( "ckpts", "temp" , sourceFolder , onefile), "ckpts/" if len(base_dir)==0 else base_dir)
                shutil.rmtree("ckpts/temp")
        else:
            urlretrieve(url,filename, create_progress_hook(filename))

    base_model_type = get_base_model_type(model_type)
    model_def = get_model_def(model_type)
    
    any_source = ("source2" if submodel_no ==2 else "source") in model_def
    any_module_source = ("module_source2" if submodel_no ==2 else "module_source") in model_def 
    model_type_handler = model_types_handlers[base_model_type]
    local_model_filename = get_local_model_filename(model_filename)

    if any_source and not module_type or any_module_source and module_type:
        model_filename = None
    else:
        if not os.path.isfile(local_model_filename):
            url = model_filename

            if not url.startswith("http"):
                raise Exception(f"Model '{model_filename}' was not found locally and no URL was provided to download it. Please add an URL in the model definition file.")
            try:
                download_file(url, local_model_filename)
            except Exception as e:
                if os.path.isfile(local_model_filename): os.remove(local_model_filename) 
                raise Exception(f"'{url}' is invalid for Model '{local_model_filename}' : {str(e)}'")
            if module_type: return
        model_filename = None

        preload_URLs = get_model_recursive_prop(model_type, "preload_URLs", return_list= True)
        for url in preload_URLs:
            filename = "ckpts/" + url.split("/")[-1]
            if not os.path.isfile(filename ): 
                if not url.startswith("http"):
                    raise Exception(f"File '{filename}' to preload was not found locally and no URL was provided to download it. Please add an URL in the model definition file.")
                try:
                    download_file(url, filename)
                except Exception as e:
                    if os.path.isfile(filename): os.remove(filename) 
                    raise Exception(f"Preload URL '{url}' is invalid: {str(e)}'")
                
        model_loras = get_model_recursive_prop(model_type, "loras", return_list= True)
        for url in model_loras:
            filename = os.path.join(get_lora_dir(model_type), url.split("/")[-1])
            if not os.path.isfile(filename ): 
                if not url.startswith("http"):
                    raise Exception(f"Lora '{filename}' was not found in the Loras Folder and no URL was provided to download it. Please add an URL in the model definition file.")
                try:
                    download_file(url, filename)
                except Exception as e:
                    if os.path.isfile(filename): os.remove(filename) 
                    raise Exception(f"Lora URL '{url}' is invalid: {str(e)}'")
    if module_type: return            
    model_files = model_type_handler.query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization)
    if not isinstance(model_files, list): model_files = [model_files]
    for one_repo in model_files:
        process_files_def(**one_repo)

offload.default_verboseLevel = verbose_level


def sanitize_file_name(file_name, rep =""):
    return file_name.replace("/",rep).replace("\\",rep).replace(":",rep).replace("|",rep).replace("?",rep).replace("<",rep).replace(">",rep).replace("\"",rep).replace("\n",rep).replace("\r",rep) 

def extract_preset(model_type, lset_name, loras):
    loras_choices = []
    loras_choices_files = []
    loras_mult_choices = ""
    prompt =""
    full_prompt =""
    lset_name = sanitize_file_name(lset_name)
    lora_dir = get_lora_dir(model_type)
    if not lset_name.endswith(".lset"):
        lset_name_filename = os.path.join(lora_dir, lset_name + ".lset" ) 
    else:
        lset_name_filename = os.path.join(lora_dir, lset_name ) 
    error = ""
    if not os.path.isfile(lset_name_filename):
        error = f"Preset '{lset_name}' not found "
    else:
        missing_loras = []

        with open(lset_name_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        lset = json.loads(text)

        loras_choices_files = lset["loras"]
        for lora_file in loras_choices_files:
            choice = os.path.join(lora_dir, lora_file)
            if choice not in loras:
                missing_loras.append(lora_file)
            else:
                loras_choice_no = loras.index(choice)
                loras_choices.append(str(loras_choice_no))

        if len(missing_loras) > 0:
            error = f"Unable to apply Lora preset '{lset_name} because the following Loras files are missing or invalid: {missing_loras}"
        
        loras_mult_choices = lset["loras_mult"]
        prompt = lset.get("prompt", "")
        full_prompt = lset.get("full_prompt", False)
    return loras_choices, loras_mult_choices, prompt, full_prompt, error


def setup_loras(model_type, transformer,  lora_dir, lora_preselected_preset, split_linear_modules_map = None):
    loras =[]
    loras_names = []
    default_loras_choices = []
    default_loras_multis_str = ""
    loras_presets = []
    default_lora_preset = ""
    default_lora_preset_prompt = ""

    from pathlib import Path

    lora_dir = get_lora_dir(model_type)
    if lora_dir != None :
        if not os.path.isdir(lora_dir):
            raise Exception("--lora-dir should be a path to a directory that contains Loras")


    if lora_dir != None:
        dir_loras =  glob.glob( os.path.join(lora_dir , "*.sft") ) + glob.glob( os.path.join(lora_dir , "*.safetensors") ) 
        dir_loras.sort()
        loras += [element for element in dir_loras if element not in loras ]

        dir_presets_settings = glob.glob( os.path.join(lora_dir , "*.json") ) 
        dir_presets_settings.sort()
        dir_presets =   glob.glob( os.path.join(lora_dir , "*.lset") ) 
        dir_presets.sort()
        # loras_presets = [ Path(Path(file_path).parts[-1]).stem for file_path in dir_presets_settings + dir_presets]
        loras_presets = [ Path(file_path).parts[-1] for file_path in dir_presets_settings + dir_presets]

    if transformer !=None:
        loras = offload.load_loras_into_model(transformer, loras,  activate_all_loras=False, check_only= True, preprocess_sd=get_loras_preprocessor(transformer, model_type), split_linear_modules_map = split_linear_modules_map) #lora_multiplier,

    if len(loras) > 0:
        loras_names = [ Path(lora).stem for lora in loras  ]

    if len(lora_preselected_preset) > 0:
        if not os.path.isfile(os.path.join(lora_dir, lora_preselected_preset + ".lset")):
            raise Exception(f"Unknown preset '{lora_preselected_preset}'")
        default_lora_preset = lora_preselected_preset
        default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, _ , error = extract_preset(model_type, default_lora_preset, loras)
        if len(error) > 0:
            print(error[:200])
    return loras, loras_names, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset

def get_transformer_model(model, submodel_no = 1):
    if submodel_no > 1:
        model_key = f"model{submodel_no}"
        if not hasattr(model, model_key): return None

    if hasattr(model, "model"):
        if submodel_no > 1:
            return getattr(model, f"model{submodel_no}")
        else:
            return model.model
    elif hasattr(model, "transformer"):
        return model.transformer
    else:
        raise Exception("no transformer found")

def init_pipe(pipe, kwargs, override_profile):
    preload =int(args.preload)
    if preload == 0:
        preload = server_config.get("preload_in_VRAM", 0)

    kwargs["extraModelsToQuantize"]=  None
    profile = override_profile if override_profile != -1 else default_profile
    if profile in (2, 4, 5):
        budgets = { "transformer" : 100 if preload  == 0 else preload, "text_encoder" : 100 if preload  == 0 else preload, "*" : max(1000 if profile==5 else 3000 , preload) }
        if "transformer2" in pipe:
            budgets["transformer2"] = 100 if preload  == 0 else preload
        kwargs["budgets"] = budgets
    elif profile == 3:
        kwargs["budgets"] = { "*" : "70%" }

    if "transformer2" in pipe:
        if profile in [3,4]:
            kwargs["pinnedMemory"] = ["transformer", "transformer2"]

    return profile

def reset_prompt_enhancer():
    global prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer, enhancer_offloadobj
    prompt_enhancer_image_caption_model = None
    prompt_enhancer_image_caption_processor = None
    prompt_enhancer_llm_model = None
    prompt_enhancer_llm_tokenizer = None
    if enhancer_offloadobj is not None:
        enhancer_offloadobj.release()
        enhancer_offloadobj = None

def setup_prompt_enhancer(pipe, kwargs):
    global prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer
    model_no = server_config.get("enhancer_enabled", 0) 
    if model_no != 0:
        from transformers import ( AutoModelForCausalLM, AutoProcessor, AutoTokenizer, LlamaForCausalLM )
        prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained( "ckpts/Florence2", trust_remote_code=True)
        prompt_enhancer_image_caption_processor = AutoProcessor.from_pretrained( "ckpts/Florence2", trust_remote_code=True)
        pipe["prompt_enhancer_image_caption_model"] = prompt_enhancer_image_caption_model
        prompt_enhancer_image_caption_model._model_dtype = torch.float
        # def preprocess_sd(sd, map):
        #     new_sd ={}
        #     for k, v in sd.items():
        #         k = "model." + k.replace(".model.", ".")
        #         if "lm_head.weight" in k: k = "lm_head.weight"
        #         new_sd[k] = v
        #     return new_sd, map
        # prompt_enhancer_llm_model = offload.fast_load_transformers_model("c:/temp/joy/model-00001-of-00004.safetensors", modelClass= LlavaForConditionalGeneration, defaultConfigPath="ckpts/llama-joycaption-beta-one-hf-llava/config.json", preprocess_sd=preprocess_sd)
        # offload.save_model(prompt_enhancer_llm_model, "joy_llava_quanto_int8.safetensors", do_quantize= True)

        if model_no == 1:
            budget = 5000
            prompt_enhancer_llm_model = offload.fast_load_transformers_model("ckpts/Llama3_2/Llama3_2_quanto_bf16_int8.safetensors")
            prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained("ckpts/Llama3_2")
        else:
            budget = 10000
            prompt_enhancer_llm_model = offload.fast_load_transformers_model("ckpts/llama-joycaption-beta-one-hf-llava/llama_joycaption_quanto_bf16_int8.safetensors")
            prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained("ckpts/llama-joycaption-beta-one-hf-llava")
        pipe["prompt_enhancer_llm_model"] = prompt_enhancer_llm_model
        if not "budgets" in kwargs: kwargs["budgets"] = {}
        kwargs["budgets"]["prompt_enhancer_llm_model"] = budget 
    else:
        reset_prompt_enhancer()



def load_models(model_type, override_profile = -1):
    global transformer_type, loaded_profile
    base_model_type = get_base_model_type(model_type)
    model_def = get_model_def(model_type)
    save_quantized = args.save_quantized and model_def != None
    model_filename = get_model_filename(model_type=model_type, quantization= "" if save_quantized else transformer_quantization, dtype_policy = transformer_dtype_policy) 
    if "URLs2" in model_def:
        model_filename2 = get_model_filename(model_type=model_type, quantization= "" if save_quantized else transformer_quantization, dtype_policy = transformer_dtype_policy, submodel_no=2) # !!!!
    else:
        model_filename2 = None
    modules = get_model_recursive_prop(model_type, "modules",  return_list= True)
    modules = [get_model_recursive_prop(module, "modules", sub_prop_name  ="_list",  return_list= True) if isinstance(module, str) else module for module in modules ]
    if save_quantized and "quanto" in model_filename:
        save_quantized = False
        print("Need to provide a non quantized model to create a quantized model to be saved") 
    if save_quantized and len(modules) > 0:
        print(f"Unable to create a finetune quantized model as some modules are declared in the finetune definition. If your finetune includes already the module weights you can remove the 'modules' entry and try again. If not you will need also to change temporarly the model 'architecture' to an architecture that wont require the modules part ({modules}) to quantize and then add back the original 'modules' and 'architecture' entries.")
        save_quantized = False
    quantizeTransformer = not save_quantized and model_def !=None and transformer_quantization in ("int8", "fp8") and model_def.get("auto_quantize", False) and not "quanto" in model_filename
    if quantizeTransformer and len(modules) > 0:
        print(f"Autoquantize is not yet supported if some modules are declared")
        quantizeTransformer = False
    model_family = get_model_family(model_type)
    transformer_dtype = get_transformer_dtype(model_family, transformer_dtype_policy)
    if quantizeTransformer or "quanto" in model_filename:
        transformer_dtype = torch.bfloat16 if "bf16" in model_filename or "BF16" in model_filename else transformer_dtype
        transformer_dtype = torch.float16 if "fp16" in model_filename or"FP16" in model_filename else transformer_dtype
    perc_reserved_mem_max = args.perc_reserved_mem_max
    vram_safety_coefficient = args.vram_safety_coefficient 
    model_file_list = [model_filename]
    model_type_list = [model_type]
    module_type_list = [None]
    model_submodel_no_list = [1]
    if model_filename2 != None:
        model_file_list += [model_filename2]
        model_type_list += [model_type]
        module_type_list += [False]
        model_submodel_no_list += [2]
    for module_type in modules:
        if isinstance(module_type,dict):
            URLs1 = module_type.get("URLs", None)
            if URLs1 is None: raise Exception(f"No URLs defined for Module {module_type}")
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, URLs = URLs1))
            URLs2 = module_type.get("URLs2", None)
            if URLs2 is None: raise Exception(f"No URL2s defined for Module {module_type}")
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, URLs = URLs2))
            model_type_list += [model_type] * 2
            module_type_list += [True] * 2
            model_submodel_no_list += [1,2]
        else:
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, module_type= module_type))
            model_type_list.append(model_type)
            module_type_list.append(True)
            model_submodel_no_list.append(0) 
    local_model_file_list= []
    for filename, file_model_type, file_module_type, submodel_no in zip(model_file_list, model_type_list, module_type_list, model_submodel_no_list): 
        download_models(filename, file_model_type, file_module_type, submodel_no)
        local_model_file_list.append( get_local_model_filename(filename) )
    VAE_dtype = torch.float16 if server_config.get("vae_precision","16") == "16" else torch.float
    mixed_precision_transformer =  server_config.get("mixed_precision","0") == "1"
    transformer_type = None
    for module_type, filename in zip(module_type_list, local_model_file_list):
        if module_type is None:  
            print(f"Loading Model '{filename}' ...")
        else: 
            print(f"Loading Module '{filename}' ...")

    wan_model, pipe = model_types_handlers[base_model_type].load_model(
                local_model_file_list, model_type, base_model_type, model_def, quantizeTransformer = quantizeTransformer, text_encoder_quantization = text_encoder_quantization,
                dtype = transformer_dtype, VAE_dtype = VAE_dtype, mixed_precision_transformer = mixed_precision_transformer, save_quantized = save_quantized, submodel_no_list   = model_submodel_no_list, )

    kwargs = {}
    profile = init_pipe(pipe, kwargs, override_profile)
    if server_config.get("enhancer_mode", 0) == 0:
        setup_prompt_enhancer(pipe, kwargs)
    loras_transformer = ["transformer"]
    if "transformer2" in pipe:
        loras_transformer += ["transformer2"]        
    offloadobj = offload.profile(pipe, profile_no= profile, compile = compile, quantizeTransformer = False, loras = loras_transformer, coTenantsMap= {}, perc_reserved_mem_max = perc_reserved_mem_max , vram_safety_coefficient = vram_safety_coefficient , convertWeightsFloatTo = transformer_dtype, **kwargs)  
    if len(args.gpu) > 0:
        torch.set_default_device(args.gpu)
    transformer_type = model_type
    loaded_profile = profile
    return wan_model, offloadobj 

if not "P" in preload_model_policy:
    wan_model, offloadobj, transformer = None, None, None
    reload_needed = True
else:
    wan_model, offloadobj = load_models(transformer_type)
    if check_loras:
        transformer = get_transformer_model(wan_model)
        setup_loras(transformer_type, transformer,  get_lora_dir(transformer_type), "", None)
        exit()

gen_in_progress = False

def get_auto_attention():
    for attn in ["sage2","sage","sdpa"]:
        if attn in attention_modes_supported:
            return attn
    return "sdpa"

def generate_header(model_type, compile, attention_mode):

    description_container = [""]
    get_model_name(model_type, description_container)
    model_filename = os.path.basename(get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)) or "" 
    description  = description_container[0]
    header = f"<DIV style=height:{60 if server_config.get('display_stats', 0) == 1 else 40}px>{description}</DIV>"
    overridden_attention = get_overridden_attention(model_type)
    attn_mode = attention_mode if overridden_attention == None else overridden_attention 
    header += "<DIV style='align:right;width:100%'><FONT SIZE=3>Attention mode <B>" + (attn_mode if attn_mode!="auto" else "auto/" + get_auto_attention() )
    if attention_mode not in attention_modes_installed:
        header += " -NOT INSTALLED-"
    elif attention_mode not in attention_modes_supported:
        header += " -NOT SUPPORTED-"
    elif overridden_attention is not None and attention_mode != overridden_attention:
        header += " -MODEL SPECIFIC-"
    header += "</B>"

    if compile:
        header += ", Pytorch compilation <B>ON</B>"
    if "fp16" in model_filename:
        header += ", Data Type <B>FP16</B>"
    else:
        header += ", Data Type <B>BF16</B>"

    if "int8" in model_filename:
        header += ", Quantization <B>Scaled Int8</B>"
    header += "<FONT></DIV>"

    return header

def release_RAM():
    if gen_in_progress:
        gr.Info("Unable to release RAM when a Generation is in Progress")
    else:
        release_model()
        gr.Info("Models stored in RAM have been released")

def apply_changes(  state,
                    transformer_types_choices,
                    transformer_dtype_policy_choice,
                    text_encoder_quantization_choice,
                    VAE_precision_choice,
                    mixed_precision_choice,
                    save_path_choice,
                    image_save_path_choice,
                    attention_choice,
                    compile_choice,
                    profile_choice,
                    vae_config_choice,
                    metadata_choice,
                    quantization_choice,
                    boost_choice = 1,
                    clear_file_list = 0,
                    preload_model_policy_choice = 1,
                    UI_theme_choice = "default",
                    enhancer_enabled_choice = 0,
                    enhancer_mode_choice = 0,
                    mmaudio_enabled_choice = 0,
                    fit_canvas_choice = 0,
                    preload_in_VRAM_choice = 0,
                    depth_anything_v2_variant_choice = "vitl",
                    notification_sound_enabled_choice = 0,
                    notification_sound_volume_choice = 50,
                    max_frames_multiplier_choice = 1,
                    display_stats_choice = 0,
                    video_output_codec_choice = None,
                    image_output_codec_choice = None,
                    audio_output_codec_choice = None,
                    last_resolution_choice = None,
):
    if args.lock_config:
        return "<DIV ALIGN=CENTER>Config Locked</DIV>",*[gr.update()]*4
    if gen_in_progress:
        return "<DIV ALIGN=CENTER>Unable to change config when a generation is in progress</DIV>",*[gr.update()]*4
    global offloadobj, wan_model, server_config, loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset, loras_presets
    server_config = {
        "attention_mode" : attention_choice,  
        "transformer_types": transformer_types_choices, 
        "text_encoder_quantization" : text_encoder_quantization_choice,
        "save_path" : save_path_choice,
        "image_save_path" : image_save_path_choice,
        "compile" : compile_choice,
        "profile" : profile_choice,
        "vae_config" : vae_config_choice,
        "vae_precision" : VAE_precision_choice,
        "mixed_precision" : mixed_precision_choice,
        "metadata_type": metadata_choice,
        "transformer_quantization" : quantization_choice,
        "transformer_dtype_policy" : transformer_dtype_policy_choice,
        "boost" : boost_choice,
        "clear_file_list" : clear_file_list,
        "preload_model_policy" : preload_model_policy_choice,
        "UI_theme" : UI_theme_choice,
        "fit_canvas": fit_canvas_choice,
        "enhancer_enabled" : enhancer_enabled_choice,
        "enhancer_mode" : enhancer_mode_choice,
        "mmaudio_enabled" : mmaudio_enabled_choice,
        "preload_in_VRAM" : preload_in_VRAM_choice,
        "depth_anything_v2_variant": depth_anything_v2_variant_choice,
        "notification_sound_enabled" : notification_sound_enabled_choice,
        "notification_sound_volume" : notification_sound_volume_choice,
        "max_frames_multiplier" : max_frames_multiplier_choice,
        "display_stats" : display_stats_choice,
        "video_output_codec" : video_output_codec_choice,
        "image_output_codec" : image_output_codec_choice,
        "audio_output_codec" : audio_output_codec_choice,
        "last_model_type" : state["model_type"],
        "last_model_per_family":  state["last_model_per_family"],
        "last_advanced_choice": state["advanced"], 
        "last_resolution_choice": last_resolution_choice, 
        "last_resolution_per_group":  state["last_resolution_per_group"],
    }

    if Path(server_config_filename).is_file():
        with open(server_config_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        old_server_config = json.loads(text)
        if lock_ui_attention:
            server_config["attention_mode"] = old_server_config["attention_mode"]
        if lock_ui_compile:
            server_config["compile"] = old_server_config["compile"]

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config, indent=4))

    changes = []
    for k, v in server_config.items():
        v_old = old_server_config.get(k, None)
        if v != v_old:
            changes.append(k)

    global attention_mode, default_profile, compile, vae_config, boost, lora_dir, reload_needed, preload_model_policy, transformer_quantization, transformer_dtype_policy, transformer_types, text_encoder_quantization, save_path 
    attention_mode = server_config["attention_mode"]
    default_profile = server_config["profile"]
    compile = server_config["compile"]
    text_encoder_quantization = server_config["text_encoder_quantization"]
    vae_config = server_config["vae_config"]
    boost = server_config["boost"]
    save_path = server_config["save_path"]
    image_save_path = server_config["image_save_path"]
    preload_model_policy = server_config["preload_model_policy"]
    transformer_quantization = server_config["transformer_quantization"]
    transformer_dtype_policy = server_config["transformer_dtype_policy"]
    text_encoder_quantization = server_config["text_encoder_quantization"]
    transformer_types = server_config["transformer_types"]
    model_filename = get_model_filename(transformer_type, transformer_quantization, transformer_dtype_policy)
    state["model_filename"] = model_filename
    if "enhancer_enabled" in changes or "enhancer_mode" in changes:
        reset_prompt_enhancer()
    if all(change in ["attention_mode", "vae_config", "boost", "save_path", "metadata_type", "clear_file_list", "fit_canvas", "depth_anything_v2_variant", 
                      "notification_sound_enabled", "notification_sound_volume", "mmaudio_enabled", "max_frames_multiplier", "display_stats",
                      "video_output_codec", "image_output_codec", "audio_output_codec"] for change in changes ):
        model_family = gr.Dropdown()
        model_choice = gr.Dropdown()
    else:
        reload_needed = True
        model_family, model_choice = generate_dropdown_model_list(transformer_type)

    header = generate_header(state["model_type"], compile=compile, attention_mode= attention_mode)
    mmaudio_enabled = server_config["mmaudio_enabled"] > 0
    return "<DIV ALIGN=CENTER>The new configuration has been succesfully applied</DIV>", header, model_family, model_choice, get_unique_id()

def get_gen_info(state):
    cache = state.get("gen", None)
    if cache == None:
        cache = dict()
        state["gen"] = cache
    return cache

def build_callback(state, pipe, send_cmd, status, num_inference_steps):
    gen = get_gen_info(state)
    gen["num_inference_steps"] = num_inference_steps
    start_time = time.time()    
    def callback(step_idx = -1, latent = None, force_refresh = True, read_state = False, override_num_inference_steps = -1, pass_no = -1, denoising_extra =""):
        in_pause = False
        with gen_lock:
            process_status = gen.get("process_status", None)
            pause_msg = None
            if process_status.startswith("request:"):        
                gen["process_status"] = "process:" + process_status[len("request:"):]
                offloadobj.unload_all()
                pause_msg = gen.get("pause_msg", "Unknown Pause")
                in_pause = True

        if in_pause:
            send_cmd("progress", [0, pause_msg])
            while True:
                time.sleep(1)            
                with gen_lock:
                    process_status = gen.get("process_status", None)
                    if process_status == "process:main": break

        refresh_id =  gen.get("refresh", -1)
        if force_refresh or step_idx >= 0:
            pass
        else:
            refresh_id =  gen.get("refresh", -1)
            if refresh_id < 0:
                return
            UI_refresh = state.get("refresh", 0)
            if UI_refresh >= refresh_id:
                return  
        if override_num_inference_steps > 0:
            gen["num_inference_steps"] = override_num_inference_steps
            
        num_inference_steps = gen.get("num_inference_steps", 0)
        status = gen["progress_status"]
        state["refresh"] = refresh_id
        if read_state:
            phase, step_idx  = gen["progress_phase"] 
        else:
            step_idx += 1         
            if gen.get("abort", False):
                # pipe._interrupt = True
                phase = "Aborting"    
            elif step_idx  == num_inference_steps:
                phase = "VAE Decoding"    
            else:
                if pass_no <=0:
                    phase = "Denoising"
                elif pass_no == 1:
                    phase = "Denoising First Pass"
                elif pass_no == 2:
                    phase = "Denoising Second Pass"
                elif pass_no == 3:
                    phase = "Denoising Third Pass"
                else:
                    phase = f"Denoising {pass_no}th Pass"

                if len(denoising_extra) > 0: phase += " | " + denoising_extra

            gen["progress_phase"] = (phase, step_idx)
        status_msg = merge_status_context(status, phase)      

        elapsed_time = time.time() - start_time
        status_msg = merge_status_context(status, f"{phase} | {format_time(elapsed_time)}")              
        if step_idx >= 0:
            progress_args = [(step_idx , num_inference_steps) , status_msg  ,  num_inference_steps]
        else:
            progress_args = [0, status_msg]
        
        # progress(*progress_args)
        send_cmd("progress", progress_args)
        if latent != None:
            latent = latent.to("cpu", non_blocking=True)
            send_cmd("preview", latent)
            
        # gen["progress_args"] = progress_args
            
    return callback
def abort_generation(state):
    gen = get_gen_info(state)
    if "in_progress" in gen: # and wan_model != None:
        if wan_model != None:
            wan_model._interrupt= True
        gen["abort"] = True            
        msg = "Processing Request to abort Current Generation"
        gen["status"] = msg
        gr.Info(msg)
        return gr.Button(interactive=  False)
    else:
        return gr.Button(interactive=  True)



def refresh_gallery(state): #, msg
    gen = get_gen_info(state)

    # gen["last_msg"] = msg
    file_list = gen.get("file_list", None)      
    choice = gen.get("selected",0)
    header_text = gen.get("header_text", "")
    in_progress = "in_progress" in gen
    if gen.get("last_selected", True) and file_list is not None:
        choice = max(len(file_list) - 1,0)  

    queue = gen.get("queue", [])
    abort_interactive = not gen.get("abort", False)
    if not in_progress or len(queue) == 0:
        return gr.Gallery(selected_index=choice, value = file_list), gr.HTML("", visible= False),  gr.Button(visible=True), gr.Button(visible=False), gr.Row(visible=False), gr.Row(visible=False), update_queue_data(queue), gr.Button(interactive=  abort_interactive), gr.Button(visible= False)
    else:
        task = queue[0]
        prompt =  task["prompt"]
        params = task["params"]
        model_type = params["model_type"] 
        base_model_type = get_base_model_type(model_type)
        model_def = get_model_def(model_type) 
        onemorewindow_visible = test_any_sliding_window(base_model_type) and params.get("image_mode",0) == 0 and not params.get("mode","").startswith("edit_")
        enhanced = False
        if prompt.startswith("!enhanced!\n"):
            enhanced = True
            prompt = prompt[len("!enhanced!\n"):]
        if "\n" in prompt :
            prompts = prompt.split("\n")
            window_no= gen.get("window_no",1)
            if window_no > len(prompts):
                window_no = len(prompts)
            window_no -= 1
            prompts[window_no]="<B>" + prompts[window_no] + "</B>"
            prompt = "<BR><DIV style='height:8px'></DIV>".join(prompts)
        if enhanced:
            prompt = "<U><B>Enhanced:</B></U><BR>" + prompt

        if len(header_text) > 0:
            prompt =  "<I>" + header_text + "</I><BR><BR>" + prompt
        list_uri = []
        list_labels = []
        start_img_uri = task.get('start_image_data_base64')
        if start_img_uri != None:
            list_uri += start_img_uri
            list_labels += task.get('start_image_labels')
        end_img_uri = task.get('end_image_data_base64')
        if end_img_uri != None:
            list_uri += end_img_uri
            list_labels += task.get('end_image_labels')

        thumbnail_size = "100px"
        thumbnails = ""
        for i, (img_label, img_uri) in enumerate(zip(list_labels,list_uri)):
            thumbnails += f'<TD onclick=sendColIndex({i})><div class="hover-image" ><img src="{img_uri}" alt="{img_label}" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display: block; margin: auto; object-fit: contain;" /><span class="tooltip">{img_label}</span></div></TD>'  
        
        # Get current theme from server config  
        current_theme = server_config.get("UI_theme", "default")
        
        # Use minimal, adaptive styling that blends with any background
        # This creates a subtle container that doesn't interfere with the page's theme
        table_style = """
            border: 1px solid rgba(128, 128, 128, 0.3); 
            background-color: transparent; 
            color: inherit; 
            padding: 8px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        """
        if params.get("mode", None) in ['edit'] : onemorewindow_visible = False
        gen_buttons_visible = True
        html =  f"<TABLE WIDTH=100% ID=PINFO style='{table_style}'><TR style='height:140px'><TD width=100% style='{table_style}'>" + prompt + "</TD>" + thumbnails + "</TR></TABLE>" 
        html_output = gr.HTML(html, visible= True)
        return gr.Gallery(selected_index=choice, value = file_list), html_output, gr.Button(visible=False), gr.Button(visible=True), gr.Row(visible=True), gr.Row(visible= gen_buttons_visible), update_queue_data(queue), gr.Button(interactive=  abort_interactive), gr.Button(visible= onemorewindow_visible)



def finalize_generation(state):
    gen = get_gen_info(state)
    choice = gen.get("selected",0)
    if "in_progress" in gen:
        del gen["in_progress"]
    if gen.get("last_selected", True):
        file_list = gen.get("file_list", [])
        choice = len(file_list) - 1


    gen["extra_orders"] = 0
    time.sleep(0.2)
    global gen_in_progress
    gen_in_progress = False
    return gr.Gallery(selected_index=choice), gr.Button(interactive=  True), gr.Button(visible= True), gr.Button(visible= False), gr.Column(visible= False), gr.HTML(visible= False, value="")

def get_default_video_info():
    return "Please Select an Video / Image"    


def get_file_list(state, input_file_list):
    gen = get_gen_info(state)
    with lock:
        if "file_list" in gen:
            file_list = gen["file_list"]
            file_settings_list = gen["file_settings_list"]
        else:
            file_list = []
            file_settings_list = []
            if input_file_list != None:
                for file_path in input_file_list:
                    if isinstance(file_path, tuple): file_path = file_path[0]
                    file_settings, _ = get_settings_from_file(state, file_path, False, False, False)
                    file_list.append(file_path)
                    file_settings_list.append(file_settings)
 
            gen["file_list"] = file_list 
            gen["file_settings_list"] = file_settings_list 
    return file_list, file_settings_list

def set_file_choice(gen, file_list, choice):
    gen["last_selected"] = (choice + 1) >= len(file_list)
    gen["selected"] = choice

def select_video(state, input_file_list, event_data: gr.EventData):
    data=  event_data._data
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, input_file_list)

    if data!=None and isinstance(data, dict):
        choice = data.get("index",0)
    else:
        choice = min(len(file_list)-1, gen.get("selected",0)) if len(file_list) > 0 else -1
    set_file_choice(gen, file_list, choice)
    

    if len(file_list) > 0:
        configs = file_settings_list[choice]
        file_name = file_list[choice]
        values = [  os.path.basename(file_name)]
        labels = [ "File Name"]
        misc_values= []
        misc_labels = []
        pp_values= []
        pp_labels = []
        extension = os.path.splitext(file_name)[-1]
        if not has_video_file_extension(file_name):
            img = Image.open(file_name)
            width, height = img.size
            is_image = True
            frames_count = fps = 1
            nb_audio_tracks =  0 
        else:
            fps, width, height, frames_count = get_video_info(file_name)
            is_image = False
            nb_audio_tracks = extract_audio_tracks(file_name,query_only = True)
        if configs != None:
            video_model_name =  configs.get("type", "Unknown model")
            if "-" in video_model_name: video_model_name =  video_model_name[video_model_name.find("-")+2:] 
            misc_values += [video_model_name]
            misc_labels += ["Model"]
            video_temporal_upsampling = configs.get("temporal_upsampling", "")
            video_spatial_upsampling = configs.get("spatial_upsampling", "")
            video_film_grain_intensity = configs.get("film_grain_intensity", 0)
            video_film_grain_saturation = configs.get("film_grain_saturation", 0.5)
            video_MMAudio_setting = configs.get("MMAudio_setting", 0)
            video_MMAudio_prompt = configs.get("MMAudio_prompt", "")
            video_MMAudio_neg_prompt = configs.get("MMAudio_neg_prompt", "")
            video_seed = configs.get("seed", -1)
            video_MMAudio_seed = configs.get("MMAudio_seed", video_seed)        
            if len(video_spatial_upsampling) > 0:
                video_temporal_upsampling += " " + video_spatial_upsampling
            if len(video_temporal_upsampling) > 0:
                pp_values += [ video_temporal_upsampling ]
                pp_labels += [ "Upsampling" ]
            if video_film_grain_intensity > 0:
                pp_values += [ f"Intensity={video_film_grain_intensity}, Saturation={video_film_grain_saturation}" ]
                pp_labels += [ "Film Grain" ]
            if video_MMAudio_setting != 0:
                pp_values += [ f'Prompt="{video_MMAudio_prompt}", Neg Prompt="{video_MMAudio_neg_prompt}", Seed={video_MMAudio_seed}'  ]
                pp_labels += [ "MMAudio" ]


        if configs == None or not "seed" in configs:
            values += misc_values
            labels += misc_labels
            video_creation_date = str(get_file_creation_date(file_name))
            if "." in video_creation_date: video_creation_date = video_creation_date[:video_creation_date.rfind(".")]
            if is_image:
                values += [f"{width}x{height}"]
                labels += ["Resolution"]
            else:
                values += [f"{width}x{height}",  f"{frames_count} frames (duration={frames_count/fps:.1f} s, fps={round(fps)})"]
                labels += ["Resolution", "Frames"]
            if nb_audio_tracks  > 0:
                values +=[nb_audio_tracks]
                labels +=["Nb Audio Tracks"]

            values += pp_values
            labels += pp_labels

            values +=[video_creation_date]
            labels +=["Creation Date"]
        else: 
            video_prompt =  configs.get("prompt", "")[:1024]
            video_video_prompt_type = configs.get("video_prompt_type", "")
            video_image_prompt_type = configs.get("image_prompt_type", "")
            video_audio_prompt_type = configs.get("audio_prompt_type", "")
            def check(src, cond):
                pos, neg = cond if isinstance(cond, tuple) else (cond, None)
                if not all_letters(src, pos): return False
                if neg is not None and any_letters(src, neg): return False
                return True
            image_outputs = configs.get("image_mode",0) > 0
            map_video_prompt  = {"V" : "Control Image" if image_outputs else "Control Video", ("VA", "U") : "Mask Image" if image_outputs else "Mask Video", "I" : "Reference Images"}
            map_image_prompt  = {"V" : "Source Video", "L" : "Last Video", "S" : "Start Image", "E" : "End Image"}
            map_audio_prompt  = {"A" : "Audio Source", "B" : "Audio Source #2"}
            video_other_prompts =  [ v for s,v in map_image_prompt.items() if all_letters(video_image_prompt_type,s)] \
                                 + [ v for s,v in map_video_prompt.items() if check(video_video_prompt_type,s)] \
                                 + [ v for s,v in map_audio_prompt.items() if all_letters(video_audio_prompt_type,s)] 
            video_model_type =  configs.get("model_type", "t2v")
            model_family = get_model_family(video_model_type)
            model_def = get_model_def(video_model_type)
            multiple_submodels = model_def.get("multiple_submodels", False)
            video_other_prompts = ", ".join(video_other_prompts)
            video_resolution = configs.get("resolution", "") + f" (real: {width}x{height})"
            video_length = configs.get("video_length", 0)
            original_fps= int(video_length/frames_count*fps)
            video_length_summary = f"{video_length} frames"
            video_window_no = configs.get("window_no", 0)
            if video_window_no > 0: video_length_summary +=f", Window no {video_window_no }" 
            if is_image:
                video_length_summary = configs.get("batch_size", 1)
                video_length_label = "Number of Images"
            else:
                video_length_summary += " ("
                video_length_label = "Video Length"
                if video_length != frames_count: video_length_summary += f"real: {frames_count} frames, "
                video_length_summary += f"{frames_count/fps:.1f}s, {round(fps)} fps)"
            video_guidance_scale = configs.get("guidance_scale", None)
            video_guidance2_scale = configs.get("guidance2_scale", None)
            video_guidance3_scale = configs.get("guidance3_scale", None)
            video_audio_guidance_scale = configs.get("audio_guidance_scale", None)
            video_switch_threshold = configs.get("switch_threshold", 0)
            video_switch_threshold2 = configs.get("switch_threshold2", 0)
            video_model_switch_phase = configs.get("model_switch_phase", 1)
            video_guidance_phases = configs.get("guidance_phases", 0)
            video_embedded_guidance_scale = configs.get("embedded_guidance_scale", None)
            video_guidance_label = "Guidance"
            if model_def.get("embedded_guidance", False):
                video_guidance_scale = video_embedded_guidance_scale
                video_guidance_label = "Embedded Guidance Scale"
            elif video_guidance_phases > 0:
                if video_guidance_phases == 1:
                    video_guidance_scale = f"{video_guidance_scale}"
                elif video_guidance_phases == 2:
                    if multiple_submodels:
                        video_guidance_scale = f"{video_guidance_scale} (High Noise), {video_guidance2_scale} (Low Noise) with Switch at Noise Level {video_switch_threshold}"
                    else:
                        video_guidance_scale = f"{video_guidance_scale}, {video_guidance2_scale} with Guidance Switch at Noise Level {video_switch_threshold}"
                else:
                    video_guidance_scale = f"{video_guidance_scale}, {video_guidance2_scale} & {video_guidance3_scale} with Switch at Noise Levels {video_switch_threshold} & {video_switch_threshold2}"
                    if multiple_submodels:
                        video_guidance_scale += f" + Model Switch at {video_switch_threshold if video_model_switch_phase ==1 else video_switch_threshold2}"
            video_flow_shift = configs.get("flow_shift", None)
            if image_outputs: video_flow_shift = None 
            video_video_guide_outpainting = configs.get("video_guide_outpainting", "")
            video_outpainting = ""
            if len(video_video_guide_outpainting) > 0  and not video_video_guide_outpainting.startswith("#") \
                    and (any_letters(video_video_prompt_type, "VFK") ) :
                video_video_guide_outpainting = video_video_guide_outpainting.split(" ")
                video_outpainting = f"Top={video_video_guide_outpainting[0]}%, Bottom={video_video_guide_outpainting[1]}%, Left={video_video_guide_outpainting[2]}%, Right={video_video_guide_outpainting[3]}%" 
            video_num_inference_steps = configs.get("num_inference_steps", 0)
            video_creation_date = str(get_file_creation_date(file_name))
            if "." in video_creation_date: video_creation_date = video_creation_date[:video_creation_date.rfind(".")]
            video_generation_time =  str(configs.get("generation_time", "0")) + "s"
            video_activated_loras = configs.get("activated_loras", [])
            video_loras_multipliers = configs.get("loras_multipliers", "")
            video_loras_multipliers =  preparse_loras_multipliers(video_loras_multipliers)
            video_loras_multipliers += [""] * len(video_activated_loras)
            video_activated_loras = [ f"<TR><TD style='padding-top:0px;padding-left:0px'>{lora}</TD><TD>x{multiplier if len(multiplier)>0 else '1'}</TD></TR>" for lora, multiplier in zip(video_activated_loras, video_loras_multipliers) ]
            video_activated_loras_str = "<TABLE style='border:0px;padding:0px'>" + "".join(video_activated_loras) + "</TABLE>" if len(video_activated_loras) > 0 else ""
            values +=  misc_values + [video_prompt]
            labels +=  misc_labels + ["Text Prompt"]
            if len(video_other_prompts) >0 :
                values += [video_other_prompts]
                labels += ["Other Prompts"]
            if len(video_outpainting) >0:
                values += [video_outpainting]
                labels += ["Outpainting"]
            video_sample_solver = configs.get("sample_solver", "")
            if model_def.get("sample_solvers", None) is not None and len(video_sample_solver) > 0 :
                values += [video_sample_solver]
                labels += ["Sampler Solver"]                                        
            values += [video_resolution, video_length_summary, video_seed, video_guidance_scale, video_audio_guidance_scale, video_flow_shift, video_num_inference_steps]
            labels += [ "Resolution", video_length_label, "Seed", video_guidance_label, "Audio Guidance Scale", "Shift Scale", "Num Inference steps"]
            video_negative_prompt = configs.get("negative_prompt", "")
            if len(video_negative_prompt) > 0:
                values += [video_negative_prompt]
                labels += ["Negative Prompt"]        
            video_NAG_scale = configs.get("NAG_scale", None)
            if video_NAG_scale is not None and video_NAG_scale > 1: 
                values += [video_NAG_scale]
                labels += ["NAG Scale"]      
            video_apg_switch = configs.get("apg_switch", None)
            if video_apg_switch is not None and video_apg_switch != 0: 
                values += ["on"]
                labels += ["APG"]      
                
            video_skip_steps_cache_type = configs.get("skip_steps_cache_type", "")
            video_skip_steps_multiplier = configs.get("skip_steps_multiplier", 0)
            video_skip_steps_cache_start_step_perc = configs.get("skip_steps_start_step_perc", 0)
            if len(video_skip_steps_cache_type) > 0:
                video_skip_steps_cache = "TeaCache" if video_skip_steps_cache_type == "tea" else "MagCache"
                video_skip_steps_cache += f" x{video_skip_steps_multiplier }"
                if video_skip_steps_cache_start_step_perc >0:  video_skip_steps_cache += f", Start from {video_skip_steps_cache_start_step_perc}%"
                values += [ video_skip_steps_cache ]
                labels += [ "Skip Steps" ]

            values += pp_values
            labels += pp_labels

            if len(video_activated_loras_str) > 0:
                values += [video_activated_loras_str]
                labels += ["Loras"] 
            if nb_audio_tracks  > 0:
                values +=[nb_audio_tracks]
                labels +=["Nb Audio Tracks"]
            values += [ video_creation_date, video_generation_time ]
            labels += [ "Creation Date", "Generation Time" ]
        labels = [label for value, label in zip(values, labels) if value is not None]
        values = [value for value in values if value is not None]

        table_style = """<STYLE>
            #video_info, #video_info TR, #video_info TD {
            background-color: transparent; 
            color: inherit; 
            padding: 4px;
            border:0px !important;
            font-size:12px;
            }
            </STYLE>
        """
        rows = [f"<TR><TD style='text-align: right;' WIDTH=1% NOWRAP VALIGN=TOP>{label}</TD><TD><B>{value}</B></TD></TR>" for label, value in zip(labels, values)]
        html = f"{table_style}<TABLE ID=video_info WIDTH=100%>" + "".join(rows) + "</TABLE>"
    else:
        html =  get_default_video_info()
    visible= len(file_list) > 0
    return choice, html, gr.update(visible=visible and not is_image) , gr.update(visible=visible and is_image), gr.update(visible=visible and not is_image) , gr.update(visible=visible and not is_image) 

def convert_image(image):

    from PIL import ImageOps
    from typing import cast
    if isinstance(image, str):
        image = Image.open(image)
    image = image.convert('RGB')
    return cast(Image, ImageOps.exif_transpose(image))

def get_resampled_video(video_in, start_frame, max_frames, target_fps, bridge='torch'):
    if isinstance(video_in, str) and has_image_file_extension(video_in):
        video_in = Image.open(video_in)
    if isinstance(video_in, Image.Image):
        return torch.from_numpy(np.array(video_in).astype(np.uint8)).unsqueeze(0)
    
    from shared.utils.utils import resample

    import decord
    decord.bridge.set_bridge(bridge)
    reader = decord.VideoReader(video_in)
    fps = round(reader.get_avg_fps())
    if max_frames < 0:
        max_frames = max(len(reader)/ fps * target_fps + max_frames, 0)


    frame_nos = resample(fps, len(reader), max_target_frames_count= max_frames, target_fps=target_fps, start_target_frame= start_frame)
    frames_list = reader.get_batch(frame_nos)
    # print(f"frame nos: {frame_nos}")
    return frames_list

# def get_resampled_video(video_in, start_frame, max_frames, target_fps):
#     from torchvision.io import VideoReader
#     import torch
#     from shared.utils.utils import resample

#     vr = VideoReader(video_in, "video")
#     meta = vr.get_metadata()["video"]

#     fps = round(float(meta["fps"][0]))
#     duration_s = float(meta["duration"][0])
#     num_src_frames = int(round(duration_s * fps))  # robust length estimate

#     if max_frames < 0:
#         max_frames = max(int(num_src_frames / fps * target_fps + max_frames), 0)

#     frame_nos = resample(
#         fps, num_src_frames,
#         max_target_frames_count=max_frames,
#         target_fps=target_fps,
#         start_target_frame=start_frame
#     )
#     if len(frame_nos) == 0:
#         return torch.empty((0,))  # nothing to return

#     target_ts = [i / fps for i in frame_nos]

#     # Read forward once, grabbing frames when we pass each target timestamp
#     frames = []
#     vr.seek(target_ts[0])
#     idx = 0
#     tol = 0.5 / fps  # half-frame tolerance
#     for frame in vr:
#         t = float(frame["pts"])       # seconds
#         if idx < len(target_ts) and t + tol >= target_ts[idx]:
#             frames.append(frame["data"].permute(1,2,0))  # Tensor [H, W, C]
#             idx += 1
#             if idx >= len(target_ts):
#                 break

#     return frames


def get_preprocessor(process_type, inpaint_color):
    if process_type=="pose":
        from preprocessing.dwpose.pose import PoseBodyFaceVideoAnnotator
        cfg_dict = {
            "DETECTION_MODEL": "ckpts/pose/yolox_l.onnx",
            "POSE_MODEL": "ckpts/pose/dw-ll_ucoco_384.onnx",
            "RESIZE_SIZE": 1024
        }
        anno_ins = lambda img: PoseBodyFaceVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="depth":
        # from preprocessing.midas.depth import DepthVideoAnnotator
        # cfg_dict = {
        #     "PRETRAINED_MODEL": "ckpts/depth/dpt_hybrid-midas-501f0c75.pt"
        # }
        # anno_ins = lambda img: DepthVideoAnnotator(cfg_dict).forward(img)[0]

        from preprocessing.depth_anything_v2.depth import DepthV2VideoAnnotator

        if server_config.get("depth_anything_v2_variant", "vitl") == "vitl":
            cfg_dict = {
                "PRETRAINED_MODEL": "ckpts/depth/depth_anything_v2_vitl.pth",
                'MODEL_VARIANT': 'vitl'
            }
        else:
            cfg_dict = {
                "PRETRAINED_MODEL": "ckpts/depth/depth_anything_v2_vitb.pth",
                'MODEL_VARIANT': 'vitb',
            }

        anno_ins = lambda img: DepthV2VideoAnnotator(cfg_dict).forward(img)
    elif process_type=="gray":
        from preprocessing.gray import GrayVideoAnnotator
        cfg_dict = {}
        anno_ins = lambda img: GrayVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="canny":
        from preprocessing.canny import CannyVideoAnnotator
        cfg_dict = {
                "PRETRAINED_MODEL": "ckpts/scribble/netG_A_latest.pth"
            }
        anno_ins = lambda img: CannyVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="scribble":
        from preprocessing.scribble import ScribbleVideoAnnotator
        cfg_dict = {
                "PRETRAINED_MODEL": "ckpts/scribble/netG_A_latest.pth"
            }
        anno_ins = lambda img: ScribbleVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="flow":
        from preprocessing.flow import FlowVisAnnotator
        cfg_dict = {
                "PRETRAINED_MODEL": "ckpts/flow/raft-things.pth"
            }
        anno_ins = lambda img: FlowVisAnnotator(cfg_dict).forward(img)
    elif process_type=="inpaint":
        anno_ins = lambda img :  len(img) * [inpaint_color]
    elif process_type == None or process_type in ["raw", "identity"]:
        anno_ins = lambda img : img
    else:
        raise Exception(f"process type '{process_type}' non supported")
    return anno_ins


def process_images_multithread(image_processor, items, process_type, wrap_in_list = True, max_workers: int = os.cpu_count()/ 2) :
    if not items:
       return []    

    import concurrent.futures
    start_time = time.time()
    # print(f"Preprocessus:{process_type} started")
    if process_type in ["prephase", "upsample"]: 
        if wrap_in_list :
            items = [ [img] for img in items]
        if max_workers == 1:
            results = [image_processor(img) for img in items]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(image_processor, img): idx for idx, img in enumerate(items)}
                results = [None] * len(items)
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()

        if wrap_in_list: 
            results = [ img[0] for img in results]
    else:
        results=  image_processor(items) 

    end_time = time.time()
    # print(f"duration:{end_time-start_time:.1f}")

    return results

def extract_faces_from_video_with_mask(input_video_path, input_mask_path, max_frames, start_frame, target_fps, size = 512):
    if not input_video_path or max_frames <= 0:
        return None, None
    pad_frames = 0
    if start_frame < 0:
        pad_frames= -start_frame
        max_frames += start_frame
        start_frame = 0

    any_mask = input_mask_path != None
    video = get_resampled_video(input_video_path, start_frame, max_frames, target_fps)
    if len(video) == 0: return None
    if any_mask:
        mask_video = get_resampled_video(input_mask_path, start_frame, max_frames, target_fps)
    frame_height, frame_width, _ = video[0].shape

    num_frames = min(len(video), len(mask_video))
    if num_frames == 0: return None
    video, mask_video = video[:num_frames], mask_video[:num_frames]

    from preprocessing.face_preprocessor  import FaceProcessor 
    face_processor = FaceProcessor()

    face_list = []
    for frame_idx in range(num_frames):
        frame = video[frame_idx].cpu().numpy() 
        # video[frame_idx] = None
        if any_mask:
            mask = Image.fromarray(mask_video[frame_idx].cpu().numpy()) 
            # mask_video[frame_idx] = None
            if (frame_width, frame_height) != mask.size:
                mask = mask.resize((frame_width, frame_height), resample=Image.Resampling.LANCZOS)
            mask = np.array(mask)
            alpha_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            alpha_mask[mask > 127] = 1
            frame = frame * alpha_mask
        frame = Image.fromarray(frame)
        face = face_processor.process(frame, resize_to=size, face_crop_scale = 1)
        face_list.append(face)

    face_processor = None
    gc.collect()
    torch.cuda.empty_cache()

    face_tensor= torch.tensor(np.stack(face_list, dtype= np.float32) / 127.5 - 1).permute(-1, 0, 1, 2 ) # t h w c -> c t h w
    if pad_frames > 0:
        face_tensor= torch.cat([face_tensor[:, -1:].expand(-1, pad_frames, -1, -1), face_tensor ], dim=2)
        
    if args.save_masks:
        from preprocessing.dwpose.pose import save_one_video
        saved_faces_frames = [np.array(face) for face in face_list ]
        save_one_video(f"faces.mp4", saved_faces_frames, fps=target_fps, quality=8, macro_block_size=None)
    return face_tensor

def get_default_workers():
    return os.cpu_count()/ 2

def preprocess_video_with_mask(input_video_path, input_mask_path, height, width,  max_frames, start_frame=0, fit_canvas = None, fit_crop = False, target_fps = 16, block_size= 16, expand_scale = 2, process_type = "inpaint", process_type2 = None, to_bbox = False, RGB_Mask = False, negate_mask = False, process_outside_mask = None, inpaint_color = 127, outpainting_dims = None, proc_no = 1):

    def mask_to_xyxy_box(mask):
        rows, cols = np.where(mask == 255)
        xmin = min(cols)
        xmax = max(cols) + 1
        ymin = min(rows)
        ymax = max(rows) + 1
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, mask.shape[1])
        ymax = min(ymax, mask.shape[0])
        box = [xmin, ymin, xmax, ymax]
        box = [int(x) for x in box]
        return box
    inpaint_color = int(inpaint_color)
    pad_frames = 0
    if start_frame < 0:
        pad_frames= -start_frame
        max_frames += start_frame
        start_frame = 0

    if not input_video_path or max_frames <= 0:
        return None, None
    any_mask = input_mask_path != None
    pose_special = "pose" in process_type
    any_identity_mask = False
    if process_type == "identity":
        any_identity_mask = True
        negate_mask = False
        process_outside_mask = None
    preproc = get_preprocessor(process_type, inpaint_color)
    preproc2 = None
    if process_type2 != None:
        preproc2 = get_preprocessor(process_type2, inpaint_color) if process_type != process_type2 else preproc
    if process_outside_mask == process_type :
        preproc_outside = preproc
    elif preproc2 != None and process_outside_mask == process_type2 :
        preproc_outside = preproc2
    else:
        preproc_outside = get_preprocessor(process_outside_mask, inpaint_color)
    video = get_resampled_video(input_video_path, start_frame, max_frames, target_fps)
    if any_mask:
        mask_video = get_resampled_video(input_mask_path, start_frame, max_frames, target_fps)

    if len(video) == 0 or any_mask and len(mask_video) == 0:
        return None, None
    if fit_crop and outpainting_dims != None:
        fit_crop = False
        fit_canvas = 0 if fit_canvas is not None else None

    frame_height, frame_width, _ = video[0].shape

    if outpainting_dims != None:
        if fit_canvas != None:
            frame_height, frame_width = get_outpainting_full_area_dimensions(frame_height,frame_width, outpainting_dims)
        else:
            frame_height, frame_width = height, width

    if fit_canvas != None:
        height, width = calculate_new_dimensions(height, width, frame_height, frame_width, fit_into_canvas = fit_canvas, block_size = block_size)

    if outpainting_dims != None:
        final_height, final_width = height, width
        height, width, margin_top, margin_left =  get_outpainting_frame_location(final_height, final_width,  outpainting_dims, 1)        

    if any_mask:
        num_frames = min(len(video), len(mask_video))
    else:
        num_frames = len(video)

    if any_identity_mask:
        any_mask = True

    proc_list =[]
    proc_list_outside =[]
    proc_mask = []

    # for frame_idx in range(num_frames):
    def prep_prephase(frame_idx):
        frame = Image.fromarray(video[frame_idx].cpu().numpy()) #.asnumpy()
        if fit_crop:
            frame = rescale_and_crop(frame, width, height)
        else:
            frame = frame.resize((width, height), resample=Image.Resampling.LANCZOS) 
        frame = np.array(frame) 
        if any_mask:
            if any_identity_mask:
                mask = np.full( (height, width, 3), 0, dtype= np.uint8)
            else:
                mask = Image.fromarray(mask_video[frame_idx].cpu().numpy()) #.asnumpy()
                if fit_crop:
                    mask = rescale_and_crop(mask, width, height)
                else:
                    mask = mask.resize((width, height), resample=Image.Resampling.LANCZOS) 
                mask = np.array(mask)

            if len(mask.shape) == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            original_mask = mask.copy()
            if expand_scale != 0:
                kernel_size = abs(expand_scale)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                op_expand = cv2.dilate if expand_scale > 0 else cv2.erode
                mask = op_expand(mask, kernel, iterations=3)

            _, mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)
            if to_bbox and np.sum(mask == 255) > 0:
                x0, y0, x1, y1 = mask_to_xyxy_box(mask)
                mask = mask * 0
                mask[y0:y1, x0:x1] = 255
            if negate_mask:
                mask = 255 - mask
                if pose_special:
                    original_mask = 255 - original_mask

        if pose_special and any_mask:            
            target_frame = np.where(original_mask[..., None], frame, 0) 
        else:
            target_frame = frame 

        if any_mask:
            return (target_frame, frame, mask) 
        else:
            return (target_frame, None, None)
    max_workers = get_default_workers()
    proc_lists = process_images_multithread(prep_prephase, [frame_idx for frame_idx in range(num_frames)], "prephase", wrap_in_list= False, max_workers=max_workers)
    proc_list, proc_list_outside, proc_mask = [None] * len(proc_lists), [None] * len(proc_lists), [None] * len(proc_lists)
    for frame_idx, frame_group in enumerate(proc_lists): 
        proc_list[frame_idx], proc_list_outside[frame_idx], proc_mask[frame_idx] = frame_group
    prep_prephase = None
    video = None
    mask_video = None

    if preproc2 != None:
        proc_list2 = process_images_multithread(preproc2, proc_list, process_type2, max_workers=max_workers)
        #### to be finished ...or not
    proc_list = process_images_multithread(preproc, proc_list, process_type, max_workers=max_workers)
    if any_mask:
        proc_list_outside = process_images_multithread(preproc_outside, proc_list_outside, process_outside_mask, max_workers=max_workers)
    else:
        proc_list_outside = proc_mask = len(proc_list) * [None]

    masked_frames = []
    masks = []
    for frame_no, (processed_img, processed_img_outside, mask) in enumerate(zip(proc_list, proc_list_outside, proc_mask)):
        if any_mask :
            masked_frame = np.where(mask[..., None], processed_img, processed_img_outside)
            if process_outside_mask != None:
                mask = np.full_like(mask, 255)
            mask = torch.from_numpy(mask)
            if RGB_Mask:
                mask =  mask.unsqueeze(-1).repeat(1,1,3)
            if outpainting_dims != None:
                full_frame= torch.full( (final_height, final_width, mask.shape[-1]), 255, dtype= torch.uint8, device= mask.device)
                full_frame[margin_top:margin_top+height, margin_left:margin_left+width] = mask
                mask = full_frame 
            masks.append(mask[:, :, 0:1].clone())
        else:
            masked_frame = processed_img

        if isinstance(masked_frame, int):
            masked_frame= np.full( (height, width, 3), inpaint_color, dtype= np.uint8)

        masked_frame = torch.from_numpy(masked_frame)
        if masked_frame.shape[-1] == 1:
            masked_frame =  masked_frame.repeat(1,1,3).to(torch.uint8)

        if outpainting_dims != None:
            full_frame= torch.full( (final_height, final_width, masked_frame.shape[-1]),  inpaint_color, dtype= torch.uint8, device= masked_frame.device)
            full_frame[margin_top:margin_top+height, margin_left:margin_left+width] = masked_frame
            masked_frame = full_frame 

        masked_frames.append(masked_frame)
        proc_list[frame_no] = proc_list_outside[frame_no] = proc_mask[frame_no] = None


    # if args.save_masks:
    #     from preprocessing.dwpose.pose import save_one_video
    #     saved_masked_frames = [mask.cpu().numpy() for mask in masked_frames ]
    #     save_one_video(f"masked_frames{'' if proc_no==1 else str(proc_no)}.mp4", saved_masked_frames, fps=target_fps, quality=8, macro_block_size=None)
    #     if any_mask:
    #         saved_masks = [mask.cpu().numpy() for mask in masks ]
    #         save_one_video("masks.mp4", saved_masks, fps=target_fps, quality=8, macro_block_size=None)
    preproc = None
    preproc_outside = None
    gc.collect()
    torch.cuda.empty_cache()
    if pad_frames > 0:
        masked_frames = masked_frames[0] * pad_frames + masked_frames
        if any_mask: masked_frames = masks[0] * pad_frames + masks
    masked_frames = torch.stack(masked_frames).permute(-1,0,1,2).float().div_(127.5).sub_(1.)
    masks = torch.stack(masks).permute(-1,0,1,2).float().div_(255) if any_mask else None

    return masked_frames, masks

def preprocess_video(height, width, video_in, max_frames, start_frame=0, fit_canvas = None, fit_crop = False, target_fps = 16, block_size = 16):

    frames_list = get_resampled_video(video_in, start_frame, max_frames, target_fps)

    if len(frames_list) == 0:
        return None

    if fit_canvas == None or fit_crop:
        new_height = height
        new_width = width
    else:
        frame_height, frame_width, _ = frames_list[0].shape
        if fit_canvas :
            scale1  = min(height / frame_height, width /  frame_width)
            scale2  = min(height / frame_width, width /  frame_height)
            scale = max(scale1, scale2)
        else:
            scale =   ((height * width ) /  (frame_height * frame_width))**(1/2)

        new_height = (int(frame_height * scale) // block_size) * block_size
        new_width = (int(frame_width * scale) // block_size) * block_size

    processed_frames_list = []
    for frame in frames_list:
        frame = Image.fromarray(np.clip(frame.cpu().numpy(), 0, 255).astype(np.uint8))
        if fit_crop:
            frame  = rescale_and_crop(frame, new_width, new_height)
        else:
            frame = frame.resize((new_width,new_height), resample=Image.Resampling.LANCZOS) 
        processed_frames_list.append(frame)

    np_frames = [np.array(frame) for frame in processed_frames_list]

    # from preprocessing.dwpose.pose import save_one_video
    # save_one_video("test.mp4", np_frames, fps=8, quality=8, macro_block_size=None)

    torch_frames = []
    for np_frame in np_frames:
        torch_frame = torch.from_numpy(np_frame)
        torch_frames.append(torch_frame)

    return torch.stack(torch_frames) 

 
def parse_keep_frames_video_guide(keep_frames, video_length):
        
    def absolute(n):
        if n==0:
            return 0
        elif n < 0:
            return max(0, video_length + n)
        else:
            return min(n-1, video_length-1)
    keep_frames = keep_frames.strip()
    if len(keep_frames) == 0:
        return [True] *video_length, "" 
    frames =[False] *video_length
    error = ""
    sections = keep_frames.split(" ")
    for section in sections:
        section = section.strip()
        if ":" in section:
            parts = section.split(":")
            if not is_integer(parts[0]):
                error =f"Invalid integer {parts[0]}"
                break
            start_range = absolute(int(parts[0]))
            if not is_integer(parts[1]):
                error =f"Invalid integer {parts[1]}"
                break
            end_range = absolute(int(parts[1]))
            for i in range(start_range, end_range + 1):
                frames[i] = True
        else:
            if not is_integer(section) or int(section) == 0:
                error =f"Invalid integer {section}"
                break
            index = absolute(int(section))
            frames[index] = True

    if len(error ) > 0:
        return [], error
    for i in range(len(frames)-1, 0, -1):
        if frames[i]:
            break
    frames= frames[0: i+1]
    return  frames, error


def perform_temporal_upsampling(sample, previous_last_frame, temporal_upsampling, fps):
    exp = 0
    if temporal_upsampling == "rife2":
        exp = 1
    elif temporal_upsampling == "rife4":
        exp = 2
    output_fps = fps
    if exp > 0: 
        from postprocessing.rife.inference import temporal_interpolation
        if previous_last_frame != None:
            sample = torch.cat([previous_last_frame, sample], dim=1)
            previous_last_frame = sample[:, -1:].clone()
            sample = temporal_interpolation( os.path.join("ckpts", "flownet.pkl"), sample, exp, device=processing_device)
            sample = sample[:, 1:]
        else:
            sample = temporal_interpolation( os.path.join("ckpts", "flownet.pkl"), sample, exp, device=processing_device)
            previous_last_frame = sample[:, -1:].clone()

        output_fps = output_fps * 2**exp
    return sample, previous_last_frame, output_fps 


def perform_spatial_upsampling(sample, spatial_upsampling):
    from shared.utils.utils import resize_lanczos 
    if spatial_upsampling == "lanczos1.5":
        scale = 1.5
    else:
        scale = 2
    h, w = sample.shape[-2:]
    h *= scale
    h = round(h/16) * 16
    w *= scale
    w = round(w/16) * 16
    h = int(h)
    w = int(w)
    frames_to_upsample = [sample[:, i] for i in range( sample.shape[1]) ] 
    def upsample_frames(frame):
        return resize_lanczos(frame, h, w).unsqueeze(1)
    sample = torch.cat(process_images_multithread(upsample_frames, frames_to_upsample, "upsample", wrap_in_list = False, max_workers=get_default_workers()), dim=1)
    frames_to_upsample = None
    return sample 

def any_audio_track(model_type):
    base_model_type = get_base_model_type(model_type)
    return base_model_type in ["fantasy", "hunyuan_avatar", "hunyuan_custom_audio"] or get_model_def(model_type).get("multitalk_class", False)

def get_available_filename(target_path, video_source, suffix = "", force_extension = None):
    name, extension =  os.path.splitext(os.path.basename(video_source))
    if force_extension != None:
        extension = force_extension
    name+= suffix
    full_path= os.path.join(target_path, f"{name}{extension}")
    if not os.path.exists(full_path):
        return full_path
    counter = 2
    while True:
        full_path= os.path.join(target_path, f"{name}({counter}){extension}")
        if not os.path.exists(full_path):
            return full_path
        counter += 1

def set_seed(seed):
    import random
    seed = random.randint(0, 99999999) if seed == None or seed < 0 else seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed

def edit_video(
                send_cmd,
                state,
                mode,
                video_source,
                seed,   
                temporal_upsampling,
                spatial_upsampling,
                film_grain_intensity,
                film_grain_saturation,
                MMAudio_setting,
                MMAudio_prompt,
                MMAudio_neg_prompt,
                repeat_generation,
                audio_source,
                **kwargs
                ):



    gen = get_gen_info(state)

    if gen.get("abort", False): return 
    abort = False
		
		
	
    configs, _ = get_settings_from_file(state, video_source, False, False, False)
    if configs == None: configs = { "type" : get_model_record("Post Processing") }

    has_already_audio = False
    audio_tracks = []
    if MMAudio_setting == 0:
        audio_tracks, audio_metadata  = extract_audio_tracks(video_source)
        has_already_audio = len(audio_tracks) > 0
    
    if audio_source is not None:
        audio_tracks = [audio_source]

    with lock:
        file_list = gen["file_list"]
        file_settings_list = gen["file_settings_list"]



    seed = set_seed(seed)

    from shared.utils.utils import get_video_info
    fps, width, height, frames_count = get_video_info(video_source)        
    frames_count = min(frames_count, max_source_video_frames)
    sample = None

    if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 or film_grain_intensity > 0:                
        send_cmd("progress", [0, get_latest_status(state,"Upsampling" if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 else "Adding Film Grain"  )])
        sample = get_resampled_video(video_source, 0, max_source_video_frames, fps)
        sample = sample.float().div_(127.5).sub_(1.).permute(-1,0,1,2)
        frames_count = sample.shape[1] 

    output_fps  = round(fps)
    if len(temporal_upsampling) > 0:
        sample, previous_last_frame, output_fps = perform_temporal_upsampling(sample, None, temporal_upsampling, fps)
        configs["temporal_upsampling"] = temporal_upsampling
        frames_count = sample.shape[1] 


    if len(spatial_upsampling) > 0:
        sample = perform_spatial_upsampling(sample, spatial_upsampling )
        configs["spatial_upsampling"] = spatial_upsampling

    if film_grain_intensity > 0:
        from postprocessing.film_grain import add_film_grain
        sample = add_film_grain(sample, film_grain_intensity, film_grain_saturation) 
        configs["film_grain_intensity"] = film_grain_intensity
        configs["film_grain_saturation"] = film_grain_saturation

    any_mmaudio = MMAudio_setting != 0 and server_config.get("mmaudio_enabled", 0) != 0 and frames_count >=output_fps
    if any_mmaudio: download_mmaudio()

    tmp_path = None
    any_change = False
    if sample != None:
        video_path =get_available_filename(save_path, video_source, "_tmp") if any_mmaudio or has_already_audio else get_available_filename(save_path, video_source, "_post")  
        save_video( tensor=sample[None], save_file=video_path, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1), codec_type= server_config.get("video_output_codec", None))

        if any_mmaudio or has_already_audio: tmp_path = video_path
        any_change = True
    else:
        video_path = video_source

    repeat_no = 0
    extra_generation = 0
    initial_total_windows = 0
    any_change_initial = any_change
    while not gen.get("abort", False): 
        any_change = any_change_initial
        extra_generation += gen.get("extra_orders",0)
        gen["extra_orders"] = 0
        total_generation = repeat_generation + extra_generation
        gen["total_generation"] = total_generation         
        if repeat_no >= total_generation: break
        repeat_no +=1
        gen["repeat_no"] = repeat_no
        suffix =  "" if "_post" in video_source else "_post"

        if audio_source is not None:
            audio_prompt_type = configs.get("audio_prompt_type", "")
            if not "T" in audio_prompt_type:audio_prompt_type += "T"
            configs["audio_prompt_type"] = audio_prompt_type
            any_change = True

        if any_mmaudio:
            send_cmd("progress", [0, get_latest_status(state,"MMAudio Soundtrack Generation")])
            from postprocessing.mmaudio.mmaudio import video_to_audio
            new_video_path = get_available_filename(save_path, video_source, suffix)
            video_to_audio(video_path, prompt = MMAudio_prompt, negative_prompt = MMAudio_neg_prompt, seed = seed, num_steps = 25, cfg_strength = 4.5, duration= frames_count /output_fps, save_path = new_video_path , persistent_models = server_config.get("mmaudio_enabled", 0) == 2, verboseLevel = verbose_level)
            configs["MMAudio_setting"] = MMAudio_setting
            configs["MMAudio_prompt"] = MMAudio_prompt
            configs["MMAudio_neg_prompt"] = MMAudio_neg_prompt
            configs["MMAudio_seed"] = seed
            any_change = True
        elif len(audio_tracks) > 0:
            # combine audio files and new video file
            new_video_path = get_available_filename(save_path, video_source, suffix)
            combine_video_with_audio_tracks(video_path, audio_tracks, new_video_path, audio_metadata=audio_metadata)
        else:
            new_video_path = video_path
        if tmp_path != None:
            os.remove(tmp_path)

        if any_change:
            if mode == "edit_remux":
                print(f"Remuxed Video saved to Path: "+ new_video_path)
            else:
                print(f"Postprocessed video saved to Path: "+ new_video_path)
            with lock:
                file_list.append(new_video_path)
                file_settings_list.append(configs)

            if configs != None:    
                from mutagen.mp4 import MP4
                file = MP4(new_video_path)
                file.tags['©cmt'] = [json.dumps(configs)]
                file.save()        

            send_cmd("output")
            seed = set_seed(-1)
    if has_already_audio:
        cleanup_temp_audio_files(audio_tracks)
    clear_status(state)

def get_overridden_attention(model_type):
    model_def = get_model_def(model_type)
    override_attention = model_def.get("attention", None)
    if override_attention is None: return None
    gpu_version = gpu_major * 10 + gpu_minor
    attention_list = match_nvidia_architecture(override_attention, gpu_version) 
    if len(attention_list ) == 0: return None
    override_attention = attention_list[0]
    if override_attention is not None and override_attention not in attention_modes_supported: return None
    return override_attention

def get_transformer_loras(model_type):
    model_def = get_model_def(model_type)
    transformer_loras_filenames = get_model_recursive_prop(model_type, "loras", return_list=True)
    lora_dir = get_lora_dir(model_type)
    transformer_loras_filenames = [ os.path.join(lora_dir, os.path.basename(filename)) for filename in transformer_loras_filenames]
    transformer_loras_multipliers = get_model_recursive_prop(model_type, "loras_multipliers", return_list=True) + [1.] * len(transformer_loras_filenames)
    transformer_loras_multipliers = transformer_loras_multipliers[:len(transformer_loras_filenames)]
    return transformer_loras_filenames, transformer_loras_multipliers

class DynamicClass:
    def __init__(self, **kwargs):
        self._data = {}
        # Preassign default properties from kwargs
        for key, value in kwargs.items():
            self._data[key] = value
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_data'):
                super().__setattr__('_data', {})
            self._data[name] = value
    
    def assign(self, **kwargs):
        """Assign multiple properties at once"""
        for key, value in kwargs.items():
            self._data[key] = value
        return self  # For method chaining
    
    def update(self, dict):
        """Alias for assign() - more dict-like"""
        return self.assign(**dict)


def process_prompt_enhancer(prompt_enhancer, original_prompts,  image_start, original_image_refs, is_image, seed ):

    text_encoder_max_tokens = 256
    from models.ltx_video.utils.prompt_enhance_utils import generate_cinematic_prompt
    prompt_images = []
    if "I" in prompt_enhancer:
        if image_start != None:
            prompt_images += image_start
        if original_image_refs != None:
            prompt_images += original_image_refs[:1]
    prompt_images = [Image.open(img) if isinstance(img,str) else img for img in prompt_images]
    if len(original_prompts) == 0 and not "T" in prompt_enhancer:
        return None
    else:
        from shared.utils.utils import seed_everything
        seed = seed_everything(seed)
        # for i, original_prompt in enumerate(original_prompts):
        prompts = generate_cinematic_prompt(
            prompt_enhancer_image_caption_model,
            prompt_enhancer_image_caption_processor,
            prompt_enhancer_llm_model,
            prompt_enhancer_llm_tokenizer,
            original_prompts if "T" in prompt_enhancer else ["an image"],
            prompt_images if len(prompt_images) > 0 else None,
            video_prompt = not is_image,
            max_new_tokens=text_encoder_max_tokens,
        )
        return prompts

def enhance_prompt(state, prompt, prompt_enhancer, multi_images_gen_type, override_profile,  progress=gr.Progress()):
    global enhancer_offloadobj
    prefix = "#!PROMPT!:"
    model_type = state["model_type"]
    inputs = get_model_settings(state, model_type)
    original_prompts = inputs["prompt"]

    original_prompts, errors = prompt_parser.process_template(original_prompts, keep_comments= True)
    if len(errors) > 0:
        gr.Info("Error processing prompt template: " + errors)
        return gr.update(), gr.update()
    original_prompts = original_prompts.replace("\r", "").split("\n")

    prompts_to_process = []
    skip_next_non_comment = False
    for prompt in original_prompts:
        if prompt.startswith(prefix):
            new_prompt = prompt[len(prefix):].strip()
            prompts_to_process.append(new_prompt)
            skip_next_non_comment = True
        else:
            if not prompt.startswith("#") and not skip_next_non_comment and len(prompt) > 0:
                prompts_to_process.append(prompt)
            skip_next_non_comment = False

    original_prompts = prompts_to_process
    num_prompts = len(original_prompts) 
    image_start = inputs["image_start"]
    if image_start is None or not "I" in prompt_enhancer:
        image_start = [None] * num_prompts
    else:
        image_start = [img[0] for img in image_start]
        if len(image_start) == 1:
            image_start = image_start * num_prompts
        else:
            if multi_images_gen_type !=1:
                gr.Info("On Demand Prompt Enhancer with multiple Start Images requires that option 'Match images and text prompts' is set")
                return gr.update(), gr.update()

            if len(image_start) != num_prompts:
                gr.Info("On Demand Prompt Enhancer supports only mutiple Start Images if their number matches the number of Text Prompts")
                return gr.update(), gr.update()
 
    if enhancer_offloadobj is None:
        status = "Please Wait While Loading Prompt Enhancer"
        progress(0, status)
        kwargs = {}
        pipe = {}
        download_models()

    gen = get_gen_info(state)
    original_process_status = None
    while True:
        with gen_lock:
            process_status = gen.get("process_status", None)
            if process_status is None:
                original_process_status = process_status 
                gen["process_status"] = "process:prompt_enhancer"
                break
            elif process_status == "process:main":
                original_process_status = process_status 
                gen["process_status"] = "request:prompt_enhancer"
                gen["pause_msg"] = "Generation Suspended while using Prompt Enhancer"
                break
            elif process_status == "process:prompt_enhancer":
                break
        time.sleep(1)

    if original_process_status is not None:
        while True:
            with gen_lock:
                process_status = gen.get("process_status", None)
                if process_status == "process:prompt_enhancer": break
                if process_status is None:
                    # handle case when main process has finished at some point in between the last check and now
                    gen["process_status"] = "process:prompt_enhancer"
                    break
            time.sleep(0.1)

    if enhancer_offloadobj is None:
        profile = init_pipe(pipe, kwargs, override_profile)
        setup_prompt_enhancer(pipe, kwargs)
        enhancer_offloadobj = offload.profile(pipe, profile_no= profile, **kwargs)  

    original_image_refs = inputs["image_refs"]
    if original_image_refs is not None:
        original_image_refs = [ convert_image(tup[0]) for tup in original_image_refs ]        
    is_image = inputs["image_mode"] > 0
    seed = inputs["seed"]
    seed = set_seed(seed)
    enhanced_prompts = []
    for i, (one_prompt, one_image) in enumerate(zip(original_prompts, image_start)):
        start_images = [one_image] if one_image is not None else None
        status = f'Please Wait While Enhancing Prompt' if num_prompts==1 else f'Please Wait While Enhancing Prompt #{i+1}'
        progress((i , num_prompts), desc=status, total= num_prompts)

        try:
            enhanced_prompt = process_prompt_enhancer(prompt_enhancer, [one_prompt],  start_images, original_image_refs, is_image, seed )    
        except Exception as e:
            enhancer_offloadobj.unload_all()
            with gen_lock:
                gen["process_status"] = original_process_status
            raise gr.Error(e)
        if enhanced_prompt is not None:
            enhanced_prompt = enhanced_prompt[0].replace("\n", "").replace("\r", "")
            enhanced_prompts.append(prefix + " " + one_prompt)
            enhanced_prompts.append(enhanced_prompt)

    enhancer_offloadobj.unload_all()
    with gen_lock:
        gen["process_status"] = original_process_status

    prompt = '\n'.join(enhanced_prompts)
    if num_prompts > 1:
        gr.Info(f'{num_prompts} Prompts have been Enhanced')
    else:
        gr.Info(f'Prompt "{original_prompts[0][:100]}" has been enhanced')
    return prompt, prompt

def get_outpainting_dims(video_guide_outpainting):
    return None if video_guide_outpainting== None or len(video_guide_outpainting) == 0 or video_guide_outpainting == "0 0 0 0" or video_guide_outpainting.startswith("#") else [int(v) for v in video_guide_outpainting.split(" ")] 

def generate_video(
    task,
    send_cmd,
    image_mode,
    prompt,
    negative_prompt,    
    resolution,
    video_length,
    batch_size,
    seed,
    force_fps,
    num_inference_steps,
    guidance_scale,
    guidance2_scale,
    guidance3_scale,
    switch_threshold,
    switch_threshold2,
    guidance_phases,
    model_switch_phase,
    audio_guidance_scale,
    flow_shift,
    sample_solver,
    embedded_guidance_scale,
    repeat_generation,
    multi_prompts_gen_type,
    multi_images_gen_type,
    skip_steps_cache_type,
    skip_steps_multiplier,
    skip_steps_start_step_perc,    
    activated_loras,
    loras_multipliers,
    image_prompt_type,
    image_start,
    image_end,
    model_mode,
    video_source,
    keep_frames_video_source,
    video_prompt_type,
    image_refs,
    frames_positions,
    video_guide,
    image_guide,
    keep_frames_video_guide,
    denoising_strength,
    video_guide_outpainting,
    video_mask,
    image_mask,
    control_net_weight,
    control_net_weight2,
    mask_expand,
    audio_guide,
    audio_guide2,
    audio_source,
    audio_prompt_type,
    speakers_locations,
    sliding_window_size,
    sliding_window_overlap,
    sliding_window_color_correction_strength,
    sliding_window_overlap_noise,
    sliding_window_discard_last_frames,
    image_refs_relative_size,
    remove_background_images_ref,
    temporal_upsampling,
    spatial_upsampling,
    film_grain_intensity,
    film_grain_saturation,
    MMAudio_setting,
    MMAudio_prompt,
    MMAudio_neg_prompt,    
    RIFLEx_setting,
    NAG_scale,
    NAG_tau,
    NAG_alpha,
    slg_switch,
    slg_layers,    
    slg_start_perc,
    slg_end_perc,
    apg_switch,
    cfg_star_switch,
    cfg_zero_step,
    prompt_enhancer,
    min_frames_if_references,
    override_profile,
    state,
    model_type,
    model_filename,
    mode,
):



    def remove_temp_filenames(temp_filenames_list):
        for temp_filename in temp_filenames_list: 
            if temp_filename!= None and os.path.isfile(temp_filename):
                os.remove(temp_filename)

    process_map_outside_mask = { "Y" : "depth", "W": "scribble", "X": "inpaint", "Z": "flow"}
    process_map_video_guide = { "P": "pose", "D" : "depth", "S": "scribble", "E": "canny", "L": "flow", "C": "gray", "M": "inpaint", "U": "identity"}
    processes_names = { "pose": "Open Pose", "depth": "Depth Mask", "scribble" : "Shapes", "flow" : "Flow Map", "gray" : "Gray Levels", "inpaint" : "Inpaint Mask", "identity": "Identity Mask", "raw" : "Raw Format", "canny" : "Canny Edges"}

    global wan_model, offloadobj, reload_needed
    gen = get_gen_info(state)
    torch.set_grad_enabled(False) 
    if mode.startswith("edit_"):
        edit_video(send_cmd, state, mode, video_source, seed, temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation, MMAudio_setting, MMAudio_prompt, MMAudio_neg_prompt, repeat_generation, audio_source)
        return
    with lock:
        file_list = gen["file_list"]
        file_settings_list = gen["file_settings_list"]


    model_def = get_model_def(model_type) 
    is_image = image_mode > 0
    if is_image:
        if min_frames_if_references >= 1000:
            video_length = min_frames_if_references - 1000
        else:
            video_length = min_frames_if_references if "I" in video_prompt_type or "V" in video_prompt_type else 1 
    else:
        batch_size = 1
    temp_filenames_list = []

    if image_guide is not None and isinstance(image_guide, Image.Image):
        video_guide = image_guide
        image_guide = None

    if image_mask is not None and isinstance(image_mask, Image.Image):
        video_mask = image_mask
        image_mask = None

    if model_def.get("no_background_removal", False): remove_background_images_ref = 0
    
    base_model_type = get_base_model_type(model_type)
    model_family = get_model_family(base_model_type)
    model_handler = get_model_handler(base_model_type)
    block_size = model_handler.get_vae_block_size(base_model_type) if hasattr(model_handler, "get_vae_block_size") else 16

    if "P" in preload_model_policy and not "U" in preload_model_policy:
        while wan_model == None:
            time.sleep(1)
        
    if model_type !=  transformer_type or reload_needed or override_profile>0 and override_profile != loaded_profile or override_profile<0 and default_profile != loaded_profile:
        wan_model = None
        release_model()
        send_cmd("status", f"Loading model {get_model_name(model_type)}...")
        wan_model, offloadobj = load_models(model_type, override_profile)
        send_cmd("status", "Model loaded")
        reload_needed=  False
    overridden_attention = get_overridden_attention(model_type)
    # if overridden_attention is not None and overridden_attention !=  attention_mode: print(f"Attention mode has been overriden to {overridden_attention} for model type '{model_type}'")
    attn = overridden_attention if overridden_attention is not None else attention_mode
    if attn == "auto":
        attn = get_auto_attention()
    elif not attn in attention_modes_supported:
        send_cmd("info", f"You have selected attention mode '{attention_mode}'. However it is not installed or supported on your system. You should either install it or switch to the default 'sdpa' attention.")
        send_cmd("exit")
        return
    
    width, height = resolution.split("x")
    width, height = int(width) // block_size *  block_size, int(height) // block_size *  block_size
    default_image_size = (height, width)

    if slg_switch == 0:
        slg_layers = None

    offload.shared_state["_attention"] =  attn
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
    if hasattr(wan_model.vae, "get_VAE_tile_size"):
        VAE_tile_size = wan_model.vae.get_VAE_tile_size(vae_config, device_mem_capacity, server_config.get("vae_precision", "16") == "32")
    else:
        VAE_tile_size = None

    trans = get_transformer_model(wan_model)
    trans2 = get_transformer_model(wan_model, 2)
    audio_sampling_rate = 16000

    prompts = prompt.split("\n")
    prompts = [part for part in prompts if len(prompt)>0]
    parsed_keep_frames_video_source= max_source_video_frames if len(keep_frames_video_source) ==0 else int(keep_frames_video_source) 
    transformer_loras_filenames, transformer_loras_multipliers  = get_transformer_loras(model_type)
    if guidance_phases < 1: guidance_phases = 1
    if transformer_loras_filenames != None:
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(transformer_loras_multipliers, len(transformer_loras_filenames), num_inference_steps, nb_phases = guidance_phases )
        if len(errors) > 0: raise Exception(f"Error parsing Transformer Loras: {errors}")
        loras_selected = transformer_loras_filenames 

    if hasattr(wan_model, "get_loras_transformer"):
        extra_loras_transformers, extra_loras_multipliers = wan_model.get_loras_transformer(get_model_recursive_prop, **locals())
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(extra_loras_multipliers, len(extra_loras_transformers), num_inference_steps, nb_phases = guidance_phases, merge_slist= loras_slists )
        if len(errors) > 0: raise Exception(f"Error parsing Extra Transformer Loras: {errors}")
        loras_selected += extra_loras_transformers 

    loras = state["loras"]
    if len(loras) > 0:
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(loras_multipliers, len(activated_loras), num_inference_steps, nb_phases = guidance_phases, merge_slist= loras_slists )
        if len(errors) > 0: raise Exception(f"Error parsing Loras: {errors}")
        lora_dir = get_lora_dir(model_type)
        loras_selected += [ os.path.join(lora_dir, lora) for lora in activated_loras]

    if len(loras_selected) > 0:
        pinnedLora = loaded_profile !=5  # and transformer_loras_filenames == None False # # # 
        split_linear_modules_map = getattr(trans,"split_linear_modules_map", None)
        offload.load_loras_into_model(trans , loras_selected, loras_list_mult_choices_nums, activate_all_loras=True, preprocess_sd=get_loras_preprocessor(trans, base_model_type), pinnedLora=pinnedLora, split_linear_modules_map = split_linear_modules_map) 
        errors = trans._loras_errors
        if len(errors) > 0:
            error_files = [msg for _ ,  msg  in errors]
            raise gr.Error("Error while loading Loras: " + ", ".join(error_files))
        if trans2 is not None: 
            offload.sync_models_loras(trans, trans2)
        
    seed = None if seed == -1 else seed
    # negative_prompt = "" # not applicable in the inference
    original_filename = model_filename 
    model_filename = get_model_filename(base_model_type)  

    current_video_length = video_length
    # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(None).total_memory / 1048576
    guide_inpaint_color = model_def.get("guide_inpaint_color", 127.5)
    extract_guide_from_window_start = model_def.get("extract_guide_from_window_start", False) 
    hunyuan_custom = "hunyuan_video_custom" in model_filename
    hunyuan_custom_audio =  hunyuan_custom and "audio" in model_filename
    hunyuan_custom_edit =  hunyuan_custom and "edit" in model_filename
    hunyuan_avatar = "hunyuan_video_avatar" in model_filename
    fantasy = base_model_type in ["fantasy"]
    multitalk = model_def.get("multitalk_class", False)

    if "B" in audio_prompt_type or "X" in audio_prompt_type:
        from models.wan.multitalk.multitalk import parse_speakers_locations
        speakers_bboxes, error = parse_speakers_locations(speakers_locations)
    else:
        speakers_bboxes = None        
    if "L" in image_prompt_type:
        if len(file_list)>0:
            video_source = file_list[-1]
        else:
            mp4_files = glob.glob(os.path.join(save_path, "*.mp4"))
            video_source = max(mp4_files, key=os.path.getmtime) if mp4_files else None                            

    fps = get_computed_fps(force_fps, base_model_type , video_guide, video_source )
    control_audio_tracks = source_audio_tracks = source_audio_metadata = []
    if "R" in audio_prompt_type and video_guide is not None and MMAudio_setting == 0 and not any_letters(audio_prompt_type, "ABX"):
        control_audio_tracks, _  = extract_audio_tracks(video_guide)
    if video_source is not None:
        source_audio_tracks, source_audio_metadata = extract_audio_tracks(video_source)
        video_fps, _, _, video_frames_count = get_video_info(video_source)
        video_source_duration = video_frames_count / video_fps
    else:
        video_source_duration = 0

    reset_control_aligment = "T" in video_prompt_type

    if test_any_sliding_window(model_type) :
        if video_source is not None:
            current_video_length +=  sliding_window_overlap - 1
        sliding_window = current_video_length > sliding_window_size
        reuse_frames = min(sliding_window_size - 4, sliding_window_overlap) 
    else:
        sliding_window = False
        sliding_window_size = current_video_length
        reuse_frames = 0

    _, _, latent_size = get_model_min_frames_and_step(model_type)  
    original_image_refs = image_refs
    image_refs = None if image_refs is None else [] + image_refs # work on a copy as it is going to be modified
    # image_refs = None
    # nb_frames_positions= 0
    # Output Video Ratio Priorities:
    # Source Video or Start Image > Control Video > Image Ref (background or positioned frames only) >  UI Width, Height
    # Image Ref (non background and non positioned frames) are boxed in a white canvas in order to keep their own width/height ratio
    frames_to_inject = []
    any_background_ref  = 0
    if "K" in video_prompt_type: 
        any_background_ref = 2 if model_def.get("all_image_refs_are_background_ref", False) else 1

    outpainting_dims = get_outpainting_dims(video_guide_outpainting)
    fit_canvas = server_config.get("fit_canvas", 0)
    fit_crop = fit_canvas == 2
    if fit_crop and outpainting_dims is not None:
        fit_crop = False
        fit_canvas = 0

    joint_pass = boost ==1 #and profile != 1 and profile != 3  
    
    skip_steps_cache = None if len(skip_steps_cache_type) == 0 else DynamicClass(cache_type = skip_steps_cache_type) 

    if skip_steps_cache != None:
        skip_steps_cache.update({     
        "multiplier" : skip_steps_multiplier,
        "start_step":  int(skip_steps_start_step_perc*num_inference_steps/100)
        })
        model_handler.set_cache_parameters(skip_steps_cache_type, base_model_type, model_def, locals(), skip_steps_cache)
        if skip_steps_cache_type == "mag":
            def_mag_ratios = model_def.get("magcache_ratios", None) if model_def != None else None
            if def_mag_ratios is not None: skip_steps_cache.def_mag_ratios = def_mag_ratios
        elif skip_steps_cache_type == "tea":
            def_tea_coefficients = model_def.get("teacache_coefficients", None) if model_def != None else None
            if def_tea_coefficients is not None: skip_steps_cache.coefficients = def_tea_coefficients
        else:
            raise Exception(f"unknown cache type {skip_steps_cache_type}")
    trans.cache = skip_steps_cache
    if trans2 is not None: trans2.cache = skip_steps_cache

    output_new_audio_data = None
    output_new_audio_filepath = None
    original_audio_guide = audio_guide
    original_audio_guide2 = audio_guide2
    audio_proj_split = None
    audio_proj_full = None
    audio_scale = None
    audio_context_lens = None
    if (fantasy or multitalk or hunyuan_avatar or hunyuan_custom_audio) and audio_guide != None:
        from models.wan.fantasytalking.infer import parse_audio
        from preprocessing.extract_vocals import get_vocals
        import librosa
        duration = librosa.get_duration(path=audio_guide)
        combination_type = "add"
        clean_audio_files = "V" in audio_prompt_type
        if audio_guide2 is not None:
            duration2 = librosa.get_duration(path=audio_guide2)
            if "C" in audio_prompt_type: duration += duration2
            else: duration = min(duration, duration2)
            combination_type = "para" if "P" in audio_prompt_type else "add" 
            if clean_audio_files:
                audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, audio_guide, "_clean", ".wav"))
                audio_guide2 = get_vocals(original_audio_guide2, get_available_filename(save_path, audio_guide2, "_clean2", ".wav"))
                temp_filenames_list += [audio_guide, audio_guide2]
        else:
            if "X" in audio_prompt_type: 
                # dual speaker, voice separation
                from preprocessing.speakers_separator import extract_dual_audio
                combination_type = "para"
                if args.save_speakers:
                    audio_guide, audio_guide2  = "speaker1.wav", "speaker2.wav"
                else:
                    audio_guide, audio_guide2  = get_available_filename(save_path, audio_guide, "_tmp1", ".wav"),  get_available_filename(save_path, audio_guide, "_tmp2", ".wav")
                    temp_filenames_list +=   [audio_guide, audio_guide2]                  
                if clean_audio_files:
                    clean_audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, original_audio_guide, "_clean", ".wav"))
                    temp_filenames_list += [clean_audio_guide]
                extract_dual_audio(clean_audio_guide if clean_audio_files else original_audio_guide, audio_guide, audio_guide2)

            elif clean_audio_files:
                # Single Speaker
                audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, audio_guide, "_clean", ".wav"))
                temp_filenames_list += [audio_guide]

            output_new_audio_filepath = original_audio_guide

        current_video_length = min(int(fps * duration //latent_size) * latent_size + latent_size + 1, current_video_length)
        if fantasy:
            # audio_proj_split_full, audio_context_lens_full = parse_audio(audio_guide, num_frames= max_source_video_frames, fps= fps,  padded_frames_for_embeddings= (reuse_frames if reset_control_aligment else 0), device= processing_device  )
            audio_scale = 1.0
        elif multitalk:
            from models.wan.multitalk.multitalk import get_full_audio_embeddings
            # pad audio_proj_full if aligned to beginning of window to simulate source window overlap
            min_audio_duration =  current_video_length/fps if reset_control_aligment else video_source_duration + current_video_length/fps
            audio_proj_full, output_new_audio_data = get_full_audio_embeddings(audio_guide1 = audio_guide, audio_guide2= audio_guide2, combination_type= combination_type , num_frames= max_source_video_frames, sr= audio_sampling_rate, fps =fps, padded_frames_for_embeddings = (reuse_frames if reset_control_aligment else 0), min_audio_duration = min_audio_duration) 
            if output_new_audio_data is not None: # not none if modified
                if clean_audio_files: # need to rebuild the sum of audios with original audio
                    _, output_new_audio_data = get_full_audio_embeddings(audio_guide1 = original_audio_guide, audio_guide2= original_audio_guide2, combination_type= combination_type , num_frames= max_source_video_frames, sr= audio_sampling_rate, fps =fps, padded_frames_for_embeddings = (reuse_frames if reset_control_aligment else 0), min_audio_duration = min_audio_duration, return_sum_only= True) 
                output_new_audio_filepath=  None # need to build original speaker track if it changed size (due to padding at the end) or if it has been combined

    if hunyuan_custom_edit and video_guide != None:
        import cv2
        cap = cv2.VideoCapture(video_guide)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_video_length = min(current_video_length, length)


    seed = set_seed(seed)

    torch.set_grad_enabled(False) 
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()
    wan_model._interrupt = False
    abort = False
    if gen.get("abort", False):
        return 
    # gen["abort"] = False
    gen["prompt"] = prompt    
    repeat_no = 0
    extra_generation = 0
    initial_total_windows = 0
    discard_last_frames = sliding_window_discard_last_frames
    default_requested_frames_to_generate = current_video_length
    nb_frames_positions = 0
    if sliding_window:
        initial_total_windows= compute_sliding_window_no(default_requested_frames_to_generate, sliding_window_size, discard_last_frames, reuse_frames) 
        current_video_length = sliding_window_size
    else:
        initial_total_windows = 1

    first_window_video_length = current_video_length
    original_prompts = prompts.copy()
    gen["sliding_window"] = sliding_window    
    while not abort: 
        extra_generation += gen.get("extra_orders",0)
        gen["extra_orders"] = 0
        total_generation = repeat_generation + extra_generation
        gen["total_generation"] = total_generation     
        gen["header_text"] = ""    
        if repeat_no >= total_generation: break
        repeat_no +=1
        gen["repeat_no"] = repeat_no
        src_video = src_video2 = src_mask = src_mask2 = src_faces = src_ref_images = src_ref_masks = sparse_video_image = None
        prefix_video = pre_video_frame = None
        source_video_overlap_frames_count = 0 # number of frames overalapped in source video for first window
        source_video_frames_count = 0  # number of frames to use in source video (processing starts source_video_overlap_frames_count frames before )
        frames_already_processed = None
        overlapped_latents = None
        context_scale = None
        window_no = 0
        extra_windows = 0
        guide_start_frame = 0 # pos of of first control video frame of current window  (reuse_frames later than the first processed frame)
        keep_frames_parsed = [] # aligned to the first control frame of current window (therefore ignore previous reuse_frames)
        pre_video_guide = None # reuse_frames of previous window
        image_size = default_image_size #  default frame dimensions for budget until it is change due to a resize
        sample_fit_canvas = fit_canvas
        current_video_length = first_window_video_length
        gen["extra_windows"] = 0
        gen["total_windows"] = 1
        gen["window_no"] = 1
        num_frames_generated = 0 # num of new frames created (lower than the number of frames really processed due to overlaps and discards)
        requested_frames_to_generate = default_requested_frames_to_generate # num  of num frames to create (if any source window this num includes also the overlapped source window frames)
        start_time = time.time()
        if prompt_enhancer_image_caption_model != None and prompt_enhancer !=None and len(prompt_enhancer)>0 and server_config.get("enhancer_mode", 0) == 0:
            send_cmd("progress", [0, get_latest_status(state, "Enhancing Prompt")])
            enhanced_prompts = process_prompt_enhancer(prompt_enhancer, original_prompts,  image_start, original_image_refs, is_image, seed )
            if enhanced_prompts is not None:
                print(f"Enhanced prompts: {enhanced_prompts}" )
                task["prompt"] = "\n".join(["!enhanced!"] + enhanced_prompts)
                send_cmd("output")
                prompt = enhanced_prompts[0]            
                abort = gen.get("abort", False)

        while not abort:
            enable_RIFLEx = RIFLEx_setting == 0 and current_video_length > (6* get_model_fps(base_model_type)+1) or RIFLEx_setting == 1
            prompt =  prompts[window_no] if window_no < len(prompts) else prompts[-1]
            new_extra_windows = gen.get("extra_windows",0)
            gen["extra_windows"] = 0
            extra_windows += new_extra_windows
            requested_frames_to_generate +=  new_extra_windows * (sliding_window_size - discard_last_frames - reuse_frames)
            sliding_window = sliding_window  or extra_windows > 0
            if sliding_window and window_no > 0:
                # num_frames_generated -= reuse_frames
                if (requested_frames_to_generate - num_frames_generated) <  latent_size:
                    break
                current_video_length = min(sliding_window_size, ((requested_frames_to_generate - num_frames_generated + reuse_frames + discard_last_frames) // latent_size) * latent_size + 1 )

            total_windows = initial_total_windows + extra_windows
            gen["total_windows"] = total_windows
            if window_no >= total_windows:
                break
            window_no += 1
            gen["window_no"] = window_no
            return_latent_slice = None 
            if reuse_frames > 0:                
                return_latent_slice = slice(-(reuse_frames - 1 + discard_last_frames ) // latent_size - 1, None if discard_last_frames == 0 else -(discard_last_frames // latent_size) )
            refresh_preview  = {"image_guide" : image_guide, "image_mask" : image_mask} if image_mode >= 1 else {}

            image_start_tensor = image_end_tensor = None
            if window_no == 1 and (video_source is not None or image_start is not None):
                if image_start is not None:
                    image_start_tensor, new_height, new_width = calculate_dimensions_and_resize_image(image_start, height, width, sample_fit_canvas, fit_crop, block_size = block_size)
                    if fit_crop: refresh_preview["image_start"] = image_start_tensor 
                    image_start_tensor = convert_image_to_tensor(image_start_tensor)
                    pre_video_guide =  prefix_video = image_start_tensor.unsqueeze(1)
                else:
                    prefix_video  = preprocess_video(width=width, height=height,video_in=video_source, max_frames= parsed_keep_frames_video_source , start_frame = 0, fit_canvas= sample_fit_canvas, fit_crop = fit_crop, target_fps = fps, block_size = block_size )
                    prefix_video  = prefix_video.permute(3, 0, 1, 2)
                    prefix_video  = prefix_video.float().div_(127.5).sub_(1.) # c, f, h, w
                    if fit_crop or "L" in image_prompt_type: refresh_preview["video_source"] = convert_tensor_to_image(prefix_video, 0) 

                    new_height, new_width = prefix_video.shape[-2:]                    
                    pre_video_guide =  prefix_video[:, -reuse_frames:]
                pre_video_frame = convert_tensor_to_image(prefix_video[:, -1])
                source_video_overlap_frames_count = pre_video_guide.shape[1]
                source_video_frames_count = prefix_video.shape[1]
                if sample_fit_canvas != None: 
                    image_size  = pre_video_guide.shape[-2:]
                    sample_fit_canvas = None
                guide_start_frame =  prefix_video.shape[1]
            if image_end is not None:
                image_end_list=  image_end if isinstance(image_end, list) else [image_end]
                if len(image_end_list) >= window_no:
                    new_height, new_width = image_size                    
                    image_end_tensor, _, _ = calculate_dimensions_and_resize_image(image_end_list[window_no-1], new_height, new_width, sample_fit_canvas, fit_crop, block_size = block_size)
                    # image_end_tensor =image_end_list[window_no-1].resize((new_width, new_height), resample=Image.Resampling.LANCZOS) 
                    refresh_preview["image_end"] = image_end_tensor 
                    image_end_tensor = convert_image_to_tensor(image_end_tensor)
                image_end_list= None
            window_start_frame = guide_start_frame - (reuse_frames if window_no > 1 else source_video_overlap_frames_count)
            guide_end_frame = guide_start_frame + current_video_length - (source_video_overlap_frames_count if window_no == 1 else reuse_frames)
            alignment_shift = source_video_frames_count if reset_control_aligment else 0
            aligned_guide_start_frame = guide_start_frame - alignment_shift
            aligned_guide_end_frame = guide_end_frame - alignment_shift
            aligned_window_start_frame = window_start_frame - alignment_shift  
            if fantasy:
                audio_proj_split , audio_context_lens = parse_audio(audio_guide, start_frame = aligned_window_start_frame, num_frames= current_video_length, fps= fps,  device= processing_device  )
            if multitalk:
                from models.wan.multitalk.multitalk import get_window_audio_embeddings
                # special treatment for start frame pos when alignement to first frame requested as otherwise the start frame number will be negative due to overlapped frames (has been previously compensated later with padding)
                audio_proj_split = get_window_audio_embeddings(audio_proj_full, audio_start_idx= aligned_window_start_frame + (source_video_overlap_frames_count if reset_control_aligment else 0 ), clip_length = current_video_length)

            if repeat_no == 1 and window_no == 1 and image_refs is not None and len(image_refs) > 0:
                frames_positions_list = []
                if frames_positions is not None and len(frames_positions)> 0:
                    positions = frames_positions.replace(","," ").split(" ")
                    cur_end_pos =  -1 + (source_video_frames_count - source_video_overlap_frames_count)
                    last_frame_no = requested_frames_to_generate + source_video_frames_count - source_video_overlap_frames_count
                    joker_used = False
                    project_window_no = 1
                    for pos in positions :
                        if len(pos) > 0:
                            if pos in ["L", "l"]:
                                cur_end_pos += sliding_window_size if project_window_no > 1 else current_video_length 
                                if cur_end_pos >= last_frame_no-1 and not joker_used:
                                    joker_used = True
                                    cur_end_pos = last_frame_no -1
                                project_window_no += 1
                                frames_positions_list.append(cur_end_pos)
                                cur_end_pos -= sliding_window_discard_last_frames + reuse_frames
                            else:
                                frames_positions_list.append(int(pos)-1 + alignment_shift)
                    frames_positions_list = frames_positions_list[:len(image_refs)]
                nb_frames_positions = len(frames_positions_list) 
                if nb_frames_positions > 0:
                    frames_to_inject = [None] * (max(frames_positions_list) + 1)
                    for i, pos in enumerate(frames_positions_list):
                        frames_to_inject[pos] = image_refs[i] 


            video_guide_processed = video_mask_processed = video_guide_processed2 = video_mask_processed2 = sparse_video_image = None
            if video_guide is not None:
                keep_frames_parsed_full, error = parse_keep_frames_video_guide(keep_frames_video_guide, source_video_frames_count -source_video_overlap_frames_count + requested_frames_to_generate)
                if len(error) > 0:
                    raise gr.Error(f"invalid keep frames {keep_frames_video_guide}")
                guide_frames_extract_start = aligned_window_start_frame if extract_guide_from_window_start else aligned_guide_start_frame
                extra_control_frames = model_def.get("extra_control_frames", 0)
                if extra_control_frames > 0 and aligned_guide_start_frame >= extra_control_frames: guide_frames_extract_start -= extra_control_frames
                        
                keep_frames_parsed = [True] * -guide_frames_extract_start if guide_frames_extract_start  <0 else []
                keep_frames_parsed += keep_frames_parsed_full[max(0, guide_frames_extract_start): aligned_guide_end_frame ] 
                guide_frames_extract_count = len(keep_frames_parsed)

                # Extract Faces to video
                if "B" in video_prompt_type:
                    send_cmd("progress", [0, get_latest_status(state, "Extracting Face Movements")])
                    src_faces = extract_faces_from_video_with_mask(video_guide, video_mask, max_frames= guide_frames_extract_count, start_frame= guide_frames_extract_start, size= 512, target_fps = fps)
                    if src_faces is not None and src_faces.shape[1] < current_video_length:
                        src_faces = torch.cat([src_faces, torch.full( (3, current_video_length - src_faces.shape[1], 512, 512 ), -1, dtype = src_faces.dtype, device= src_faces.device) ], dim=1)

                # Sparse Video to Video
                sparse_video_image = None
                if "R" in video_prompt_type:
                    sparse_video_image = get_video_frame(video_guide, aligned_guide_start_frame, return_last_if_missing = True, target_fps = fps, return_PIL = True)

                # Generic Video Preprocessing
                process_outside_mask = process_map_outside_mask.get(filter_letters(video_prompt_type, "YWX"), None)
                preprocess_type, preprocess_type2 =  "raw", None 
                for process_num, process_letter in enumerate( filter_letters(video_prompt_type, "PEDSLCMU")):
                    if process_num == 0:
                        preprocess_type = process_map_video_guide.get(process_letter, "raw")
                    else:
                        preprocess_type2 = process_map_video_guide.get(process_letter, None)
                status_info = "Extracting " + processes_names[preprocess_type]
                extra_process_list = ([] if preprocess_type2==None else [preprocess_type2]) + ([] if process_outside_mask==None or process_outside_mask == preprocess_type else [process_outside_mask])
                if len(extra_process_list) == 1:
                    status_info += " and " + processes_names[extra_process_list[0]]
                elif len(extra_process_list) == 2:
                    status_info +=  ", " + processes_names[extra_process_list[0]] + " and " + processes_names[extra_process_list[1]]
                context_scale = [control_net_weight /2, control_net_weight2 /2] if preprocess_type2 is not None else [control_net_weight]
                if not (preprocess_type == "identity" and preprocess_type2 is None and video_mask is None):send_cmd("progress", [0, get_latest_status(state, status_info)])
                inpaint_color = 0 if preprocess_type=="pose" and process_outside_mask == "inpaint" else guide_inpaint_color
                video_guide_processed, video_mask_processed = preprocess_video_with_mask(video_guide if sparse_video_image is None else sparse_video_image, video_mask, height=image_size[0], width = image_size[1], max_frames= guide_frames_extract_count, start_frame = guide_frames_extract_start, fit_canvas = sample_fit_canvas, fit_crop = fit_crop, target_fps = fps,  process_type = preprocess_type, expand_scale = mask_expand, RGB_Mask = True, negate_mask = "N" in video_prompt_type, process_outside_mask = process_outside_mask, outpainting_dims = outpainting_dims, proc_no =1, inpaint_color =inpaint_color, block_size = block_size )
                if preprocess_type2 != None:
                    video_guide_processed2, video_mask_processed2 = preprocess_video_with_mask(video_guide, video_mask, height=image_size[0], width = image_size[1], max_frames= guide_frames_extract_count, start_frame = guide_frames_extract_start, fit_canvas = sample_fit_canvas, fit_crop = fit_crop, target_fps = fps,  process_type = preprocess_type2, expand_scale = mask_expand, RGB_Mask = True, negate_mask = "N" in video_prompt_type, process_outside_mask = process_outside_mask, outpainting_dims = outpainting_dims, proc_no =2, block_size = block_size )

                if video_guide_processed is not None  and sample_fit_canvas is not None:
                    image_size = video_guide_processed.shape[-2:]
                    sample_fit_canvas = None

            if window_no == 1 and image_refs is not None and len(image_refs) > 0:
                if sample_fit_canvas is not None and (nb_frames_positions > 0 or "K" in video_prompt_type) :
                    from shared.utils.utils import get_outpainting_full_area_dimensions
                    w, h = image_refs[0].size
                    if outpainting_dims != None:
                        h, w = get_outpainting_full_area_dimensions(h,w, outpainting_dims)
                    image_size = calculate_new_dimensions(height, width, h, w, fit_canvas)
                sample_fit_canvas = None
                if repeat_no == 1:
                    if fit_crop:
                        if any_background_ref == 2:
                            end_ref_position = len(image_refs)
                        elif any_background_ref == 1:
                            end_ref_position = nb_frames_positions + 1
                        else:
                            end_ref_position = nb_frames_positions 
                        for i, img in enumerate(image_refs[:end_ref_position]):
                            image_refs[i] = rescale_and_crop(img, default_image_size[1], default_image_size[0])
                        refresh_preview["image_refs"] = image_refs

                    if len(image_refs) > nb_frames_positions:
                        src_ref_images = image_refs[nb_frames_positions:]
                        if remove_background_images_ref > 0:
                            send_cmd("progress", [0, get_latest_status(state, "Removing Images References Background")])

                        src_ref_images, src_ref_masks  = resize_and_remove_background(src_ref_images , image_size[1], image_size[0],
                                                                                        remove_background_images_ref > 0, any_background_ref, 
                                                                                        fit_into_canvas= model_def.get("fit_into_canvas_image_refs", 1),
                                                                                        block_size=block_size,
                                                                                        outpainting_dims =outpainting_dims,
                                                                                        background_ref_outpainted = model_def.get("background_ref_outpainted", True),
                                                                                        return_tensor= model_def.get("return_image_refs_tensor", False) )
 

            frames_to_inject_parsed = frames_to_inject[ window_start_frame if extract_guide_from_window_start else guide_start_frame: guide_end_frame]
            if video_guide is not None or len(frames_to_inject_parsed) > 0 or model_def.get("forced_guide_mask_inputs", False): 
                any_mask = video_mask is not None or model_def.get("forced_guide_mask_inputs", False)
                any_guide_padding = model_def.get("pad_guide_video", False)
                from shared.utils.utils import prepare_video_guide_and_mask
                src_videos, src_masks = prepare_video_guide_and_mask(   [video_guide_processed] + ([] if video_guide_processed2 is None else [video_guide_processed2]), 
                                                                        [video_mask_processed] + ([] if video_guide_processed2 is None else [video_mask_processed2]),
                                                                        None if extract_guide_from_window_start or model_def.get("dont_cat_preguide", False) or sparse_video_image is not None else pre_video_guide, 
                                                                        image_size, current_video_length, latent_size,
                                                                        any_mask, any_guide_padding, guide_inpaint_color, 
                                                                        keep_frames_parsed, frames_to_inject_parsed , outpainting_dims)
                video_guide_processed = video_guide_processed2 = video_mask_processed = video_mask_processed2 = None
                if len(src_videos) == 1:
                    src_video, src_video2, src_mask, src_mask2 = src_videos[0], None, src_masks[0], None 
                else:
                    src_video, src_video2 = src_videos 
                    src_mask, src_mask2 = src_masks 
                src_videos = src_masks = None
                if src_video is None:
                    abort = True 
                    break
                if src_faces is not None:
                    if src_faces.shape[1] < src_video.shape[1]:
                        src_faces = torch.concat( [src_faces,  src_faces[:, -1:].repeat(1, src_video.shape[1] - src_faces.shape[1], 1,1)], dim =1)
                    else:
                        src_faces = src_faces[:, :src_video.shape[1]]
                if video_guide is not None or len(frames_to_inject_parsed) > 0:
                    if args.save_masks:
                        if src_video is not None: 
                            save_video( src_video, "masked_frames.mp4", fps)
                            if any_mask: save_video( src_mask, "masks.mp4", fps, value_range=(0, 1))
                        if src_video2 is not None: 
                            save_video( src_video2, "masked_frames2.mp4", fps)
                            if any_mask: save_video( src_mask2, "masks2.mp4", fps, value_range=(0, 1))
                if video_guide is not None:                        
                    preview_frame_no = 0 if extract_guide_from_window_start or model_def.get("dont_cat_preguide", False) or sparse_video_image is not None else (guide_start_frame - window_start_frame) 
                    refresh_preview["video_guide"] = convert_tensor_to_image(src_video, preview_frame_no)
                    if src_video2 is not None:
                        refresh_preview["video_guide"] = [refresh_preview["video_guide"], convert_tensor_to_image(src_video2, preview_frame_no)] 
                    if src_mask is not None and video_mask is not None:                        
                        refresh_preview["video_mask"] = convert_tensor_to_image(src_mask, preview_frame_no, mask_levels = True)

            if src_ref_images is not None or nb_frames_positions:
                if len(frames_to_inject_parsed):
                    new_image_refs = [convert_tensor_to_image(src_video, frame_no + (0 if extract_guide_from_window_start else (aligned_guide_start_frame - aligned_window_start_frame)) ) for frame_no, inject in enumerate(frames_to_inject_parsed) if inject]
                else:
                    new_image_refs = []
                if src_ref_images is not None:
                    new_image_refs +=  [convert_tensor_to_image(img) if torch.is_tensor(img) else img for img in src_ref_images  ]
                refresh_preview["image_refs"] = new_image_refs
                new_image_refs = None

            if len(refresh_preview) > 0:
                new_inputs= locals()
                new_inputs.update(refresh_preview)
                update_task_thumbnails(task, new_inputs)
                send_cmd("output")

            if window_no ==  1:                
                conditioning_latents_size = ( (source_video_overlap_frames_count-1) // latent_size) + 1 if source_video_overlap_frames_count > 0 else 0
            else:
                conditioning_latents_size = ( (reuse_frames-1) // latent_size) + 1

            status = get_latest_status(state)
            gen["progress_status"] = status 
            gen["progress_phase"] = ("Encoding Prompt", -1 )
            callback = build_callback(state, trans, send_cmd, status, num_inference_steps)
            progress_args = [0, merge_status_context(status, "Encoding Prompt")]
            send_cmd("progress", progress_args)

            if skip_steps_cache !=  None:
                skip_steps_cache.update({
                "num_steps" : num_inference_steps,                
                "skipped_steps" : 0,
                "previous_residual": None,
                "previous_modulated_input":  None,
                })
            # samples = torch.empty( (1,2)) #for testing
            # if False:
            def set_header_text(txt):
                gen["header_text"] = txt
                send_cmd("output")

            try:
                samples = wan_model.generate(
                    input_prompt = prompt,
                    image_start = image_start_tensor,  
                    image_end = image_end_tensor,
                    input_frames = src_video,
                    input_frames2 = src_video2,
                    input_ref_images=  src_ref_images,
                    input_ref_masks = src_ref_masks,
                    input_masks = src_mask,
                    input_masks2 = src_mask2,
                    input_video= pre_video_guide,
                    input_faces = src_faces,
                    denoising_strength=denoising_strength,
                    prefix_frames_count = source_video_overlap_frames_count if window_no <= 1 else reuse_frames,
                    frame_num= (current_video_length // latent_size)* latent_size + 1,
                    batch_size = batch_size,
                    height = image_size[0],
                    width = image_size[1],
                    fit_into_canvas = fit_canvas,
                    shift=flow_shift,
                    sample_solver=sample_solver,
                    sampling_steps=num_inference_steps,
                    guide_scale=guidance_scale,
                    guide2_scale = guidance2_scale,
                    guide3_scale = guidance3_scale,
                    switch_threshold = switch_threshold, 
                    switch2_threshold = switch_threshold2,
                    guide_phases= guidance_phases,
                    model_switch_phase = model_switch_phase,
                    embedded_guidance_scale=embedded_guidance_scale,
                    n_prompt=negative_prompt,
                    seed=seed,
                    callback=callback,
                    enable_RIFLEx = enable_RIFLEx,
                    VAE_tile_size = VAE_tile_size,
                    joint_pass = joint_pass,
                    slg_layers = slg_layers,
                    slg_start = slg_start_perc/100,
                    slg_end = slg_end_perc/100,
                    apg_switch = apg_switch,
                    cfg_star_switch = cfg_star_switch,
                    cfg_zero_step = cfg_zero_step,
                    audio_cfg_scale= audio_guidance_scale,
                    audio_guide=audio_guide,
                    audio_guide2=audio_guide2,
                    audio_proj= audio_proj_split,
                    audio_scale= audio_scale,
                    audio_context_lens= audio_context_lens,
                    context_scale = context_scale,
                    model_mode = model_mode,
                    causal_block_size = 5,
                    causal_attention = True,
                    fps = fps,
                    overlapped_latents = overlapped_latents,
                    return_latent_slice= return_latent_slice,
                    overlap_noise = sliding_window_overlap_noise,
                    overlap_size = sliding_window_overlap,
                    color_correction_strength = sliding_window_color_correction_strength,
                    conditioning_latents_size = conditioning_latents_size,
                    keep_frames_parsed = keep_frames_parsed,
                    model_filename = model_filename,
                    model_type = base_model_type,
                    loras_slists = loras_slists,
                    NAG_scale = NAG_scale,
                    NAG_tau = NAG_tau,
                    NAG_alpha = NAG_alpha,
                    speakers_bboxes =speakers_bboxes,
                    image_mode =  image_mode,
                    video_prompt_type= video_prompt_type,
                    window_no = window_no, 
                    offloadobj = offloadobj,
                    set_header_text= set_header_text,
                    pre_video_frame = pre_video_frame,
                    original_input_ref_images = original_image_refs[nb_frames_positions:] if original_image_refs is not None else [],
                    image_refs_relative_size = image_refs_relative_size,
                    outpainting_dims = outpainting_dims,
                )
            except Exception as e:
                if len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0:
                    cleanup_temp_audio_files(control_audio_tracks + source_audio_tracks)
                remove_temp_filenames(temp_filenames_list)
                clear_gen_cache()
                offloadobj.unload_all()
                trans.cache = None 
                offload.unload_loras_from_model(trans)
                if trans2 is not None: 
                    trans2.cache = None 
                    offload.unload_loras_from_model(trans2)
                skip_steps_cache = None
                # if compile:
                #     cache_size = torch._dynamo.config.cache_size_limit                                      
                #     torch.compiler.reset()
                #     torch._dynamo.config.cache_size_limit = cache_size

                gc.collect()
                torch.cuda.empty_cache()
                s = str(e)
                keyword_list = {"CUDA out of memory" : "VRAM", "Tried to allocate":"VRAM", "CUDA error: out of memory": "RAM", "CUDA error: too many resources requested": "RAM"}
                crash_type = ""
                for keyword, tp  in keyword_list.items():
                    if keyword in s:
                        crash_type = tp 
                        break
                state["prompt"] = ""
                if crash_type == "VRAM":
                    new_error = "The generation of the video has encountered an error: it is likely that you have unsufficient VRAM and you should therefore reduce the video resolution or its number of frames."
                elif crash_type == "RAM":
                    new_error = "The generation of the video has encountered an error: it is likely that you have unsufficient RAM and / or Reserved RAM allocation should be reduced using 'perc_reserved_mem_max' or using a different Profile."
                else:
                    new_error =  gr.Error(f"The generation of the video has encountered an error, please check your terminal for more information. '{s}'")
                tb = traceback.format_exc().split('\n')[:-1] 
                print('\n'.join(tb))
                send_cmd("error", new_error)
                clear_status(state)
                return

            if skip_steps_cache != None :
                skip_steps_cache.previous_residual = None
                skip_steps_cache.previous_modulated_input = None
                print(f"Skipped Steps:{skip_steps_cache.skipped_steps}/{skip_steps_cache.num_steps}" )

            if samples != None:
                if isinstance(samples, dict):
                    overlapped_latents = samples.get("latent_slice", None)
                    samples= samples["x"]
                samples = samples.to("cpu")
            clear_gen_cache()
            offloadobj.unload_all()
            gc.collect()
            torch.cuda.empty_cache()

            # time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
            # save_prompt = "_in_" + original_prompts[0]
            # file_name = f"{time_flag}_seed{seed}_{sanitize_file_name(save_prompt[:50]).strip()}.mp4"
            # sample = samples.cpu()
            # cache_video( tensor=sample[None].clone(), save_file=os.path.join(save_path, file_name), fps=16, nrow=1, normalize=True, value_range=(-1, 1))

            if samples == None:
                abort = True
                state["prompt"] = ""
                send_cmd("output")  
            else:
                sample = samples.cpu()
                abort = not is_image and sample.shape[1] < current_video_length    
                # if True: # for testing
                #     torch.save(sample, "output.pt")
                # else:
                #     sample =torch.load("output.pt")
                if gen.get("extra_windows",0) > 0:
                    sliding_window = True 
                if sliding_window :
                    # guide_start_frame = guide_end_frame
                    guide_start_frame += current_video_length
                    if discard_last_frames > 0:
                        sample = sample[: , :-discard_last_frames]
                        guide_start_frame -= discard_last_frames
                    if reuse_frames == 0:
                        pre_video_guide =  sample[:,max_source_video_frames :].clone()
                    else:
                        pre_video_guide =  sample[:, -reuse_frames:].clone()


                if prefix_video != None and window_no == 1:
                    # remove source video overlapped frames at the beginning of the generation
                    sample = torch.cat([ prefix_video[:, :-source_video_overlap_frames_count], sample], dim = 1)
                    guide_start_frame -= source_video_overlap_frames_count 
                elif sliding_window and window_no > 1 and reuse_frames > 0:
                    # remove sliding window overlapped frames at the beginning of the generation
                    sample = sample[: , reuse_frames:]
                    guide_start_frame -= reuse_frames 

                num_frames_generated = guide_start_frame - (source_video_frames_count - source_video_overlap_frames_count) 

                if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0:                
                    send_cmd("progress", [0, get_latest_status(state,"Upsampling")])
                
                output_fps  = fps
                if len(temporal_upsampling) > 0:
                    sample, previous_last_frame, output_fps = perform_temporal_upsampling(sample, previous_last_frame if sliding_window and window_no > 1 else None, temporal_upsampling, fps)

                if len(spatial_upsampling) > 0:
                    sample = perform_spatial_upsampling(sample, spatial_upsampling )
                if film_grain_intensity> 0:
                    from postprocessing.film_grain import add_film_grain
                    sample = add_film_grain(sample, film_grain_intensity, film_grain_saturation) 
                if sliding_window :
                    if frames_already_processed == None:
                        frames_already_processed = sample
                    else:
                        sample = torch.cat([frames_already_processed, sample], dim=1)
                    frames_already_processed = sample

                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
                save_prompt = original_prompts[0]

                from shared.utils.utils import truncate_for_filesystem
                extension = "jpg" if is_image else "mp4" 

                if os.name == 'nt':
                    file_name = f"{time_flag}_seed{seed}_{sanitize_file_name(truncate_for_filesystem(save_prompt,50)).strip()}.{extension}"
                else:
                    file_name = f"{time_flag}_seed{seed}_{sanitize_file_name(truncate_for_filesystem(save_prompt,100)).strip()}.{extension}"
                video_path = os.path.join(save_path, file_name)
                any_mmaudio = MMAudio_setting != 0 and server_config.get("mmaudio_enabled", 0) != 0 and sample.shape[1] >=fps

                if is_image:    
                    image_path = os.path.join(image_save_path, file_name)
                    sample =  sample.transpose(1,0)  #c f h w -> f c h w 
                    new_image_path = []
                    for no, img in enumerate(sample):  
                        img_path = os.path.splitext(image_path)[0] + ("" if no==0 else f"_{no}") + ".jpg" 
                        new_image_path.append(save_image(img, save_file = img_path, quality = server_config.get("image_output_codec", None)))

                    video_path= new_image_path
                elif len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0 or output_new_audio_filepath is not None or any_mmaudio or output_new_audio_data is not None or audio_source is not None:
                    video_path = os.path.join(save_path, file_name)
                    save_path_tmp = video_path[:-4] + "_tmp.mp4"
                    save_video( tensor=sample[None], save_file=save_path_tmp, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1), codec_type = server_config.get("video_output_codec", None))
                    output_new_audio_temp_filepath = None
                    new_audio_from_start =  reset_control_aligment
                    source_audio_duration = source_video_frames_count / fps
                    if any_mmaudio:
                        send_cmd("progress", [0, get_latest_status(state,"MMAudio Soundtrack Generation")])
                        from postprocessing.mmaudio.mmaudio import video_to_audio
                        output_new_audio_filepath = output_new_audio_temp_filepath = get_available_filename(save_path, f"tmp{time_flag}.wav" )
                        video_to_audio(save_path_tmp, prompt = MMAudio_prompt, negative_prompt = MMAudio_neg_prompt, seed = seed, num_steps = 25, cfg_strength = 4.5, duration= sample.shape[1] /fps, save_path = output_new_audio_filepath, persistent_models = server_config.get("mmaudio_enabled", 0) == 2, audio_file_only = True, verboseLevel = verbose_level)
                        new_audio_from_start =  False
                    elif audio_source is not None:
                        output_new_audio_filepath = audio_source
                        new_audio_from_start =  True
                    elif output_new_audio_data is not None:
                        import soundfile as sf
                        output_new_audio_filepath = output_new_audio_temp_filepath = get_available_filename(save_path, f"tmp{time_flag}.wav" )
                        sf.write(output_new_audio_filepath, output_new_audio_data, audio_sampling_rate)                       
                    if output_new_audio_filepath is not None:
                        new_audio_tracks = [output_new_audio_filepath]
                    else:
                        new_audio_tracks = control_audio_tracks

                    combine_and_concatenate_video_with_audio_tracks(video_path, save_path_tmp,  source_audio_tracks, new_audio_tracks, source_audio_duration, audio_sampling_rate, new_audio_from_start = new_audio_from_start, source_audio_metadata= source_audio_metadata, verbose = verbose_level>=2 )
                    os.remove(save_path_tmp)
                    if output_new_audio_temp_filepath is not None: os.remove(output_new_audio_temp_filepath)

                else:
                    save_video( tensor=sample[None], save_file=video_path, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1),  codec_type= server_config.get("video_output_codec", None))

                end_time = time.time()

                inputs = get_function_arguments(generate_video, locals())
                inputs.pop("send_cmd")
                inputs.pop("task")
                inputs.pop("mode")
                inputs["model_type"] = model_type
                inputs["model_filename"] = original_filename
                if is_image:
                    inputs["image_quality"] = server_config.get("image_output_codec", None)
                else:
                    inputs["video_quality"] = server_config.get("video_output_codec", None)

                modules = get_model_recursive_prop(model_type, "modules", return_list= True)
                if len(modules) > 0 : inputs["modules"] = modules
                if len(transformer_loras_filenames) > 0:
                    inputs.update({
                    "transformer_loras_filenames" : transformer_loras_filenames,
                    "transformer_loras_multipliers" : transformer_loras_multipliers
                    })                
                configs = prepare_inputs_dict("metadata", inputs, model_type)
                if sliding_window: configs["window_no"] = window_no
                configs["prompt"] = "\n".join(original_prompts)
                if prompt_enhancer_image_caption_model != None and prompt_enhancer !=None and len(prompt_enhancer)>0:
                    configs["enhanced_prompt"] = "\n".join(prompts)
                configs["generation_time"] = round(end_time-start_time)
                # if is_image: configs["is_image"] = True
                metadata_choice = server_config.get("metadata_type","metadata")
                video_path = [video_path] if not isinstance(video_path, list) else video_path
                for no, path in enumerate(video_path): 
                    if metadata_choice == "json":
                        with open(path.replace(f'.{extension}', '.json'), 'w') as f:
                            json.dump(configs, f, indent=4)
                    elif metadata_choice == "metadata":
                        if is_image:
                            save_image_metadata(path, configs)
                        else:
                            from mutagen.mp4 import MP4
                            file = MP4(path)
                            file.tags['©cmt'] = [json.dumps(configs)]
                            file.save()
                    if is_image:
                        print(f"New image saved to Path: "+ path)
                    else:
                        print(f"New video saved to Path: "+ path)
                    with lock:
                        file_list.append(path)
                        file_settings_list.append(configs if no > 0 else configs.copy())
                    
                # Play notification sound for single video
                try:
                    if server_config.get("notification_sound_enabled", 0):
                        volume = server_config.get("notification_sound_volume", 50)
                        notification_sound.notify_video_completion(
                            video_path=video_path, 
                            volume=volume
                        )
                except Exception as e:
                    print(f"Error playing notification sound for individual video: {e}")

                send_cmd("output")

        seed = set_seed(-1)
    clear_status(state)
    trans.cache = None
    offload.unload_loras_from_model(trans)
    if not trans2 is None:
        trans2.cache = None
        offload.unload_loras_from_model(trans2)

    if len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0:
        cleanup_temp_audio_files(control_audio_tracks + source_audio_tracks)

    remove_temp_filenames(temp_filenames_list)

def prepare_generate_video(state):    

    if state.get("validate_success",0) != 1:
        return gr.Button(visible= True), gr.Button(visible= False), gr.Column(visible= False), gr.update(visible=False)
    else:
        return gr.Button(visible= False), gr.Button(visible= True), gr.Column(visible= True), gr.update(visible= False)


def generate_preview(model_type, latents):
    import einops
    if latents is None: return None
    model_handler = get_model_handler(model_type)
    base_model_type = get_base_model_type(model_type)
    if hasattr(model_handler, "get_rgb_factors"):
        latent_rgb_factors, latent_rgb_factors_bias = model_handler.get_rgb_factors(base_model_type )
    else:
        return None
    if latent_rgb_factors is None: return None
    latents = latents.unsqueeze(0) 
    nb_latents = latents.shape[2]
    latents_to_preview = 4
    latents_to_preview = min(nb_latents, latents_to_preview)
    skip_latent =  nb_latents / latents_to_preview
    latent_no = 0
    selected_latents = []
    while latent_no < nb_latents:
        selected_latents.append( latents[:, : , int(latent_no): int(latent_no)+1])
        latent_no += skip_latent 

    latents = torch.cat(selected_latents, dim = 2)
    weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
    bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
    images = images.add_(1.0).mul_(127.5)
    images = images.detach().cpu()
    if images.dtype == torch.bfloat16:
        images = images.to(torch.float16)
    images = images.numpy().clip(0, 255).astype(np.uint8)
    images = einops.rearrange(images, 'b c t h w -> (b h) (t w) c')
    h, w, _ = images.shape
    scale = 200 / h
    images= Image.fromarray(images)
    images = images.resize(( int(w*scale),int(h*scale)), resample=Image.Resampling.BILINEAR) 
    return images


def process_tasks(state):
    from shared.utils.thread_utils import AsyncStream, async_run

    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    progress = None

    if len(queue) == 0:
        gen["status_display"] =  False
        return
    with lock:
        gen = get_gen_info(state)
        clear_file_list = server_config.get("clear_file_list", 0)    
        file_list = gen.get("file_list", [])
        file_settings_list = gen.get("file_settings_list", [])
        if clear_file_list > 0:
            file_list_current_size = len(file_list)
            keep_file_from = max(file_list_current_size - clear_file_list, 0)
            files_removed = keep_file_from
            choice = gen.get("selected",0)
            choice = max(choice- files_removed, 0)
            file_list = file_list[ keep_file_from: ]
            file_settings_list = file_settings_list[ keep_file_from: ]
        else:
            file_list = []
            choice = 0
        gen["selected"] = choice         
        gen["file_list"] = file_list    
        gen["file_settings_list"] = file_settings_list    

    while True:
        with gen_lock:
            process_status = gen.get("process_status", None)
            if process_status is None or process_status == "process:main":
                gen["process_status"] = "process:main"
                break
        time.sleep(1)

    def release_gen():
        with gen_lock:
            process_status = gen.get("process_status", None)
            if process_status.startswith("request:"):        
                gen["process_status"] = "process:" + process_status[len("request:"):]
            else:
                gen["process_status"] = None

    start_time = time.time()

    global gen_in_progress
    gen_in_progress = True
    gen["in_progress"] = True
    gen["preview"] = None
    gen["status"] = "Generating Video"
    gen["header_text"] = ""    

    yield time.time(), time.time() 
    prompt_no = 0
    while len(queue) > 0:
        prompt_no += 1
        gen["prompt_no"] = prompt_no
        task = queue[0]
        task_id = task["id"] 
        params = task['params']

        com_stream = AsyncStream()
        send_cmd = com_stream.output_queue.push
        def generate_video_error_handler():
            try:
                import inspect
                model_type = params.get('model_type')
                known_defaults = {
                    'image_refs_relative_size': 50,
                }

                for arg_name, default_value in known_defaults.items():
                    if arg_name not in params:
                        print(f"Warning: Missing argument '{arg_name}' in loaded task. Applying default value: {default_value}")
                        params[arg_name] = default_value
                if model_type:
                    default_settings = get_default_settings(model_type)
                    expected_args = inspect.signature(generate_video).parameters.keys()
                    for arg_name in expected_args:
                        if arg_name not in params and arg_name in default_settings:
                            params[arg_name] = default_settings[arg_name]
                generate_video(task, send_cmd,  **params)
            except Exception as e:
                tb = traceback.format_exc().split('\n')[:-1] 
                print('\n'.join(tb))
                send_cmd("error",str(e))
            finally:
                send_cmd("exit", None)


        async_run(generate_video_error_handler)

        while True:
            cmd, data = com_stream.output_queue.next()               
            if cmd == "exit":
                break
            elif cmd == "info":
                gr.Info(data)
            elif cmd == "error": 
                queue.clear()
                gen["prompts_max"] = 0
                gen["prompt"] = ""
                gen["status_display"] =  False
                release_gen()
                raise gr.Error(data, print_exception= False, duration = 0)
            elif cmd == "status":
                gen["status"] = data
            elif cmd == "output":
                gen["preview"] = None
                yield time.time() , time.time() 
            elif cmd == "progress":
                gen["progress_args"] = data
                # progress(*data)
            elif cmd == "preview":
                torch.cuda.current_stream().synchronize()
                preview= None if data== None else generate_preview(params["model_type"], data) 
                gen["preview"] = preview
                yield time.time() , gr.Text()
            else:
                release_gen()
                raise Exception(f"unknown command {cmd}")

        abort = gen.get("abort", False)
        if abort:
            gen["abort"] = False
            status = "Video Generation Aborted", "Video Generation Aborted"
            # yield  gr.Text(), gr.Text()
            yield time.time() , time.time() 
            gen["status"] = status

        queue[:] = [item for item in queue if item['id'] != task['id']]
        update_global_queue_ref(queue)

    gen["prompts_max"] = 0
    gen["prompt"] = ""
    end_time = time.time()
    if abort:
        # status = f"Video generation was aborted. Total Generation Time: {end_time-start_time:.1f}s" 
        status = f"Video generation was aborted. Total Generation Time: {format_time(end_time-start_time)}" 
    else:
        # status = f"Total Generation Time: {end_time-start_time:.1f}s" 
        status = f"Total Generation Time: {format_time(end_time-start_time)}"         
        # Play notification sound when video generation completed successfully
        try:
            if server_config.get("notification_sound_enabled", 1):
                volume = server_config.get("notification_sound_volume", 50)
                notification_sound.notify_video_completion(volume=volume)
        except Exception as e:
            print(f"Error playing notification sound: {e}")
    gen["status"] = status
    gen["status_display"] =  False
    release_gen()



def get_generation_status(prompt_no, prompts_max, repeat_no, repeat_max, window_no, total_windows):
    if prompts_max == 1:        
        if repeat_max <= 1:
            status = ""
        else:
            status = f"Sample {repeat_no}/{repeat_max}"
    else:
        if repeat_max <= 1:
            status = f"Prompt {prompt_no}/{prompts_max}"
        else:
            status = f"Prompt {prompt_no}/{prompts_max}, Sample {repeat_no}/{repeat_max}"
    if total_windows > 1:
        if len(status) > 0:
            status += ", "
        status += f"Sliding Window {window_no}/{total_windows}"

    return status

refresh_id = 0

def get_new_refresh_id():
    global refresh_id
    refresh_id += 1
    return refresh_id

def merge_status_context(status="", context=""):
    if len(status) == 0:
        return context
    elif len(context) == 0:
        return status
    else:
        # Check if context already contains the time
        if "|" in context:
            parts = context.split("|")
            return f"{status} - {parts[0].strip()} | {parts[1].strip()}"
        else:
            return f"{status} - {context}"
        
def clear_status(state):
    gen = get_gen_info(state)
    gen["extra_windows"] = 0
    gen["total_windows"] = 1
    gen["window_no"] = 1
    gen["extra_orders"] = 0
    gen["repeat_no"] = 0
    gen["total_generation"] = 0

def get_latest_status(state, context=""):
    gen = get_gen_info(state)
    prompt_no = gen["prompt_no"] 
    prompts_max = gen.get("prompts_max",0)
    total_generation = gen.get("total_generation", 1)
    repeat_no = gen.get("repeat_no",0)
    total_generation += gen.get("extra_orders", 0)
    total_windows = gen.get("total_windows", 0)
    total_windows += gen.get("extra_windows", 0)
    window_no = gen.get("window_no", 0)
    status = get_generation_status(prompt_no, prompts_max, repeat_no, total_generation, window_no, total_windows)
    return merge_status_context(status, context)

def update_status(state): 
    gen = get_gen_info(state)
    gen["progress_status"] = get_latest_status(state)
    gen["refresh"] = get_new_refresh_id()


def one_more_sample(state):
    gen = get_gen_info(state)
    extra_orders = gen.get("extra_orders", 0)
    extra_orders += 1
    gen["extra_orders"]  = extra_orders
    in_progress = gen.get("in_progress", False)
    if not in_progress :
        return state
    total_generation = gen.get("total_generation", 0) + extra_orders
    gen["progress_status"] = get_latest_status(state)
    gen["refresh"] = get_new_refresh_id()
    gr.Info(f"An extra sample generation is planned for a total of {total_generation} samples for this prompt")

    return state 

def one_more_window(state):
    gen = get_gen_info(state)
    extra_windows = gen.get("extra_windows", 0)
    extra_windows += 1
    gen["extra_windows"]= extra_windows
    in_progress = gen.get("in_progress", False)
    if not in_progress :
        return state
    total_windows = gen.get("total_windows", 0) + extra_windows
    gen["progress_status"] = get_latest_status(state)
    gen["refresh"] = get_new_refresh_id()
    gr.Info(f"An extra window generation is planned for a total of {total_windows} videos for this sample")

    return state 

def get_new_preset_msg(advanced = True):
    if advanced:
        return "Enter here a Name for a Lora Preset or a Settings or Choose one"
    else:
        return "Choose a Lora Preset or a Settings file in this List"
    
def compute_lset_choices(loras_presets):
    # lset_choices = [ (preset, preset) for preset in loras_presets]
    lset_list = []
    settings_list = []
    for item in loras_presets:
        if item.endswith(".lset"):
            lset_list.append(item)
        else:
            settings_list.append(item)

    sep = '\u2500' 
    indent = chr(160) * 4
    lset_choices = []
    if len(settings_list) > 0:
        settings_list.sort()
        lset_choices += [( (sep*16) +"Settings" + (sep*17), ">settings")]
        lset_choices += [ ( indent   + os.path.splitext(preset)[0], preset) for preset in settings_list ]
    if len(lset_list) > 0:
        lset_list.sort()
        lset_choices += [( (sep*18) + "Lsets" + (sep*18), ">lset")]
        lset_choices += [ ( indent   + os.path.splitext(preset)[0], preset) for preset in lset_list ]
    return lset_choices

def get_lset_name(state, lset_name):
    presets = state["loras_presets"]
    if len(lset_name) == 0 or lset_name.startswith(">") or lset_name== get_new_preset_msg(True) or lset_name== get_new_preset_msg(False): return ""
    if lset_name in presets: return lset_name
    choices = compute_lset_choices(presets)
    for label, value in choices:
        if label == lset_name: return value
    return lset_name

def validate_delete_lset(state, lset_name):
    lset_name = get_lset_name(state, lset_name)
    if len(lset_name) == 0:
        gr.Info(f"Choose a Preset to delete")
        return  gr.Button(visible= True), gr.Checkbox(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False) 
    else:
        return  gr.Button(visible= False), gr.Checkbox(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= True), gr.Button(visible= True) 
    
def validate_save_lset(state, lset_name):
    lset_name = get_lset_name(state, lset_name)
    if len(lset_name) == 0:
        gr.Info("Please enter a name for the preset")
        return  gr.Button(visible= True), gr.Checkbox(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False),gr.Checkbox(visible= False) 
    else:
        return  gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= True), gr.Button(visible= True),gr.Checkbox(visible= True)

def cancel_lset():
    return gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Checkbox(visible= False)


def save_lset(state, lset_name, loras_choices, loras_mult_choices, prompt, save_lset_prompt_cbox):    
    if lset_name.endswith(".json") or lset_name.endswith(".lset"):
        lset_name = os.path.splitext(lset_name)[0]

    loras_presets = state["loras_presets"] 
    loras = state["loras"]
    if state.get("validate_success",0) == 0:
        pass
    lset_name = get_lset_name(state, lset_name)
    if len(lset_name) == 0:
        gr.Info("Please enter a name for the preset / settings file")
        lset_choices =[("Please enter a name for a Lora Preset / Settings file","")]
    else:
        lset_name = sanitize_file_name(lset_name)
        lset_name = lset_name.replace('\u2500',"").strip()


        if save_lset_prompt_cbox ==2:
            lset = collect_current_model_settings(state)
            extension = ".json" 
        else:
            loras_choices_files = [ Path(loras[int(choice_no)]).parts[-1] for choice_no in loras_choices  ]
            lset  = {"loras" : loras_choices_files, "loras_mult" : loras_mult_choices}
            if save_lset_prompt_cbox!=1:
                prompts = prompt.replace("\r", "").split("\n")
                prompts = [prompt for prompt in prompts if len(prompt)> 0 and prompt.startswith("#")]
                prompt = "\n".join(prompts)
            if len(prompt) > 0:
                lset["prompt"] = prompt
            lset["full_prompt"] = save_lset_prompt_cbox ==1
            extension = ".lset" 
        
        if lset_name.endswith(".json") or lset_name.endswith(".lset"): lset_name = os.path.splitext(lset_name)[0]
        old_lset_name = lset_name + ".json"
        if not old_lset_name in loras_presets:
            old_lset_name = lset_name + ".lset"
            if not old_lset_name in loras_presets: old_lset_name = ""
        lset_name = lset_name + extension

        lora_dir = get_lora_dir(state["model_type"])
        full_lset_name_filename = os.path.join(lora_dir, lset_name ) 

        with open(full_lset_name_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(lset, indent=4))

        if len(old_lset_name) > 0 :
            if save_lset_prompt_cbox ==2:
                gr.Info(f"Settings File '{lset_name}' has been updated")
            else:
                gr.Info(f"Lora Preset '{lset_name}' has been updated")
            if old_lset_name != lset_name:
                pos = loras_presets.index(old_lset_name)
                loras_presets[pos] = lset_name 
                shutil.move( os.path.join(lora_dir, old_lset_name),  get_available_filename(lora_dir, old_lset_name + ".bkp" ) ) 
        else:
            if save_lset_prompt_cbox ==2:
                gr.Info(f"Settings File '{lset_name}' has been created")
            else:
                gr.Info(f"Lora Preset '{lset_name}' has been created")
            loras_presets.append(lset_name)
        state["loras_presets"] = loras_presets

        lset_choices = compute_lset_choices(loras_presets)
        lset_choices.append( (get_new_preset_msg(), ""))
    return gr.Dropdown(choices=lset_choices, value= lset_name), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False), gr.Checkbox(visible= False)

def delete_lset(state, lset_name):
    loras_presets = state["loras_presets"]
    lset_name = get_lset_name(state, lset_name)
    if len(lset_name) > 0:
        lset_name_filename = os.path.join( get_lora_dir(state["model_type"]),  sanitize_file_name(lset_name))
        if not os.path.isfile(lset_name_filename):
            gr.Info(f"Preset '{lset_name}' not found ")
            return [gr.update()]*7 
        os.remove(lset_name_filename)
        lset_choices = compute_lset_choices(loras_presets)
        pos = next( (i for i, item in enumerate(lset_choices) if item[1]==lset_name ), -1)
        gr.Info(f"Lora Preset '{lset_name}' has been deleted")
        loras_presets.remove(lset_name)
    else:
        pos = -1
        gr.Info(f"Choose a Preset / Settings File to delete")

    state["loras_presets"] = loras_presets

    lset_choices = compute_lset_choices(loras_presets)
    lset_choices.append((get_new_preset_msg(), ""))
    selected_lset_name = "" if pos < 0 else lset_choices[min(pos, len(lset_choices)-1)][1] 
    return  gr.Dropdown(choices=lset_choices, value= selected_lset_name), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Checkbox(visible= False)

def refresh_lora_list(state, lset_name, loras_choices):
    loras_names = state["loras_names"]
    prev_lora_names_selected = [ loras_names[int(i)] for i in loras_choices]
    model_type= state["model_type"]
    loras, loras_names, loras_presets, _, _, _, _  = setup_loras(model_type, None,  get_lora_dir(model_type), lora_preselected_preset, None)
    state["loras"] = loras
    state["loras_names"] = loras_names
    state["loras_presets"] = loras_presets

    gc.collect()
    new_loras_choices = [ (loras_name, str(i)) for i,loras_name in enumerate(loras_names)]
    new_loras_dict = { loras_name: str(i) for i,loras_name in enumerate(loras_names) }
    lora_names_selected = []
    for lora in prev_lora_names_selected:
        lora_id = new_loras_dict.get(lora, None)
        if lora_id!= None:
            lora_names_selected.append(lora_id)

    lset_choices = compute_lset_choices(loras_presets)
    lset_choices.append((get_new_preset_msg( state["advanced"]), "")) 
    if not lset_name in loras_presets:
        lset_name = ""
    
    if wan_model != None:
        errors = getattr(get_transformer_model(wan_model), "_loras_errors", "")
        if errors !=None and len(errors) > 0:
            error_files = [path for path, _ in errors]
            gr.Info("Error while refreshing Lora List, invalid Lora files: " + ", ".join(error_files))
        else:
            gr.Info("Lora List has been refreshed")


    return gr.Dropdown(choices=lset_choices, value= lset_name), gr.Dropdown(choices=new_loras_choices, value= lora_names_selected) 

def update_lset_type(state, lset_name):
    return 1 if lset_name.endswith(".lset") else 2


def apply_lset(state, wizard_prompt_activated, lset_name, loras_choices, loras_mult_choices, prompt):

    state["apply_success"] = 0

    lset_name = get_lset_name(state, lset_name)
    if len(lset_name) == 0:
        gr.Info("Please choose a Lora Preset or Setting File in the list or create one")
        return wizard_prompt_activated, loras_choices, loras_mult_choices, prompt, gr.update(), gr.update(), gr.update(), gr.update()
    else:
        current_model_type = state["model_type"]
        if lset_name.endswith(".lset"):
            loras = state["loras"]
            loras_choices, loras_mult_choices, preset_prompt, full_prompt, error = extract_preset(current_model_type,  lset_name, loras)
            if len(error) > 0:
                gr.Info(error)
            else:
                if full_prompt:
                    prompt = preset_prompt
                elif len(preset_prompt) > 0:
                    prompts = prompt.replace("\r", "").split("\n")
                    prompts = [prompt for prompt in prompts if len(prompt)>0 and not prompt.startswith("#")]
                    prompt = "\n".join(prompts) 
                    prompt = preset_prompt + '\n' + prompt
                gr.Info(f"Lora Preset '{lset_name}' has been applied")
                state["apply_success"] = 1
                wizard_prompt_activated = "on"

            return wizard_prompt_activated, loras_choices, loras_mult_choices, prompt, get_unique_id(), gr.update(), gr.update(), gr.update()
        else:
            configs, _ = get_settings_from_file(state, os.path.join(get_lora_dir(current_model_type), lset_name), True, True, True)
            if configs == None:
                gr.Info("File not supported")
                return [gr.update()] * 7
            
            model_type = configs["model_type"]
            configs["lset_name"] = lset_name
            gr.Info(f"Settings File '{lset_name}' has been applied")

            if model_type == current_model_type:
                set_model_settings(state, current_model_type, configs)        
                return *[gr.update()] * 4, gr.update(), gr.update(), gr.update(), get_unique_id()
            else:
                set_model_settings(state, model_type, configs)        
                return *[gr.update()] * 4, gr.update(), *generate_dropdown_model_list(model_type), gr.update()

def extract_prompt_from_wizard(state, variables_names, prompt, wizard_prompt, allow_null_values, *args):

    prompts = wizard_prompt.replace("\r" ,"").split("\n")

    new_prompts = [] 
    macro_already_written = False
    for prompt in prompts:
        if not macro_already_written and not prompt.startswith("#") and "{"  in prompt and "}"  in prompt:
            variables =  variables_names.split("\n")   
            values = args[:len(variables)]
            macro = "! "
            for i, (variable, value) in enumerate(zip(variables, values)):
                if len(value) == 0 and not allow_null_values:
                    return prompt, "You need to provide a value for '" + variable + "'" 
                sub_values= [ "\"" + sub_value + "\"" for sub_value in value.split("\n") ]
                value = ",".join(sub_values)
                if i>0:
                    macro += " : "    
                macro += "{" + variable + "}"+ f"={value}"
            if len(variables) > 0:
                macro_already_written = True
                new_prompts.append(macro)
            new_prompts.append(prompt)
        else:
            new_prompts.append(prompt)

    prompt = "\n".join(new_prompts)
    return prompt, ""

def validate_wizard_prompt(state, wizard_prompt_activated, wizard_variables_names, prompt, wizard_prompt, *args):
    state["validate_success"] = 0

    if wizard_prompt_activated != "on":
        state["validate_success"] = 1
        return prompt

    prompt, errors = extract_prompt_from_wizard(state, wizard_variables_names, prompt, wizard_prompt, False, *args)
    if len(errors) > 0:
        gr.Info(errors)
        return prompt

    state["validate_success"] = 1

    return prompt

def fill_prompt_from_wizard(state, wizard_prompt_activated, wizard_variables_names, prompt, wizard_prompt, *args):

    if wizard_prompt_activated == "on":
        prompt, errors = extract_prompt_from_wizard(state, wizard_variables_names, prompt,  wizard_prompt, True, *args)
        if len(errors) > 0:
            gr.Info(errors)

        wizard_prompt_activated = "off"

    return wizard_prompt_activated, "", gr.Textbox(visible= True, value =prompt) , gr.Textbox(visible= False), gr.Column(visible = True), *[gr.Column(visible = False)] * 2,  *[gr.Textbox(visible= False)] * PROMPT_VARS_MAX

def extract_wizard_prompt(prompt):
    variables = []
    values = {}
    prompts = prompt.replace("\r" ,"").split("\n")
    if sum(prompt.startswith("!") for prompt in prompts) > 1:
        return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"

    new_prompts = [] 
    errors = ""
    for prompt in prompts:
        if prompt.startswith("!"):
            variables, errors = prompt_parser.extract_variable_names(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
            if len(variables) > PROMPT_VARS_MAX:
                return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"
            values, errors = prompt_parser.extract_variable_values(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
        else:
            variables_extra, errors = prompt_parser.extract_variable_names(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
            variables += variables_extra
            variables = [var for pos, var in enumerate(variables) if var not in variables[:pos]]
            if len(variables) > PROMPT_VARS_MAX:
                return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"

            new_prompts.append(prompt)
    wizard_prompt = "\n".join(new_prompts)
    return  wizard_prompt, variables, values, errors

def fill_wizard_prompt(state, wizard_prompt_activated, prompt, wizard_prompt):
    def get_hidden_textboxes(num = PROMPT_VARS_MAX ):
        return [gr.Textbox(value="", visible=False)] * num

    hidden_column =  gr.Column(visible = False)
    visible_column =  gr.Column(visible = True)

    wizard_prompt_activated  = "off"  
    if state["advanced"] or state.get("apply_success") != 1:
        return wizard_prompt_activated, gr.Text(), prompt, wizard_prompt, gr.Column(), gr.Column(), hidden_column,  *get_hidden_textboxes() 
    prompt_parts= []

    wizard_prompt, variables, values, errors =  extract_wizard_prompt(prompt)
    if len(errors) > 0:
        gr.Info( errors )
        return wizard_prompt_activated, "", gr.Textbox(prompt, visible=True), gr.Textbox(wizard_prompt, visible=False), visible_column, *[hidden_column] * 2, *get_hidden_textboxes()

    for variable in variables:
        value = values.get(variable, "")
        prompt_parts.append(gr.Textbox( placeholder=variable, info= variable, visible= True, value= "\n".join(value) ))
    any_macro = len(variables) > 0

    prompt_parts += get_hidden_textboxes(PROMPT_VARS_MAX-len(prompt_parts))

    variables_names= "\n".join(variables)
    wizard_prompt_activated  = "on"

    return wizard_prompt_activated, variables_names,  gr.Textbox(prompt, visible = False),  gr.Textbox(wizard_prompt, visible = True),   hidden_column, visible_column, visible_column if any_macro else hidden_column, *prompt_parts

def switch_prompt_type(state, wizard_prompt_activated_var, wizard_variables_names, prompt, wizard_prompt, *prompt_vars):
    if state["advanced"]:
        return fill_prompt_from_wizard(state, wizard_prompt_activated_var, wizard_variables_names, prompt, wizard_prompt, *prompt_vars)
    else:
        state["apply_success"] = 1
        return fill_wizard_prompt(state, wizard_prompt_activated_var, prompt, wizard_prompt)

visible= False
def switch_advanced(state, new_advanced, lset_name):
    state["advanced"] = new_advanced
    loras_presets = state["loras_presets"]
    lset_choices = compute_lset_choices(loras_presets)
    lset_choices.append((get_new_preset_msg(new_advanced), ""))
    server_config["last_advanced_choice"] = new_advanced
    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config, indent=4))

    if lset_name== get_new_preset_msg(True) or lset_name== get_new_preset_msg(False) or lset_name=="":
        lset_name =  get_new_preset_msg(new_advanced)

    if only_allow_edit_in_advanced:
        return  gr.Row(visible=new_advanced), gr.Row(visible=new_advanced), gr.Button(visible=new_advanced), gr.Row(visible= not new_advanced), gr.Dropdown(choices=lset_choices, value= lset_name)
    else:
        return  gr.Row(visible=new_advanced), gr.Row(visible=True), gr.Button(visible=True), gr.Row(visible= False), gr.Dropdown(choices=lset_choices, value= lset_name)


def prepare_inputs_dict(target, inputs, model_type = None, model_filename = None ):
    
    state = inputs.pop("state")
    loras = state["loras"]
    if "loras_choices" in inputs:
        loras_choices = inputs.pop("loras_choices")
        inputs.pop("model_filename", None)
        activated_loras = [Path( loras[int(no)]).parts[-1]  for no in loras_choices ]
        inputs["activated_loras"] = activated_loras

    if target == "state":
        return inputs
    
    if "lset_name" in inputs:
        inputs.pop("lset_name")
        
    unsaved_params = ["image_start", "image_end", "image_refs", "video_guide", "image_guide", "video_source", "video_mask", "image_mask", "audio_guide", "audio_guide2", "audio_source"]
    for k in unsaved_params:
        inputs.pop(k)
    if model_type == None: model_type = state["model_type"]
    inputs["type"] = get_model_record(get_model_name(model_type))  
    inputs["settings_version"] = settings_version
    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)
    model_family = get_model_family(base_model_type)
    if model_type != base_model_type:
        inputs["base_model_type"] = base_model_type
    diffusion_forcing = base_model_type in ["sky_df_1.3B", "sky_df_14B"]
    vace =  test_vace_module(base_model_type) 
    t2v=   base_model_type in ["t2v"]
    ltxv = base_model_type in ["ltxv_13B"]
    recammaster = base_model_type in ["recam_1.3B"]
    phantom = base_model_type in ["phantom_1.3B", "phantom_14B"]
    flux = model_family in ["flux"]
    hunyuan_video_custom =  base_model_type in ["hunyuan_custom", "hunyuan_custom_audio", "hunyuan_custom_edit"]
    if target == "settings":
        return inputs

    image_outputs = inputs.get("image_mode",0) > 0

    pop=[]    
    if "force_fps" in inputs and len(inputs["force_fps"])== 0:
        pop += ["force_fps"]

    if model_def.get("sample_solvers", None) is None:
        pop += ["sample_solver"]
    
    # if not (test_class_i2v(base_model_type) or diffusion_forcing or ltxv or recammaster or vace):
    #     pop += ["image_prompt_type"]

    if any_audio_track(base_model_type) or server_config.get("mmaudio_enabled", 0) == 0:
        pop += ["MMAudio_setting", "MMAudio_prompt", "MMAudio_neg_prompt"]

    video_prompt_type = inputs["video_prompt_type"]
    if not "G" in video_prompt_type:
        pop += ["denoising_strength"]

    if not (server_config.get("enhancer_enabled", 0) > 0 and server_config.get("enhancer_mode", 0) == 0):
        pop += ["prompt_enhancer"]

    if model_def.get("model_modes", None) is None:
        pop += ["model_mode"]

    if not vace and not phantom and not hunyuan_video_custom:
        unsaved_params = ["keep_frames_video_guide", "mask_expand"] #"video_prompt_type",  
        if base_model_type in ["t2v"]: unsaved_params = unsaved_params[1:]
        pop += unsaved_params

    if not "I" in video_prompt_type:
        pop += ["remove_background_images_ref"]
        if not model_def.get("any_image_refs_relative_size", False):
            pop += ["image_refs_relative_size"]

    if not vace:
        pop += ["frames_positions", "control_net_weight", "control_net_weight2"] 
                
    if model_def.get("video_guide_outpainting", None) is None:
        pop += ["video_guide_outpainting"] 

    if not (vace or t2v):
        pop += ["min_frames_if_references"]

    if not (diffusion_forcing or ltxv or vace):
        pop += ["keep_frames_video_source"]

    if not test_any_sliding_window( base_model_type):
        pop += ["sliding_window_size", "sliding_window_overlap", "sliding_window_overlap_noise", "sliding_window_discard_last_frames", "sliding_window_color_correction_strength"]

    if not (base_model_type in ["fantasy"] or model_def.get("multitalk_class", False)):
        pop += ["audio_guidance_scale", "speakers_locations"]

    if not model_def.get("embedded_guidance", False):
        pop += ["embedded_guidance_scale"]

    if not (model_def.get("tea_cache", False) or model_def.get("mag_cache", False)) :
        pop += ["skip_steps_cache_type", "skip_steps_multiplier", "skip_steps_start_step_perc"]

    guidance_max_phases = model_def.get("guidance_max_phases", 0)
    guidance_phases = inputs.get("guidance_phases", 1)
    if guidance_max_phases < 1:
        pop += ["guidance_scale", "guidance_phases"]

    if guidance_max_phases < 2 or guidance_phases < 2:
        pop += ["guidance2_scale", "switch_threshold"]

    if guidance_max_phases < 3 or guidance_phases < 3:
        pop += ["guidance3_scale", "switch_threshold2", "model_switch_phase"]

    if ltxv or image_outputs:
        pop += ["flow_shift"]

    if model_def.get("no_negative_prompt", False) :
        pop += ["negative_prompt" ] 

    if not model_def.get("skip_layer_guidance", False):
        pop += ["slg_switch", "slg_layers", "slg_start_perc", "slg_end_perc"]

    if not model_def.get("cfg_zero", False):
        pop += [ "cfg_zero_step"  ] 

    if not model_def.get("cfg_star", False):
        pop += ["cfg_star_switch" ] 

    if not model_def.get("adaptive_projected_guidance", False):
        pop += ["apg_switch"] 

    if not model_family == "wan" or diffusion_forcing:
        pop +=["NAG_scale", "NAG_tau", "NAG_alpha" ]

    for k in pop:
        if k in inputs: inputs.pop(k)

    if target == "metadata":
        inputs = {k: v for k,v in inputs.items() if v != None  }

    return inputs

def get_function_arguments(func, locals):
    args_names = list(inspect.signature(func).parameters)
    kwargs = typing.OrderedDict()
    for k in args_names:
        kwargs[k] = locals[k]
    return kwargs


def init_generate(state, input_file_list, last_choice):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, input_file_list)

    set_file_choice(gen, file_list, last_choice)
    return get_unique_id(), ""

def video_to_control_video(state, input_file_list, choice):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update()
    gr.Info("Selected Video was copied to Control Video input")
    return file_list[choice]

def video_to_source_video(state, input_file_list, choice):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update()
    gr.Info("Selected Video was copied to Source Video input")    
    return file_list[choice]

def image_to_ref_image_add(state, input_file_list, choice, target, target_name):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update()
    model_type = state["model_type"]
    model_def = get_model_def(model_type)    
    if model_def.get("one_image_ref_needed", False):
        gr.Info(f"Selected Image was set to {target_name}")
        target =[file_list[choice]]
    else:
        gr.Info(f"Selected Image was added to {target_name}")
        if target == None:
            target =[]
        target.append( file_list[choice])
    return target

def image_to_ref_image_set(state, input_file_list, choice, target, target_name):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update()
    gr.Info(f"Selected Image was copied to {target_name}")
    return file_list[choice]

def image_to_ref_image_guide(state, input_file_list, choice):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update(), gr.update()
    ui_settings = get_current_model_settings(state)
    gr.Info(f"Selected Image was copied to Control Image")
    new_image = file_list[choice]
    if ui_settings["image_mode"]==2 or True:
        return new_image, new_image
    else:
        return new_image, None



def apply_post_processing(state, input_file_list, choice, PP_temporal_upsampling, PP_spatial_upsampling, PP_film_grain_intensity, PP_film_grain_saturation):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list)  :
        return gr.update(), gr.update(), gr.update()
    
    if not file_list[choice].endswith(".mp4"):
        gr.Info("Post processing is only available with Videos")
        return gr.update(), gr.update(), gr.update()
    overrides = {
        "temporal_upsampling":PP_temporal_upsampling,
        "spatial_upsampling":PP_spatial_upsampling,
        "film_grain_intensity": PP_film_grain_intensity, 
        "film_grain_saturation": PP_film_grain_saturation,
    }

    gen["edit_video_source"] = file_list[choice]
    gen["edit_overrides"] = overrides

    in_progress = gen.get("in_progress", False)
    return "edit_postprocessing", get_unique_id() if not in_progress else gr.update(), get_unique_id() if in_progress else gr.update()


def remux_audio(state, input_file_list, choice, PP_MMAudio_setting, PP_MMAudio_prompt, PP_MMAudio_neg_prompt, PP_MMAudio_seed, PP_repeat_generation, PP_custom_audio):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list)  :
        return gr.update(), gr.update(), gr.update()
    
    if not file_list[choice].endswith(".mp4"):
        gr.Info("Post processing is only available with Videos")
        return gr.update(), gr.update(), gr.update()
    overrides = {
        "MMAudio_setting" : PP_MMAudio_setting, 
        "MMAudio_prompt" : PP_MMAudio_prompt,
        "MMAudio_neg_prompt": PP_MMAudio_neg_prompt,
        "seed": PP_MMAudio_seed,
        "repeat_generation": PP_repeat_generation,
        "audio_source": PP_custom_audio,
    }

    gen["edit_video_source"] = file_list[choice]
    gen["edit_overrides"] = overrides

    in_progress = gen.get("in_progress", False)
    return "edit_remux", get_unique_id() if not in_progress else gr.update(), get_unique_id() if in_progress else gr.update()


def eject_video_from_gallery(state, input_file_list, choice):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, input_file_list)
    with lock:
        if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list)  :
            return gr.update(), gr.update(), gr.update()
        
        extend_list = file_list[choice + 1:] # inplace List change
        file_list[:] = file_list[:choice]
        file_list.extend(extend_list)

        extend_list = file_settings_list[choice + 1:]
        file_settings_list[:] = file_settings_list[:choice]
        file_settings_list.extend(extend_list)
        choice = min(choice, len(file_list))
    return gr.Gallery(value = file_list, selected_index= choice), gr.update() if len(file_list) >0 else get_default_video_info(), gr.Row(visible= len(file_list) > 0)

def add_videos_to_gallery(state, input_file_list, choice, files_to_load):
    gen = get_gen_info(state)
    if files_to_load == None:
        return gr.update(),gr.update(), gr.update()
    file_list, file_settings_list = get_file_list(state, input_file_list)
    with lock:
        valid_files_count = 0
        invalid_files_count = 0
        for file_path in files_to_load:
            file_settings, _ = get_settings_from_file(state, file_path, False, False, False)
            if file_settings == None:
                fps = 0
                try:
                    if has_video_file_extension(file_path):
                        fps, width, height, frames_count = get_video_info(file_path)
                    elif has_image_file_extension(file_path):
                        width, height = Image.open(file_path).size
                        fps = 1 
                except:
                    pass
                if fps == 0:
                    invalid_files_count += 1 
                    continue
            file_list.append(file_path)
            file_settings_list.append(file_settings)
            valid_files_count +=1

    if valid_files_count== 0 and invalid_files_count ==0:
        gr.Info("No Video to Add")
    else:
        txt = ""
        if valid_files_count > 0:
            txt = f"{valid_files_count} files were added. " if valid_files_count > 1 else  f"One file was added."
        if invalid_files_count > 0:
            txt += f"Unable to add {invalid_files_count} files which were invalid. " if invalid_files_count > 1 else  f"Unable to add one file which was invalid."
        gr.Info(txt)
    if choice != None and choice <= 0:
        choice = len(file_list)
        gen["selected"] = choice
    return gr.Gallery(value = file_list, selected_index=choice, preview= True), gr.Files(value=[]),  gr.Tabs(selected="video_info")

def get_model_settings(state, model_type):
    all_settings = state.get("all_settings", None)    
    return None if all_settings == None else all_settings.get(model_type, None)

def set_model_settings(state, model_type, settings):
    all_settings = state.get("all_settings", None)    
    if all_settings == None:
        all_settings = {}
        state["all_settings"] = all_settings
    all_settings[model_type] = settings
    
def collect_current_model_settings(state):
    model_filename = state["model_filename"]
    model_type = state["model_type"]
    settings = get_model_settings(state, model_type)
    settings["state"] = state
    settings = prepare_inputs_dict("metadata", settings)
    settings["model_filename"] = model_filename 
    settings["model_type"] = model_type 
    return settings 

def export_settings(state):
    model_type = state["model_type"]
    text = json.dumps(collect_current_model_settings(state), indent=4)
    text_base64 = base64.b64encode(text.encode('utf8')).decode('utf-8')
    return text_base64, sanitize_file_name(model_type + "_" + datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss") + ".json")


def use_video_settings(state, input_file_list, choice):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if choice != None and choice >=0 and len(file_list)>0:
        configs = file_settings_list[choice]
        file_name= file_list[choice]
        if configs == None:
            gr.Info("No Settings to Extract")
        else:
            current_model_type = state["model_type"]
            model_type = configs["model_type"] 
            models_compatible = are_model_types_compatible(model_type,current_model_type) 
            if models_compatible:
                model_type = current_model_type
            defaults = get_model_settings(state, model_type) 
            defaults = get_default_settings(model_type) if defaults == None else defaults
            defaults.update(configs)
            prompt = configs.get("prompt", "")
            set_model_settings(state, model_type, defaults)
            if has_image_file_extension(file_name):
                gr.Info(f"Settings Loaded from Image with prompt '{prompt[:100]}'")
            else:
                gr.Info(f"Settings Loaded from Video with prompt '{prompt[:100]}'")
            if models_compatible:
                return gr.update(), gr.update(), str(time.time())
            else:
                return *generate_dropdown_model_list(model_type), gr.update()
    else:
        gr.Info(f"No Video is Selected")

    return gr.update(), gr.update()

def get_settings_from_file(state, file_path, allow_json, merge_with_defaults, switch_type_if_compatible):    
    configs = None
    any_image_or_video = False
    if file_path.endswith(".json") and allow_json:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
        except:
            pass
    elif file_path.endswith(".mp4"):
        from mutagen.mp4 import MP4
        try:
            file = MP4(file_path)
            tags = file.tags['©cmt'][0] 
            configs = json.loads(tags)
            any_image_or_video = True
        except:
            pass
    elif has_image_file_extension(file_path):
        try:
            configs = read_image_metadata(file_path)
            any_image_or_video = True
        except:
            pass
    if configs is None: return None, False
    try:
        if not "WanGP" in configs.get("type", ""): configs = None 
    except:
        configs = None
    if configs is None: return None, False
        

    current_model_filename = state["model_filename"]
    current_model_type = state["model_type"]
    
    model_type = configs.get("model_type", None)
    if get_base_model_type(model_type) == None:
        model_type = configs.get("base_model_type", None)
  
    if model_type == None:
        model_filename = configs.get("model_filename", current_model_filename)
        model_type = get_model_type(model_filename)
        if model_type == None:
            model_type = current_model_type
    elif not model_type in model_types:
        model_type = current_model_type
    fix_settings(model_type, configs)
    if switch_type_if_compatible and are_model_types_compatible(model_type,current_model_type):
        model_type = current_model_type
    if merge_with_defaults:
        defaults = get_model_settings(state, model_type) 
        defaults = get_default_settings(model_type) if defaults == None else defaults
        defaults.update(configs)
        configs = defaults
    configs["model_type"] = model_type

    return configs, any_image_or_video

def record_image_mode_tab(state, evt:gr.SelectData):
    state["image_mode_tab"] = evt.index

def switch_image_mode(state):
    image_mode = state.get("image_mode_tab", 0)
    model_type =state["model_type"]
    ui_defaults = get_model_settings(state, model_type)        

    ui_defaults["image_mode"] = image_mode
    video_prompt_type = ui_defaults.get("video_prompt_type", "") 
    model_def = get_model_def( model_type)
    inpaint_support = model_def.get("inpaint_support", False)
    if inpaint_support:
        if image_mode == 1:
            video_prompt_type = del_in_sequence(video_prompt_type, "VAG" + all_guide_processes)  
            video_prompt_type = add_to_sequence(video_prompt_type, "KI")
        elif image_mode == 2:
            video_prompt_type = del_in_sequence(video_prompt_type, "KI" + all_guide_processes)
            video_prompt_type = add_to_sequence(video_prompt_type, "VAG")  
        ui_defaults["video_prompt_type"] = video_prompt_type 
        
    return  str(time.time())

def load_settings_from_file(state, file_path):
    gen = get_gen_info(state)

    if file_path==None:
        return gr.update(), gr.update(), None

    configs, any_video_or_image_file = get_settings_from_file(state, file_path, True, True, True)
    if configs == None:
        gr.Info("File not supported")
        return gr.update(), gr.update(), None

    current_model_type = state["model_type"]
    model_type = configs["model_type"]
    prompt = configs.get("prompt", "")
    is_image = configs.get("is_image", False)

    if any_video_or_image_file:    
        gr.Info(f"Settings Loaded from {'Image' if is_image else 'Video'} generated with prompt '{prompt[:100]}'")
    else:
        gr.Info(f"Settings Loaded from Settings file with prompt '{prompt[:100]}'")

    if model_type == current_model_type:
        set_model_settings(state, current_model_type, configs)        
        return gr.update(), gr.update(), str(time.time()), None
    else:
        set_model_settings(state, model_type, configs)        
        return *generate_dropdown_model_list(model_type), gr.update(), None

def save_inputs(
            target,
            image_mask_guide,
            lset_name,
            image_mode,
            prompt,
            negative_prompt,
            resolution,
            video_length,
            batch_size,
            seed,
            force_fps,
            num_inference_steps,
            guidance_scale,
            guidance2_scale,
            guidance3_scale,
            switch_threshold,
            switch_threshold2,
            guidance_phases,
            model_switch_phase,
            audio_guidance_scale,
            flow_shift,
            sample_solver,
            embedded_guidance_scale,
            repeat_generation,
            multi_prompts_gen_type,
            multi_images_gen_type,
            skip_steps_cache_type,
            skip_steps_multiplier,
            skip_steps_start_step_perc,    
            loras_choices,
            loras_multipliers,
            image_prompt_type,
            image_start,
            image_end,
            model_mode,
            video_source,
            keep_frames_video_source,
            video_guide_outpainting,
            video_prompt_type,
            image_refs,
            frames_positions,
            video_guide,
            image_guide,
            keep_frames_video_guide,
            denoising_strength,
            video_mask,
            image_mask,
            control_net_weight,
            control_net_weight2,
            mask_expand,
            audio_guide,
            audio_guide2,
            audio_source,            
            audio_prompt_type,
            speakers_locations,
            sliding_window_size,
            sliding_window_overlap,
            sliding_window_color_correction_strength,
            sliding_window_overlap_noise,
            sliding_window_discard_last_frames,
            image_refs_relative_size,
            remove_background_images_ref,
            temporal_upsampling,
            spatial_upsampling,
            film_grain_intensity,
            film_grain_saturation,
            MMAudio_setting,
            MMAudio_prompt,
            MMAudio_neg_prompt,            
            RIFLEx_setting,
            NAG_scale,
            NAG_tau,
            NAG_alpha,
            slg_switch, 
            slg_layers,
            slg_start_perc,
            slg_end_perc,
            apg_switch,
            cfg_star_switch,
            cfg_zero_step,
            prompt_enhancer,
            min_frames_if_references,
            override_profile,
            mode,
            state,
):


    model_filename = state["model_filename"]
    model_type = state["model_type"]
    if image_mask_guide is not None and image_mode >= 1 and video_prompt_type is not None and "A" in video_prompt_type and not "U" in video_prompt_type:
    # if image_mask_guide is not None and image_mode == 2:
        if "background" in image_mask_guide: 
            image_guide = image_mask_guide["background"]
        if "layers" in image_mask_guide and len(image_mask_guide["layers"])>0: 
            image_mask = image_mask_guide["layers"][0] 
        image_mask_guide = None
    inputs = get_function_arguments(save_inputs, locals())
    inputs.pop("target")
    inputs.pop("image_mask_guide")
    cleaned_inputs = prepare_inputs_dict(target, inputs)
    if target == "settings":
        defaults_filename = get_settings_file_name(model_type)

        with open(defaults_filename, "w", encoding="utf-8") as f:
            json.dump(cleaned_inputs, f, indent=4)

        gr.Info("New Default Settings saved")
    elif target == "state":
        set_model_settings(state, model_type, cleaned_inputs)        

def download_loras():
    from huggingface_hub import  snapshot_download    
    yield gr.Row(visible=True), "<B><FONT SIZE=3>Please wait while the Loras are being downloaded</B></FONT>" #, *[gr.Column(visible=False)] * 2
    lora_dir = get_lora_dir("i2v")
    log_path = os.path.join(lora_dir, "log.txt")
    if not os.path.isfile(log_path):
        tmp_path = os.path.join(lora_dir, "tmp_lora_dowload")
        import glob
        snapshot_download(repo_id="DeepBeepMeep/Wan2.1",  allow_patterns="loras_i2v/*", local_dir= tmp_path)
        for f in glob.glob(os.path.join(tmp_path, "loras_i2v", "*.*")):
            target_file = os.path.join(lora_dir,  Path(f).parts[-1] )
            if os.path.isfile(target_file):
                os.remove(f)
            else:
                shutil.move(f, lora_dir) 
    try:
        os.remove(tmp_path)
    except:
        pass
    yield gr.Row(visible=True), "<B><FONT SIZE=3>Loras have been completely downloaded</B></FONT>" #, *[gr.Column(visible=True)] * 2

    from datetime import datetime
    dt = datetime.today().strftime('%Y-%m-%d')
    with open( log_path, "w", encoding="utf-8") as writer:
        writer.write(f"Loras downloaded on the {dt} at {time.time()} on the {time.time()}")
    return


def handle_queue_action(state, action_string):
    if not action_string:
        return gr.HTML(), gr.Tabs()
        
    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    
    try:
        action, row_index_str = action_string.split('_')
        row_index = int(row_index_str)
    except (IndexError, ValueError):
        return gr.HTML(), gr.Tabs()

    if action == "edit":
        state["editing_task_index"] = row_index
        task_to_edit_index = row_index + 1
        
        if task_to_edit_index < len(queue):
            task_data = queue[task_to_edit_index]
            gr.Info(f"Loading task '{task_data['prompt'][:50]}...' for editing.")
            return update_queue_data(queue), gr.Tabs(selected="edit")
        else:
            gr.Warning("Task index out of bounds.")
            return update_queue_data(queue), gr.Tabs()
            
    elif action == "up":
        return move_up(queue, [row_index]), gr.Tabs()
    elif action == "down":
        return move_down(queue, [row_index]), gr.Tabs()
    elif action == "remove":
        new_queue_data = remove_task(queue, [row_index])
        gen["prompts_max"] = gen.get("prompts_max", 0) - 1
        update_status(state)
        return new_queue_data, gr.Tabs()

    return update_queue_data(queue), gr.Tabs()

def change_model(state, model_choice):
    if model_choice == None:
        return
    model_filename = get_model_filename(model_choice, transformer_quantization, transformer_dtype_policy)
    state["model_filename"] = model_filename
    last_model_per_family = state["last_model_per_family"] 
    last_model_per_family[get_model_family(model_choice, for_ui= True)] = model_choice
    server_config["last_model_per_family"] = last_model_per_family
    server_config["last_model_type"] = model_choice

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config, indent=4))

    state["model_type"] = model_choice
    header = generate_header(model_choice, compile=compile, attention_mode=attention_mode)
    
    return header

def get_current_model_settings(state):
    model_type = state["model_type"]
    ui_defaults = get_model_settings(state, model_type)
    if ui_defaults == None:
        ui_defaults = get_default_settings(model_type)
        set_model_settings(state, model_type, ui_defaults)
    return ui_defaults 

def fill_inputs(state):
    ui_defaults = get_current_model_settings(state)
 
    return generate_video_tab(update_form = True, state_dict = state, ui_defaults = ui_defaults)

def preload_model_when_switching(state):
    global reload_needed, wan_model, offloadobj
    if "S" in preload_model_policy:
        model_type = state["model_type"] 
        if  model_type !=  transformer_type:
            wan_model = None
            release_model()            
            model_filename = get_model_name(model_type)
            yield f"Loading model {model_filename}..."
            wan_model, offloadobj = load_models(model_type)
            yield f"Model loaded"
            reload_needed=  False 
        return   
    return gr.Text()

def unload_model_if_needed(state):
    global wan_model
    if "U" in preload_model_policy:
        if wan_model != None:
            wan_model = None
            release_model()

def all_letters(source_str, letters):
    for letter in letters:
        if not letter in source_str:
            return False
    return True    

def any_letters(source_str, letters):
    for letter in letters:
        if letter in source_str:
            return True
    return False

def filter_letters(source_str, letters, default= ""):
    ret = ""
    for letter in letters:
        if letter in source_str:
            ret += letter
    if len(ret) == 0:
        return default
    return ret    

def add_to_sequence(source_str, letters):
    ret = source_str
    for letter in letters:
        if not letter in source_str:
            ret += letter
    return ret    

def del_in_sequence(source_str, letters):
    ret = source_str
    for letter in letters:
        if letter in source_str:
            ret = ret.replace(letter, "")
    return ret    

def refresh_audio_prompt_type_remux(state, audio_prompt_type, remux):
    audio_prompt_type = del_in_sequence(audio_prompt_type, "R")
    audio_prompt_type = add_to_sequence(audio_prompt_type, remux)
    return audio_prompt_type

def refresh_remove_background_sound(state, audio_prompt_type, remove_background_sound):
    audio_prompt_type = del_in_sequence(audio_prompt_type, "V")
    if remove_background_sound:
        audio_prompt_type = add_to_sequence(audio_prompt_type, "V")
    return audio_prompt_type


def refresh_audio_prompt_type_sources(state, audio_prompt_type, audio_prompt_type_sources):
    audio_prompt_type = del_in_sequence(audio_prompt_type, "XCPAB")
    audio_prompt_type = add_to_sequence(audio_prompt_type, audio_prompt_type_sources)
    return audio_prompt_type, gr.update(visible = "A" in audio_prompt_type), gr.update(visible = "B" in audio_prompt_type), gr.update(visible = ("B" in audio_prompt_type or "X" in audio_prompt_type)), gr.update(visible= any_letters(audio_prompt_type, "ABX"))

def refresh_image_prompt_type_radio(state, image_prompt_type, image_prompt_type_radio):
    image_prompt_type = del_in_sequence(image_prompt_type, "VLTS")
    image_prompt_type = add_to_sequence(image_prompt_type, image_prompt_type_radio)
    any_video_source = len(filter_letters(image_prompt_type, "VL"))>0
    model_def = get_model_def(state["model_type"])
    image_prompt_types_allowed = model_def.get("image_prompt_types_allowed", "")
    end_visible = "E" in image_prompt_types_allowed and any_letters(image_prompt_type, "SVL")
    return image_prompt_type, gr.update(visible = "S" in image_prompt_type ), gr.update(visible = end_visible and ("E" in image_prompt_type) ), gr.update(visible = "V" in image_prompt_type) , gr.update(visible = any_video_source), gr.update(visible = end_visible)

def refresh_image_prompt_type_endcheckbox(state, image_prompt_type, image_prompt_type_radio, end_checkbox):
    image_prompt_type = del_in_sequence(image_prompt_type, "E")
    if end_checkbox: image_prompt_type += "E"
    image_prompt_type = add_to_sequence(image_prompt_type, image_prompt_type_radio)
    return image_prompt_type, gr.update(visible = "E" in image_prompt_type )

def refresh_video_prompt_type_image_refs(state, video_prompt_type, video_prompt_type_image_refs, image_mode):
    model_type = state["model_type"]
    model_def = get_model_def(model_type)
    image_ref_choices = model_def.get("image_ref_choices", None)
    if image_ref_choices is not None:
        video_prompt_type = del_in_sequence(video_prompt_type, image_ref_choices["letters_filter"])
    else:
        video_prompt_type = del_in_sequence(video_prompt_type, "KFI")
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_image_refs)
    visible = "I" in video_prompt_type
    any_outpainting= image_mode in model_def.get("video_guide_outpainting", [])
    rm_bg_visible= visible and not model_def.get("no_background_removal", False) 
    img_rel_size_visible = visible and model_def.get("any_image_refs_relative_size", False)
    return video_prompt_type, gr.update(visible = visible),gr.update(visible = rm_bg_visible), gr.update(visible = img_rel_size_visible), gr.update(visible = visible and "F" in video_prompt_type_image_refs), gr.update(visible= ("F" in video_prompt_type_image_refs or "K" in video_prompt_type_image_refs or "V" in video_prompt_type) and any_outpainting )

def switch_image_guide_editor(image_mode, old_video_prompt_type , video_prompt_type, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value ):
    if image_mode == 0: return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    mask_in_old = "A" in old_video_prompt_type and not "U" in old_video_prompt_type
    mask_in_new = "A" in video_prompt_type and not "U" in video_prompt_type
    image_mask_guide_value, image_mask_value, image_guide_value = {}, {}, {}
    visible = "V" in video_prompt_type
    if mask_in_old != mask_in_new:
        if mask_in_new:
            if old_image_mask_value is None:
                image_mask_guide_value["value"] = old_image_guide_value
            else:
                image_mask_guide_value["value"] = {"background" : old_image_guide_value, "composite" : None, "layers": [rgb_bw_to_rgba_mask(old_image_mask_value)]}
            image_guide_value["value"] = image_mask_value["value"] = None 
        else:
            if old_image_mask_guide_value is not None and "background" in old_image_mask_guide_value:
                image_guide_value["value"] = old_image_mask_guide_value["background"]
                if "layers" in old_image_mask_guide_value:
                    image_mask_value["value"] = old_image_mask_guide_value["layers"][0] if len(old_image_mask_guide_value["layers"]) >=1 else None
            image_mask_guide_value["value"] = {"background" : None, "composite" : None, "layers": []}
            
    image_mask_guide = gr.update(visible= visible and mask_in_new, **image_mask_guide_value)
    image_guide = gr.update(visible = visible and not mask_in_new, **image_guide_value)
    image_mask = gr.update(visible = False, **image_mask_value)
    return image_mask_guide, image_guide, image_mask

def refresh_video_prompt_type_video_mask(state, video_prompt_type, video_prompt_type_video_mask, image_mode, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value ):
    old_video_prompt_type = video_prompt_type
    video_prompt_type = del_in_sequence(video_prompt_type, "XYZWNA")
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_video_mask)
    visible= "A" in video_prompt_type     
    model_type = state["model_type"]
    model_def = get_model_def(model_type)
    image_outputs =  image_mode > 0
    image_mask_guide, image_guide, image_mask = switch_image_guide_editor(image_mode, old_video_prompt_type , video_prompt_type, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value )
    return video_prompt_type, gr.update(visible= visible and not image_outputs), image_mask_guide, image_guide, image_mask, gr.update(visible= visible )

def refresh_video_prompt_type_alignment(state, video_prompt_type, video_prompt_type_video_guide):
    video_prompt_type = del_in_sequence(video_prompt_type, "T")
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_video_guide)
    return video_prompt_type

all_guide_processes ="PDESLCMUVB"

def refresh_video_prompt_type_video_guide(state, video_prompt_type, video_prompt_type_video_guide,  image_mode, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value ):
    old_video_prompt_type = video_prompt_type
    video_prompt_type = del_in_sequence(video_prompt_type, all_guide_processes)
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_video_guide)
    visible = "V" in video_prompt_type
    model_type = state["model_type"]
    model_def = get_model_def(model_type)
    any_outpainting= image_mode in model_def.get("video_guide_outpainting", [])
    mask_visible = visible and "A" in video_prompt_type and not "U" in video_prompt_type
    image_outputs =  image_mode > 0
    keep_frames_video_guide_visible = not image_outputs and visible and not model_def.get("keep_frames_video_guide_not_supported", False)
    image_mask_guide, image_guide, image_mask = switch_image_guide_editor(image_mode, old_video_prompt_type , video_prompt_type, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value )
    mask_preprocessing = model_def.get("mask_preprocessing", None)
    if mask_preprocessing  is not None:
        mask_selector_visible = mask_preprocessing.get("visible", True)
    else:
        mask_selector_visible = True
    return video_prompt_type,  gr.update(visible = visible and not image_outputs), image_guide, gr.update(visible = keep_frames_video_guide_visible), gr.update(visible = visible and "G" in video_prompt_type), gr.update(visible= (visible or "F" in video_prompt_type or "K" in video_prompt_type) and any_outpainting), gr.update(visible= visible and mask_selector_visible and  not "U" in video_prompt_type ) ,  gr.update(visible= mask_visible and not image_outputs), image_mask, image_mask_guide, gr.update(visible= mask_visible) 


def refresh_video_prompt_type_video_guide_alt(state, video_prompt_type, video_prompt_type_video_guide_alt, image_mode):
    model_def = get_model_def(state["model_type"])
    guide_custom_choices = model_def.get("guide_custom_choices",{})
    video_prompt_type = del_in_sequence(video_prompt_type, guide_custom_choices.get("letters_filter",""))
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_video_guide_alt)
    control_video_visible = "V" in video_prompt_type
    ref_images_visible = "I" in video_prompt_type
    denoising_strength_visible = "G" in video_prompt_type
    return video_prompt_type,  gr.update(visible = control_video_visible and image_mode ==0), gr.update(visible = control_video_visible and image_mode >=1), gr.update(visible = ref_images_visible ), gr.update(visible = denoising_strength_visible )
 
# def refresh_video_prompt_video_guide_trigger(state, video_prompt_type, video_prompt_type_video_guide):
#     video_prompt_type_video_guide = video_prompt_type_video_guide.split("#")[0]
#     return refresh_video_prompt_type_video_guide(state, video_prompt_type, video_prompt_type_video_guide)

def refresh_preview(state):
    gen = get_gen_info(state)
    preview = gen.get("preview", None)
    return preview

def init_process_queue_if_any(state):                
    gen = get_gen_info(state)
    if bool(gen.get("queue",[])):
        state["validate_success"] = 1
        return gr.Button(visible=False), gr.Button(visible=True), gr.Column(visible=True)                   
    else:
        return gr.Button(visible=True), gr.Button(visible=False), gr.Column(visible=False)

def get_modal_image(image_base64, label):
    return "<DIV ALIGN=CENTER><IMG SRC=\"" + image_base64 + "\"><div style='position: absolute; top: 0; left: 0; background: rgba(0,0,0,0.7); color: white; padding: 5px; font-size: 12px;'>" + label + "</div></DIV>"

def show_modal_image(state, action_string):
    if not action_string:
        return gr.HTML(), gr.Column(visible=False)

    try:
        img_type, row_index_str = action_string.split('_')
        row_index = int(row_index_str)
    except (ValueError, IndexError):
        return gr.HTML(), gr.Column(visible=False)

    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    task_index = row_index + 1

    if task_index >= len(queue):
        return gr.HTML(), gr.Column(visible=False)

    task_item = queue[task_index]
    image_data = None
    label_data = None

    if img_type == 'start':
        image_data = task_item.get('start_image_data_base64')
        label_data = task_item.get('start_image_labels')
    elif img_type == 'end':
        image_data = task_item.get('end_image_data_base64')
        label_data = task_item.get('end_image_labels')

    if not image_data or not label_data:
        return gr.HTML(), gr.Column(visible=False)

    html_content = get_modal_image(image_data[0], label_data[0])
    return gr.HTML(value=html_content), gr.Column(visible=True)

def get_prompt_labels(multi_prompts_gen_type, image_outputs = False):
    new_line_text = "each new line of prompt will be used for a window" if multi_prompts_gen_type != 0 else "each new line of prompt will generate " + ("a new image" if image_outputs else "a new video")
    return "Prompts (" + new_line_text + ", # lines = comments, ! lines = macros)", "Prompts (" + new_line_text + ", # lines = comments)"

def get_image_end_label(multi_prompts_gen_type):
    return "Images as ending points for new Videos in the Generation Queue" if multi_prompts_gen_type == 0 else "Images as ending points for each new Window of the same Video Generation" 

def refresh_prompt_labels(multi_prompts_gen_type, image_mode):
    prompt_label, wizard_prompt_label =  get_prompt_labels(multi_prompts_gen_type, image_mode > 0)
    return gr.update(label=prompt_label), gr.update(label = wizard_prompt_label), gr.update(label=get_image_end_label(multi_prompts_gen_type))

def update_video_guide_outpainting(video_guide_outpainting_value, value, pos):
    if len(video_guide_outpainting_value) <= 1:
        video_guide_outpainting_list = ["0"] * 4
    else:
        video_guide_outpainting_list = video_guide_outpainting_value.split(" ")
    video_guide_outpainting_list[pos] = str(value)
    if all(v=="0" for v in video_guide_outpainting_list):
        return ""
    return " ".join(video_guide_outpainting_list)

def refresh_video_guide_outpainting_row(video_guide_outpainting_checkbox, video_guide_outpainting):
    video_guide_outpainting = video_guide_outpainting[1:] if video_guide_outpainting_checkbox else "#" + video_guide_outpainting 
        
    return gr.update(visible=video_guide_outpainting_checkbox), video_guide_outpainting

custom_resolutions = None
def get_resolution_choices(current_resolution_choice, model_resolutions= None):
    global custom_resolutions

    resolution_file = "resolutions.json"
    if model_resolutions is not None:
        resolution_choices = model_resolutions
    elif custom_resolutions == None and os.path.isfile(resolution_file) :
        with open(resolution_file, 'r', encoding='utf-8') as f:
            try:
                resolution_choices = json.load(f)
            except Exception as e:
                print(f'Invalid "{resolution_file}" : {e}')
                resolution_choices = None
        if resolution_choices ==  None:
            pass 
        elif not isinstance(resolution_choices, list):
            print(f'"{resolution_file}" should be a list of 2 elements lists ["Label","WxH"]')
            resolution_choices == None
        else:
            for tup in resolution_choices:
                if not isinstance(tup, list) or len(tup) != 2 or not isinstance(tup[0], str) or not isinstance(tup[1], str):
                    print(f'"{resolution_file}" contains an invalid list of two elements: {tup}')
                    resolution_choices == None
                    break
                res_list = tup[1].split("x")
                if len(res_list) != 2 or not is_integer(res_list[0])  or not is_integer(res_list[1]):
                    print(f'"{resolution_file}" contains a resolution value that is not in the format "WxH": {tup[1]}')
                    resolution_choices == None
                    break
        custom_resolutions = resolution_choices
    else:
        resolution_choices = custom_resolutions
    if resolution_choices == None:
        resolution_choices=[
            # 1080p
            ("1920x1088 (16:9)", "1920x1088"),
            ("1088x1920 (9:16)", "1088x1920"),
            ("1920x832 (21:9)", "1920x832"),
            ("832x1920 (9:21)", "832x1920"),
            # 720p
            ("1024x1024 (1:1)", "1024x1024"),
            ("1280x720 (16:9)", "1280x720"),
            ("720x1280 (9:16)", "720x1280"), 
            ("1280x544 (21:9)", "1280x544"),
            ("544x1280 (9:21)", "544x1280"),
            ("1104x832 (4:3)", "1104x832"),
            ("832x1104 (3:4)", "832x1104"),
            ("960x960 (1:1)", "960x960"),
            # 540p
            ("960x544 (16:9)", "960x544"),
            ("544x960 (9:16)", "544x960"),
            # 480p
            ("832x624 (4:3)", "832x624"), 
            ("624x832 (3:4)", "624x832"),
            ("720x720 (1:1)", "720x720"),
            ("832x480 (16:9)", "832x480"),
            ("480x832 (9:16)", "480x832"),
            ("512x512 (1:1)", "512x512"),
        ]

    if current_resolution_choice is not None:
        found = False
        for label, res in resolution_choices:
            if current_resolution_choice == res:
                found = True
                break
        if not found:
            if model_resolutions is None:
                resolution_choices.append( (current_resolution_choice, current_resolution_choice ))
            else:
                current_resolution_choice = resolution_choices[0][1]

    return resolution_choices, current_resolution_choice

group_thresholds = {
    "360p": 320 * 640,    
    "480p": 832 * 624,     
    "540p": 960 * 544,   
    "720p": 1024 * 1024,  
    "1080p": 1920 * 1088,         
    "1440p": 9999 * 9999
}
    
def categorize_resolution(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    pixel_count = width * height
    
    for group in group_thresholds.keys():
        if pixel_count <= group_thresholds[group]:
            return group
    return "1440p"

def group_resolutions(model_def, resolutions, selected_resolution):

    model_resolutions = model_def.get("resolutions", None)
    if model_resolutions is not None:
        selected_group ="Locked"
        available_groups = [selected_group ]
        selected_group_resolutions = model_resolutions
    else:
        grouped_resolutions = {}
        for resolution in resolutions:
            group = categorize_resolution(resolution[1])
            if group not in grouped_resolutions:
                grouped_resolutions[group] = []
            grouped_resolutions[group].append(resolution)
        
        available_groups = [group for group in group_thresholds if group in grouped_resolutions]
    
        selected_group = categorize_resolution(selected_resolution)
        selected_group_resolutions = grouped_resolutions.get(selected_group, [])
        available_groups.reverse()
    return available_groups, selected_group_resolutions, selected_group

def change_resolution_group(state, selected_group):
    model_type = state["model_type"]
    model_def = get_model_def(model_type)
    model_resolutions = model_def.get("resolutions", None)
    resolution_choices, _ = get_resolution_choices(None, model_resolutions)   
    if model_resolutions is None:
        group_resolution_choices = [ resolution for resolution in resolution_choices if categorize_resolution(resolution[1]) == selected_group ]
    else:
        last_resolution = group_resolution_choices[0][1]
        return gr.update(choices= group_resolution_choices, value= last_resolution) 

    last_resolution_per_group = state["last_resolution_per_group"]
    last_resolution = last_resolution_per_group.get(selected_group, "")
    if len(last_resolution) == 0 or not any( [last_resolution == resolution[1] for resolution in group_resolution_choices]):
        last_resolution = group_resolution_choices[0][1]
    return gr.update(choices= group_resolution_choices, value= last_resolution ) 
    


def record_last_resolution(state, resolution):

    model_type = state["model_type"]
    model_def = get_model_def(model_type)
    model_resolutions = model_def.get("resolutions", None)
    if model_resolutions is not None: return
    server_config["last_resolution_choice"] = resolution
    selected_group = categorize_resolution(resolution)
    last_resolution_per_group = state["last_resolution_per_group"]
    last_resolution_per_group[selected_group ] = resolution
    server_config["last_resolution_per_group"] = last_resolution_per_group
    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config, indent=4))

def get_max_frames(nb):
    return (nb - 1) * server_config.get("max_frames_multiplier",1) + 1


def change_guidance_phases(state, guidance_phases):
    model_type = state["model_type"]
    model_def = get_model_def(model_type)
    multiple_submodels = model_def.get("multiple_submodels", False)
    label ="Phase 1-2" if guidance_phases ==3 else ( "Model / Guidance Switch Threshold" if multiple_submodels  else "Guidance Switch Threshold" )
    return gr.update(visible= guidance_phases >=3 and multiple_submodels) , gr.update(visible= guidance_phases >=2), gr.update(visible= guidance_phases >=2, label = label), gr.update(visible= guidance_phases >=3), gr.update(visible= guidance_phases >=2), gr.update(visible= guidance_phases >=3)


memory_profile_choices= [   ("Profile 1, HighRAM_HighVRAM: at least 64 GB of RAM and 24 GB of VRAM, the fastest for short videos with a RTX 3090 / RTX 4090", 1),
                            ("Profile 2, HighRAM_LowVRAM: at least 64 GB of RAM and 12 GB of VRAM, the most versatile profile with high RAM, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos", 2),
                            ("Profile 3, LowRAM_HighVRAM: at least 32 GB of RAM and 24 GB of VRAM, adapted for RTX 3090 / RTX 4090 with limited RAM for good speed short video",3),
                            ("Profile 4, LowRAM_LowVRAM (Recommended): at least 32 GB of RAM and 12 GB of VRAM, if you have little VRAM or want to generate longer videos",4),
                            ("Profile 5, VerylowRAM_LowVRAM (Fail safe): at least 24 GB of RAM and 10 GB of VRAM, if you don't have much it won't be fast but maybe it will work",5)]

def detect_auto_save_form(state, evt:gr.SelectData):
    last_tab_id = state.get("last_tab_id", 0)
    state["last_tab_id"] = new_tab_id = evt.index
    if new_tab_id > 0 and last_tab_id == 0:
        return get_unique_id()
    else:
        return gr.update()

def compute_video_length_label(fps, current_video_length):
    if fps is None:
        return f"Number of frames"
    else:
        return f"Number of frames ({fps} frames = 1s), current duration: {(current_video_length / fps):.1f}s",  

def refresh_video_length_label(state, current_video_length, force_fps, video_guide, video_source):
    base_model_type = get_base_model_type(state["model_type"])
    computed_fps = get_computed_fps(force_fps, base_model_type , video_guide, video_source )
    return gr.update(label= compute_video_length_label(computed_fps, current_video_length))

def get_default_value(choices, current_value, default_value = None):
    for label, value in choices:
        if value == current_value:
            return current_value
    return default_value

def generate_video_tab(update_form = False, state_dict = None, ui_defaults = None, model_family = None, model_choice = None, header = None, main = None, main_tabs= None, tab_id='generate'):
    global inputs_names #, advanced

    if update_form:
        model_filename = state_dict["model_filename"]
        model_type = state_dict["model_type"]
        advanced_ui = state_dict["advanced"]  
    else:
        model_type = transformer_type
        model_filename = get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)
        advanced_ui = advanced
        ui_defaults=  get_default_settings(model_type)
        state_dict = {}
        state_dict["model_filename"] = model_filename
        state_dict["model_type"] = model_type
        state_dict["advanced"] = advanced_ui
        state_dict["last_model_per_family"] = server_config.get("last_model_per_family", {})
        state_dict["last_resolution_per_group"] = server_config.get("last_resolution_per_group", {})
        gen = dict()
        gen["queue"] = []
        state_dict["gen"] = gen
    model_def = get_model_def(model_type)
    if model_def == None: model_def = {} 
    base_model_type = get_base_model_type(model_type)
    model_filename = get_model_filename( base_model_type )
    preset_to_load = lora_preselected_preset if lora_preset_model == model_type else "" 

    loras, loras_names, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset = setup_loras(model_type,  None,  get_lora_dir(model_type), preset_to_load, None)

    state_dict["loras"] = loras
    state_dict["loras_presets"] = loras_presets
    state_dict["loras_names"] = loras_names

    launch_prompt = ""
    launch_preset = ""
    launch_loras = []
    launch_multis_str = ""

    if update_form:
        pass
    if len(default_lora_preset) > 0 and lora_preset_model == model_type:
        launch_preset = default_lora_preset
        launch_prompt = default_lora_preset_prompt 
        launch_loras = default_loras_choices
        launch_multis_str = default_loras_multis_str

    if len(launch_preset) == 0:
        launch_preset = ui_defaults.get("lset_name","")
    if len(launch_prompt) == 0:
        launch_prompt = ui_defaults.get("prompt","")
    if len(launch_loras) == 0:
        launch_multis_str = ui_defaults.get("loras_multipliers","")
        activated_loras = ui_defaults.get("activated_loras",[])
        if len(activated_loras) > 0:
            lora_filenames = [os.path.basename(lora_path) for lora_path in loras]
            activated_indices = []
            for lora_file in ui_defaults["activated_loras"]:
                try:
                    idx = lora_filenames.index(lora_file)
                    activated_indices.append(str(idx))
                except ValueError: 
                    print(f"Warning: Lora file {lora_file} from config not found in loras directory")
            launch_loras = activated_indices

    with gr.Row():
        with gr.Column():
            with gr.Column(visible=False, elem_id="image-modal-container") as modal_container: 
                with gr.Row(elem_id="image-modal-close-button-row"):
                    close_modal_button = gr.Button("❌", size="sm", scale=1)
                modal_html_display = gr.HTML()
                modal_action_input = gr.Text(elem_id="modal_action_input", visible=False)
                modal_action_trigger = gr.Button(elem_id="modal_action_trigger", visible=False)
            with gr.Row(visible= True): #len(loras)>0) as presets_column:
                lset_choices = compute_lset_choices(loras_presets) + [(get_new_preset_msg(advanced_ui), "")]
                with gr.Column(scale=6):
                    lset_name = gr.Dropdown(show_label=False, allow_custom_value= True, scale=5, filterable=True, choices= lset_choices, value=launch_preset)
                with gr.Column(scale=1):
                    with gr.Row(height=17): 
                        apply_lset_btn = gr.Button("Apply", size="sm", min_width= 1)
                        refresh_lora_btn = gr.Button("Refresh", size="sm", min_width= 1, visible=advanced_ui or not only_allow_edit_in_advanced)
                        if len(launch_preset) == 0 : 
                            lset_type = 2   
                        else:
                            lset_type = 1 if launch_preset.endswith(".lset") else 2
                        save_lset_prompt_drop= gr.Dropdown(
                            choices=[
                                # ("Save Loras & Only Prompt Comments", 0),
                                ("Save Only Loras & Full Prompt", 1),
                                ("Save All the Settings", 2)
                            ],  show_label= False, container=False, value = lset_type, visible= False
                        ) 
                    with gr.Row(height=17, visible=False) as refresh2_row:
                        refresh_lora_btn2 = gr.Button("Refresh", size="sm", min_width= 1)

                    with gr.Row(height=17, visible=advanced_ui or not only_allow_edit_in_advanced) as preset_buttons_rows:
                        confirm_save_lset_btn = gr.Button("Go Ahead Save it !", size="sm", min_width= 1, visible=False) 
                        confirm_delete_lset_btn = gr.Button("Go Ahead Delete it !", size="sm", min_width= 1, visible=False) 
                        save_lset_btn = gr.Button("Save", size="sm", min_width= 1, visible = True)
                        delete_lset_btn = gr.Button("Delete", size="sm", min_width= 1, visible = True)
                        cancel_lset_btn = gr.Button("Don't do it !", size="sm", min_width= 1 , visible=False)  
                        #confirm_save_lset_btn, confirm_delete_lset_btn, save_lset_btn, delete_lset_btn, cancel_lset_btn
            if not update_form:
                state = gr.State(state_dict)     
            trigger_refresh_input_type = gr.Text(interactive= False, visible= False)
            t2v =  base_model_type in ["t2v"] 
            t2v_1_3B =  base_model_type in ["t2v_1.3B"] 
            flf2v = base_model_type == "flf2v_720p"
            base_model_family = get_model_family(base_model_type)
            diffusion_forcing = "diffusion_forcing" in model_filename 
            ltxv = "ltxv" in model_filename 
            lock_inference_steps = model_def.get("lock_inference_steps", False)
            any_tea_cache = model_def.get("tea_cache", False)
            any_mag_cache = model_def.get("mag_cache", False)
            recammaster = base_model_type in ["recam_1.3B"]
            vace = test_vace_module(base_model_type)
            phantom = base_model_type in ["phantom_1.3B", "phantom_14B"]
            fantasy = base_model_type in ["fantasy"]
            multitalk = model_def.get("multitalk_class", False)
            standin = model_def.get("standin_class", False)
            infinitetalk =  base_model_type in ["infinitetalk"]
            hunyuan_t2v = "hunyuan_video_720" in model_filename
            hunyuan_i2v = "hunyuan_video_i2v" in model_filename
            hunyuan_video_custom = "hunyuan_video_custom" in model_filename
            hunyuan_video_custom =  base_model_type in ["hunyuan_custom", "hunyuan_custom_audio", "hunyuan_custom_edit"]
            hunyuan_video_custom_audio = base_model_type in ["hunyuan_custom_audio"]
            hunyuan_video_custom_edit = base_model_type in ["hunyuan_custom_edit"]
            hunyuan_video_avatar = "hunyuan_video_avatar" in model_filename
            flux =  base_model_family in ["flux"]
            qwen =  base_model_family in ["qwen"]
            image_outputs = model_def.get("image_outputs", False)
            sliding_window_enabled = test_any_sliding_window(model_type)
            multi_prompts_gen_type_value = ui_defaults.get("multi_prompts_gen_type_value",0)
            prompt_label, wizard_prompt_label = get_prompt_labels(multi_prompts_gen_type_value, image_outputs)            
            any_video_source = False
            fps = get_model_fps(base_model_type)
            image_prompt_type_value = ""
            video_prompt_type_value = ""
            any_start_image = any_end_image = any_reference_image = any_image_mask = False
            v2i_switch_supported = (vace or t2v or standin) and not image_outputs
            ti2v_2_2 = base_model_type in ["ti2v_2_2"]
            gallery_height = 350
            def get_image_gallery(label ="", value = None, single_image_mode = False, visible = False ):
                with gr.Row(visible = visible) as gallery_row:
                    gallery_amg = AdvancedMediaGallery(media_mode="image", height=gallery_height, columns=4, label=label, initial = value , single_image_mode = single_image_mode )
                    gallery_amg.mount(update_form=update_form)
                return gallery_row, gallery_amg.gallery, [gallery_row] + gallery_amg.get_toggable_elements()

            image_mode_value = ui_defaults.get("image_mode", 1 if image_outputs else 0 )
            if not v2i_switch_supported and not image_outputs:
                image_mode_value = 0
            else:
                image_outputs = image_mode_value > 0 
            inpaint_support = model_def.get("inpaint_support", False)
            image_mode = gr.Number(value =image_mode_value, visible = False)
            image_mode_tab_selected= "t2i" if image_mode_value == 1 else ("inpaint" if image_mode_value == 2 else "t2v") 
            with gr.Tabs(visible = v2i_switch_supported or inpaint_support, selected= image_mode_tab_selected ) as image_mode_tabs:
                with gr.Tab("Text to Video", id = "t2v", elem_classes="compact_tab", visible = v2i_switch_supported) as tab_t2v:
                    pass
                with gr.Tab("Text to Image", id = "t2i", elem_classes="compact_tab"):
                    pass
                with gr.Tab("Image Inpainting", id = "inpaint", elem_classes="compact_tab", visible=inpaint_support) as tab_inpaint:
                    pass

            image_prompt_types_allowed = model_def.get("image_prompt_types_allowed", "")
            model_mode_choices = model_def.get("model_modes", None)
            model_modes_visibility = [0,1,2]
            if model_mode_choices is not None: model_modes_visibility= model_mode_choices.get("image_modes", model_modes_visibility)
            
            with gr.Column(visible= image_mode_value == 0 and len(image_prompt_types_allowed)> 0 or model_mode_choices is not None and image_mode_value in model_modes_visibility ) as image_prompt_column: 
                # Video Continue /  Start Frame / End Frame
                image_prompt_type_value= ui_defaults.get("image_prompt_type","")
                image_prompt_type = gr.Text(value= image_prompt_type_value, visible= False)
                image_prompt_type_choices = []
                if "T" in image_prompt_types_allowed: 
                    image_prompt_type_choices += [("Text Prompt Only" if "S" in image_prompt_types_allowed else "New Video", "")]
                if "S" in image_prompt_types_allowed: 
                    image_prompt_type_choices += [("Start Video with Image", "S")]
                    any_start_image = True
                if "V" in image_prompt_types_allowed:
                    any_video_source = True
                    image_prompt_type_choices += [("Continue Video", "V")]
                if "L" in image_prompt_types_allowed:
                    any_video_source = True
                    image_prompt_type_choices += [("Continue Last Video", "L")]
                with gr.Group(visible= len(image_prompt_types_allowed)>1 and image_mode_value == 0) as image_prompt_type_group:
                    with gr.Row():
                        image_prompt_type_radio_allowed_values= filter_letters(image_prompt_types_allowed, "SVL")
                        image_prompt_type_radio_value = filter_letters(image_prompt_type_value, image_prompt_type_radio_allowed_values,  image_prompt_type_choices[0][1] if len(image_prompt_type_choices) > 0 else "")
                        if len(image_prompt_type_choices) > 0:
                            image_prompt_type_radio = gr.Radio( image_prompt_type_choices, value = image_prompt_type_radio_value, label="Location", show_label= False, visible= len(image_prompt_types_allowed)>1, scale= 3)
                        else:
                            image_prompt_type_radio = gr.Radio(choices=[("", "")], value="", visible= False)
                        if "E" in image_prompt_types_allowed:
                            image_prompt_type_endcheckbox = gr.Checkbox( value ="E" in image_prompt_type_value, label="End Image(s)", show_label= False, visible= any_letters(image_prompt_type_radio_value, "SVL") and not image_outputs , scale= 1)
                            any_end_image = True
                        else:
                            image_prompt_type_endcheckbox = gr.Checkbox( value =False, show_label= False, visible= False , scale= 1)
                image_start_row, image_start, image_start_extra = get_image_gallery(label= "Images as starting points for new Videos in the Generation Queue", value = ui_defaults.get("image_start", None), visible= "S" in image_prompt_type_value )
                video_source = gr.Video(label= "Video to Continue", height = gallery_height, visible= "V" in image_prompt_type_value, value= ui_defaults.get("video_source", None),)
                image_end_row, image_end, image_end_extra = get_image_gallery(label= get_image_end_label(ui_defaults.get("multi_prompts_gen_type", 0)), value = ui_defaults.get("image_end", None), visible= any_letters(image_prompt_type_value, "SVL") and ("E" in image_prompt_type_value)     ) 
                if model_mode_choices is None or image_mode_value not in model_modes_visibility:
                    model_mode = gr.Dropdown(value=None, visible=False)
                else:
                    model_mode_value = get_default_value(model_mode_choices["choices"], ui_defaults.get("model_mode", None), model_mode_choices["default"] )
                    model_mode = gr.Dropdown(choices=model_mode_choices["choices"], value=model_mode_value, label=model_mode_choices["label"],  visible=True)                        
                keep_frames_video_source = gr.Text(value=ui_defaults.get("keep_frames_video_source","") , visible= len(filter_letters(image_prompt_type_value, "VL"))>0 , scale = 2, label= "Truncate Video beyond this number of resampled Frames (empty=Keep All, negative truncates from End)" ) 

            any_control_video = any_control_image = False
            if image_mode_value ==2:
                guide_preprocessing = { "selection": ["V", "VG"]}
                mask_preprocessing = { "selection": ["A"]}
            else:
                guide_preprocessing = model_def.get("guide_preprocessing", None)
                mask_preprocessing = model_def.get("mask_preprocessing", None)
            guide_custom_choices = model_def.get("guide_custom_choices", None)
            image_ref_choices = model_def.get("image_ref_choices", None)

            # with gr.Column(visible= vace or phantom or hunyuan_video_custom or hunyuan_video_avatar or hunyuan_video_custom_edit or t2v or standin or ltxv or infinitetalk or recammaster or (flux or qwen ) and model_reference_image and image_mode_value >=1) as video_prompt_column: 
            with gr.Column(visible= guide_preprocessing is not None or mask_preprocessing is not None or guide_custom_choices is not None or image_ref_choices is not None) as video_prompt_column: 
                video_prompt_type_value= ui_defaults.get("video_prompt_type","")
                video_prompt_type = gr.Text(value= video_prompt_type_value, visible= False)
                with gr.Row(visible = image_mode_value!=2) as guide_selection_row:
                    # Control Video Preprocessing
                    if guide_preprocessing is None:
                        video_prompt_type_video_guide = gr.Dropdown(choices=[("","")], value="", label="Control Video", scale = 2, visible= False, show_label= True, )
                    else:
                        pose_label = "Pose" if image_outputs else "Motion" 
                        guide_preprocessing_labels_all = {
                            "": "No Control Video",
                            "UV": "Keep Control Video Unchanged",
                            "PV": f"Transfer Human {pose_label}",
                            "DV": "Transfer Depth",
                            "EV": "Transfer Canny Edges",
                            "SV": "Transfer Shapes",
                            "LV": "Transfer Flow",
                            "CV": "Recolorize",
                            "MV": "Perform Inpainting",
                            "V": "Use Vace raw format",
                            "PDV": f"Transfer Human {pose_label} & Depth",
                            "PSV": f"Transfer Human {pose_label} & Shapes",
                            "PLV": f"Transfer Human {pose_label} & Flow" ,
                            "DSV": "Transfer Depth & Shapes",
                            "DLV": "Transfer Depth & Flow",
                            "SLV": "Transfer Shapes & Flow",
                        }
                        guide_preprocessing_choices = []
                        guide_preprocessing_labels = guide_preprocessing.get("labels", {}) 
                        for process_type in guide_preprocessing["selection"]:
                            process_label = guide_preprocessing_labels.get(process_type, None)
                            process_label = guide_preprocessing_labels_all.get(process_type,process_type) if process_label is None else process_label
                            if image_outputs: process_label = process_label.replace("Video", "Image")
                            guide_preprocessing_choices.append( (process_label, process_type) )

                        video_prompt_type_video_guide_label = guide_preprocessing.get("label", "Control Video Process")
                        if image_outputs: video_prompt_type_video_guide_label = video_prompt_type_video_guide_label.replace("Video", "Image")
                        video_prompt_type_video_guide = gr.Dropdown(
                            guide_preprocessing_choices,
                            value=filter_letters(video_prompt_type_value,  all_guide_processes, guide_preprocessing.get("default", "") ),
                            label= video_prompt_type_video_guide_label , scale = 2, visible= guide_preprocessing.get("visible", True) , show_label= True,
                        )
                        any_control_video = True
                        any_control_image = image_outputs 

                    # Alternate Control Video Preprocessing / Options
                    if guide_custom_choices is None:
                        video_prompt_type_video_guide_alt = gr.Dropdown(choices=[("","")], value="", label="Control Video", visible= False, scale = 2 )
                    else:
                        video_prompt_type_video_guide_alt_label = guide_custom_choices.get("label", "Control Video Process")
                        if image_outputs: video_prompt_type_video_guide_alt_label = video_prompt_type_video_guide_alt_label.replace("Video", "Image")
                        video_prompt_type_video_guide_alt_choices = [(label.replace("Video", "Image") if image_outputs else label, value) for label,value in guide_custom_choices["choices"] ]
                        guide_custom_choices_value = get_default_value(video_prompt_type_video_guide_alt_choices, filter_letters(video_prompt_type_value, guide_custom_choices["letters_filter"]), guide_custom_choices.get("default", "") )
                        video_prompt_type_video_guide_alt = gr.Dropdown(
                            choices= video_prompt_type_video_guide_alt_choices,
                            # value=filter_letters(video_prompt_type_value, guide_custom_choices["letters_filter"], guide_custom_choices.get("default", "") ),
                            value=guide_custom_choices_value,
                            visible = guide_custom_choices.get("visible", True),
                            label= video_prompt_type_video_guide_alt_label, show_label= guide_custom_choices.get("show_label", True), scale = 2
                        )
                        any_control_video = True
                        any_control_image = image_outputs 

                    # Control Mask Preprocessing
                    if mask_preprocessing is None:
                        video_prompt_type_video_mask = gr.Dropdown(choices=[("","")], value="", label="Video Mask", scale = 2, visible= False, show_label= True, )
                        any_image_mask = image_outputs
                    else:
                        mask_preprocessing_labels_all = {
                            "": "Whole Frame",
                            "A": "Masked Area",
                            "NA": "Non Masked Area",
                            "XA": "Masked Area, rest Inpainted",
                            "XNA": "Non Masked Area, rest Inpainted", 
                            "YA": "Masked Area, rest Depth",
                            "YNA": "Non Masked Area, rest Depth",
                            "WA": "Masked Area, rest Shapes",
                            "WNA": "Non Masked Area, rest Shapes",
                            "ZA": "Masked Area, rest Flow",
                            "ZNA": "Non Masked Area, rest Flow"
                        }

                        mask_preprocessing_choices = []
                        mask_preprocessing_labels = mask_preprocessing.get("labels", {}) 
                        for process_type in mask_preprocessing["selection"]:
                            process_label = mask_preprocessing_labels.get(process_type, None)
                            process_label = mask_preprocessing_labels_all.get(process_type, process_type) if process_label is None else process_label
                            mask_preprocessing_choices.append( (process_label, process_type) )

                        video_prompt_type_video_mask_label = mask_preprocessing.get("label", "Area Processed")
                        video_prompt_type_video_mask = gr.Dropdown(
                            mask_preprocessing_choices,
                            value=filter_letters(video_prompt_type_value, "XYZWNA", mask_preprocessing.get("default", "")),
                            label= video_prompt_type_video_mask_label , scale = 2, visible= "V" in video_prompt_type_value and not "U" in video_prompt_type_value and mask_preprocessing.get("visible", True), 
                            show_label= True,
                        )


                    # Image Refs Selection
                    if image_ref_choices is None:
                        video_prompt_type_image_refs = gr.Dropdown(
                            # choices=[ ("None", ""),("Start", "KI"),("Ref Image", "I")],
                            choices=[ ("None", ""),],
                            value=filter_letters(video_prompt_type_value, ""),
                            visible = False,
                            label="Start / Reference Images", scale = 2
                        )
                        any_reference_image = False
                    else:
                        any_reference_image = True
                        video_prompt_type_image_refs = gr.Dropdown(
                            choices= image_ref_choices["choices"],
                            value=filter_letters(video_prompt_type_value, image_ref_choices["letters_filter"]),
                            visible = image_ref_choices.get("visible", True),
                            label=image_ref_choices.get("label", "Inject Reference Images"), show_label= True, scale = 2
                        )

                image_guide = gr.Image(label= "Control Image", height = 800, type ="pil", visible= image_mode_value==1 and "V" in video_prompt_type_value and ("U" in video_prompt_type_value or not "A" in video_prompt_type_value ) , value= ui_defaults.get("image_guide", None))
                video_guide = gr.Video(label= "Control Video", height = gallery_height, visible= (not image_outputs) and "V" in video_prompt_type_value, value= ui_defaults.get("video_guide", None))
                if image_mode_value >= 1:  
                    image_guide_value = ui_defaults.get("image_guide", None)
                    image_mask_value = ui_defaults.get("image_mask", None)
                    if image_guide_value is None:
                        image_mask_guide_value = None
                    else:
                        image_mask_guide_value = { "background" : image_guide_value, "composite" : None}
                        image_mask_guide_value["layers"] = [] if image_mask_value is None else [rgb_bw_to_rgba_mask(image_mask_value)]

                    image_mask_guide = gr.ImageEditor(
                        label="Control Image to be Inpainted" if image_mode_value == 2 else "Control Image and Mask",
                        value = image_mask_guide_value, 
                        type='pil',
                        sources=["upload", "webcam"],
                        image_mode='RGB',
                        layers=False,
                        brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                        # fixed_canvas= True,
                        # width=800,  
                        height=800,
                        # transforms=None,
                        # interactive=True, 
                        elem_id="img_editor",
                        visible= "V" in video_prompt_type_value and "A" in video_prompt_type_value and not "U" in video_prompt_type_value
                    )
                    any_control_image = True
                else:
                    image_mask_guide = gr.ImageEditor(value = None, visible = False, elem_id="img_editor")


                denoising_strength = gr.Slider(0, 1, value= ui_defaults.get("denoising_strength" ,0.5), step=0.01, label=f"Denoising Strength (the Lower the Closer to the Control {'Image' if image_outputs else 'Video'})", visible = "G" in video_prompt_type_value, show_reset_button= False)
                keep_frames_video_guide_visible = not image_outputs and  "V" in video_prompt_type_value and not model_def.get("keep_frames_video_guide_not_supported", False)
                keep_frames_video_guide = gr.Text(value=ui_defaults.get("keep_frames_video_guide","") , visible= keep_frames_video_guide_visible  , scale = 2, label= "Frames to keep in Control Video (empty=All, 1=first, a:b for a range, space to separate values)" ) #, -1=last
                video_guide_outpainting_modes = model_def.get("video_guide_outpainting", [])
                with gr.Column(visible= ("V" in video_prompt_type_value  or "K" in video_prompt_type_value  or "F" in video_prompt_type_value) and image_mode_value in video_guide_outpainting_modes) as video_guide_outpainting_col:
                    video_guide_outpainting_value = ui_defaults.get("video_guide_outpainting","#")
                    video_guide_outpainting = gr.Text(value=video_guide_outpainting_value , visible= False)
                    with gr.Group():
                        video_guide_outpainting_checkbox = gr.Checkbox(label="Enable Spatial Outpainting on Control Video, Landscape or Positioned Reference Frames" if image_mode_value == 0 else "Enable Spatial Outpainting on Control Image", value=len(video_guide_outpainting_value)>0 and not video_guide_outpainting_value.startswith("#") )
                        with gr.Row(visible = not video_guide_outpainting_value.startswith("#")) as video_guide_outpainting_row:
                            video_guide_outpainting_value = video_guide_outpainting_value[1:] if video_guide_outpainting_value.startswith("#") else video_guide_outpainting_value
                            video_guide_outpainting_list = [0] * 4 if len(video_guide_outpainting_value) == 0 else [int(v) for v in video_guide_outpainting_value.split(" ")]
                            video_guide_outpainting_top= gr.Slider(0, 100, value= video_guide_outpainting_list[0], step=5, label="Top %", show_reset_button= False)
                            video_guide_outpainting_bottom = gr.Slider(0, 100, value= video_guide_outpainting_list[1], step=5, label="Bottom %", show_reset_button= False)
                            video_guide_outpainting_left = gr.Slider(0, 100, value= video_guide_outpainting_list[2], step=5, label="Left %", show_reset_button= False)
                            video_guide_outpainting_right = gr.Slider(0, 100, value= video_guide_outpainting_list[3], step=5, label="Right %", show_reset_button= False)
                # image_mask = gr.Image(label= "Image Mask Area (for Inpainting, white = Control Area, black = Unchanged)", type ="pil", visible= image_mode_value==1 and "V" in video_prompt_type_value and "A" in video_prompt_type_value and not "U" in video_prompt_type_value , height = gallery_height, value= ui_defaults.get("image_mask", None)) 
                image_mask = gr.Image(label= "Image Mask Area (for Inpainting, white = Control Area, black = Unchanged)", type ="pil", visible= False, height = gallery_height, value= ui_defaults.get("image_mask", None)) 
                video_mask = gr.Video(label= "Video Mask Area (for Inpainting, white = Control Area, black = Unchanged)", visible= (not image_outputs) and "V" in video_prompt_type_value and "A" in video_prompt_type_value and not "U" in video_prompt_type_value , height = gallery_height, value= ui_defaults.get("video_mask", None)) 

                mask_expand = gr.Slider(-10, 50, value=ui_defaults.get("mask_expand", 0), step=1, label="Expand / Shrink Mask Area", visible= "V" in video_prompt_type_value and "A" in video_prompt_type_value and not "U" in video_prompt_type_value )

                image_refs_single_image_mode = model_def.get("one_image_ref_needed", False)
                image_refs_label = "Start Image" if hunyuan_video_avatar else ("Reference Image" if image_refs_single_image_mode else "Reference Images")  + (" (each Image will be associated to a Sliding Window)" if infinitetalk else "")
                image_refs_row, image_refs, image_refs_extra = get_image_gallery(label= image_refs_label, value = ui_defaults.get("image_refs", None), visible= "I" in video_prompt_type_value, single_image_mode=image_refs_single_image_mode)

                frames_positions = gr.Text(value=ui_defaults.get("frames_positions","") , visible= "F" in video_prompt_type_value, scale = 2, label= "Positions of Injected Frames (1=first, L=last of a window) no position for other Image Refs)" ) 
                image_refs_relative_size = gr.Slider(20, 100, value=ui_defaults.get("image_refs_relative_size", 50), step=1, label="Rescale Internaly Image Ref (% in relation to Output Video) to change Output Composition", visible = model_def.get("any_image_refs_relative_size", False) and image_outputs)

                no_background_removal = model_def.get("no_background_removal", False) or image_ref_choices is None
                background_removal_label = model_def.get("background_removal_label", "Remove Background behind People / Objects") 
 
                remove_background_images_ref = gr.Dropdown(
                    choices=[
                        ("Keep Backgrounds behind all Reference Images", 0),
                        (background_removal_label, 1),
                    ],
                    value=0 if no_background_removal else ui_defaults.get("remove_background_images_ref",1),
                    label="Automatic Removal of Background behind People or Objects in Reference Images", scale = 3, visible= "I" in video_prompt_type_value and not no_background_removal
                )

            any_audio_voices_support = any_audio_track(base_model_type) 
            audio_prompt_type_value = ui_defaults.get("audio_prompt_type", "A" if any_audio_voices_support else "") 
            audio_prompt_type = gr.Text(value= audio_prompt_type_value, visible= False)
            if any_audio_voices_support:
                any_single_speaker = not model_def.get("multi_speakers_only", False)
                if not any_single_speaker and "A" in audio_prompt_type_value and not ("B" in audio_prompt_type_value or "X" in audio_prompt_type_value): audio_prompt_type_value = del_in_sequence(audio_prompt_type_value, "XCPAB")
                any_multi_speakers = not model_def.get("one_speaker_only", False)
                if not any_multi_speakers: audio_prompt_type_value = del_in_sequence(audio_prompt_type_value, "XCPB")

                speaker_choices=[("None", "")]
                if any_single_speaker: speaker_choices += [("One Person Speaking Only", "A")]
                if any_multi_speakers:speaker_choices += [
                        ("Two speakers, Auto Separation of Speakers (will work only if Voices are distinct)", "XA"),
                        ("Two speakers, Speakers Audio sources are assumed to be played in a Row", "CAB"),
                        ("Two speakers, Speakers Audio sources are assumed to be played in Parallel", "PAB")
                    ]
                audio_prompt_type_sources = gr.Dropdown(
                    choices=speaker_choices,
                    value= filter_letters(audio_prompt_type_value, "XCPAB"),
                    label="Voices", scale = 3, visible = multitalk and not image_outputs
                )
            else:
                audio_prompt_type_sources = gr.Dropdown( choices= [""], value = "", visible=False)

            with gr.Row(visible = any_audio_voices_support and not image_outputs) as audio_guide_row:
                audio_guide = gr.Audio(value= ui_defaults.get("audio_guide", None), type="filepath", label="Voice to follow", show_download_button= True, visible= any_audio_voices_support and "A" in audio_prompt_type_value )
                audio_guide2 = gr.Audio(value= ui_defaults.get("audio_guide2", None), type="filepath", label="Voice to follow #2", show_download_button= True, visible= any_audio_voices_support and "B" in audio_prompt_type_value )
            remove_background_sound = gr.Checkbox(label="Video Motion ignores Background Music (to get a better LipSync)", value="V" in audio_prompt_type_value, visible =  any_audio_voices_support and any_letters(audio_prompt_type_value, "ABX") and not image_outputs)
            with gr.Row(visible = any_audio_voices_support and ("B" in audio_prompt_type_value or "X" in audio_prompt_type_value) and not image_outputs ) as speakers_locations_row:
                speakers_locations = gr.Text( ui_defaults.get("speakers_locations", "0:45 55:100"), label="Speakers Locations separated by a Space. Each Location = Left:Right or a BBox Left:Top:Right:Bottom", visible= True)

            advanced_prompt = advanced_ui
            prompt_vars=[]

            if advanced_prompt:
                default_wizard_prompt, variables, values= None, None, None
            else:                 
                default_wizard_prompt, variables, values, errors =  extract_wizard_prompt(launch_prompt)
                advanced_prompt  = len(errors) > 0
            with gr.Column(visible= advanced_prompt) as prompt_column_advanced:
                prompt = gr.Textbox( visible= advanced_prompt, label=prompt_label, value=launch_prompt, lines=3)

            with gr.Column(visible=not advanced_prompt and len(variables) > 0) as prompt_column_wizard_vars:
                gr.Markdown("<B>Please fill the following input fields to adapt automatically the Prompt:</B>")
                wizard_prompt_activated = "off"
                wizard_variables = ""
                with gr.Row():
                    if not advanced_prompt:
                        for variable in variables:
                            value = values.get(variable, "")
                            prompt_vars.append(gr.Textbox( placeholder=variable, min_width=80, show_label= False, info= variable, visible= True, value= "\n".join(value) ))
                        wizard_prompt_activated = "on"
                        if len(variables) > 0:
                            wizard_variables = "\n".join(variables)
                    for _ in range( PROMPT_VARS_MAX - len(prompt_vars)):
                        prompt_vars.append(gr.Textbox(visible= False, min_width=80, show_label= False))
            with gr.Column(visible=not advanced_prompt) as prompt_column_wizard:
                wizard_prompt = gr.Textbox(visible = not advanced_prompt, label=wizard_prompt_label, value=default_wizard_prompt, lines=3)
                wizard_prompt_activated_var = gr.Text(wizard_prompt_activated, visible= False)
                wizard_variables_var = gr.Text(wizard_variables, visible = False)
            with gr.Row(visible= server_config.get("enhancer_enabled", 0) > 0  ) as prompt_enhancer_row:
                on_demand_prompt_enhancer = server_config.get("enhancer_mode", 0) == 1
                prompt_enhancer_value = ui_defaults.get("prompt_enhancer", "")
                if len(prompt_enhancer_value) == 0 and on_demand_prompt_enhancer: prompt_enhancer_value = "T"  
                prompt_enhancer_btn = gr.Button( value ="Enhance Prompt", visible= on_demand_prompt_enhancer, size="lg",  elem_classes="btn_centered")
                prompt_enhancer = gr.Dropdown(
                    choices=
                        ([] if on_demand_prompt_enhancer else [("Disabled", "")]) + 
                        [("Based on Text Prompt Content", "T"),
                        ("Based on Images Prompts Content (such as Start Image and Reference Images)", "I"),
                        ("Based on both Text Prompt and Images Prompts Content", "TI"),
                    ],
                    value=prompt_enhancer_value,
                    label="Enhance Prompt using a LLM", scale = 5,
                    visible= True, show_label= not on_demand_prompt_enhancer,
                )
            with gr.Row():
                fit_canvas = server_config.get("fit_canvas", 0)
                if fit_canvas == 1:
                    label = "Outer Box Resolution (one dimension may be less to preserve video W/H ratio)"
                elif fit_canvas == 2:
                    label = "Output Resolution (Input Images wil be Cropped if the W/H ratio is different)"
                else:
                    label = "Resolution Budget (Pixels will be reallocated to preserve Inputs W/H ratio)" 
                current_resolution_choice = ui_defaults.get("resolution","832x480") if update_form or last_resolution is None else last_resolution
                model_resolutions = model_def.get("resolutions", None)
                resolution_choices, current_resolution_choice = get_resolution_choices(current_resolution_choice, model_resolutions)
                available_groups, selected_group_resolutions, selected_group = group_resolutions(model_def,resolution_choices, current_resolution_choice)
                resolution_group = gr.Dropdown(
                choices = available_groups,
                    value= selected_group,
                    label= "Category" 
                )
                resolution = gr.Dropdown(
                choices = selected_group_resolutions,
                    value= current_resolution_choice,
                    label= label,
                    scale = 5
                )
            with gr.Row():
                batch_size = gr.Slider(1, 16, value=ui_defaults.get("batch_size", 1), step=1, label="Number of Images to Generate", visible = image_outputs)
                if image_outputs:
                    video_length = gr.Slider(1, 9999, value=ui_defaults.get("video_length", 1), step=1, label="Number of frames", visible = False)
                elif recammaster:
                    video_length = gr.Slider(5, 193, value=ui_defaults.get("video_length", get_max_frames(81)), step=4, label="Number of frames (16 = 1s), locked", interactive= False, visible = True)
                else:
                    min_frames, frames_step, _ = get_model_min_frames_and_step(base_model_type)
                    
                    current_video_length = ui_defaults.get("video_length", 81 if get_model_family(base_model_type)=="wan" else 97)

                    computed_fps = get_computed_fps(ui_defaults.get("force_fps",""), base_model_type , ui_defaults.get("video_guide", None), ui_defaults.get("video_source", None))
                    video_length = gr.Slider(min_frames, get_max_frames(737 if test_any_sliding_window(base_model_type) else 337), value=current_video_length, 
                         step=frames_step, label=compute_video_length_label(computed_fps, current_video_length) , visible = True, interactive= True)

            with gr.Row(visible = not lock_inference_steps) as inference_steps_row:                                       
                num_inference_steps = gr.Slider(1, 100, value=ui_defaults.get("num_inference_steps",30), step=1, label="Number of Inference Steps", visible = True)


            show_advanced = gr.Checkbox(label="Advanced Mode", value=advanced_ui)
            with gr.Tabs(visible=advanced_ui) as advanced_row:
                guidance_max_phases = model_def.get("guidance_max_phases", 0)
                no_negative_prompt = model_def.get("no_negative_prompt", False)
                any_audio_guidance = fantasy or multitalk
                with gr.Tab("General"):
                    with gr.Column():
                        with gr.Row():                        
                            seed = gr.Slider(-1, 999999999, value=ui_defaults.get("seed",-1), step=1, label="Seed (-1 for random)", scale=2) 
                            guidance_phases_value = ui_defaults.get("guidance_phases", 1) 
                            guidance_phases = gr.Dropdown(
                                choices=[
                                    ("One Phase", 1),
                                    ("Two Phases", 2),
                                    ("Three Phases", 3)],
                                value= guidance_phases_value,
                                label="Guidance Phases",
                                visible= guidance_max_phases >=2,
                                interactive = not model_def.get("lock_guidance_phases", False)
                            )
                        with gr.Row(visible = guidance_phases_value >=2 ) as guidance_phases_row:
                            multiple_submodels = model_def.get("multiple_submodels", False)
                            model_switch_phase = gr.Dropdown(
                                choices=[
                                    ("Phase 1-2 transition", 1),
                                    ("Phase 2-3 transition", 2)],
                                value=ui_defaults.get("model_switch_phase", 1),
                                label="Model Switch",
                                visible= model_def.get("multiple_submodels", False) and guidance_phases_value >= 3 and multiple_submodels
                            )
                            label ="Phase 1-2" if guidance_phases_value ==3 else ( "Model / Guidance Switch Threshold" if multiple_submodels  else "Guidance Switch Threshold" )
                            switch_threshold = gr.Slider(0, 1000, value=ui_defaults.get("switch_threshold",0), step=1, label = label, visible= guidance_max_phases >= 2 and guidance_phases_value >= 2)
                            switch_threshold2 = gr.Slider(0, 1000, value=ui_defaults.get("switch_threshold2",0), step=1, label="Phase 2-3", visible= guidance_max_phases >= 3 and guidance_phases_value >= 3)
                        with gr.Row(visible = guidance_max_phases >=1 ) as guidance_row:
                            guidance_scale = gr.Slider(1.0, 20.0, value=ui_defaults.get("guidance_scale",5), step=0.5, label="Guidance (CFG)", visible=guidance_max_phases >=1 )
                            guidance2_scale = gr.Slider(1.0, 20.0, value=ui_defaults.get("guidance2_scale",5), step=0.5, label="Guidance2 (CFG)", visible= guidance_max_phases >=2 and guidance_phases_value >= 2)
                            guidance3_scale = gr.Slider(1.0, 20.0, value=ui_defaults.get("guidance3_scale",5), step=0.5, label="Guidance3 (CFG)", visible= guidance_max_phases >=3  and guidance_phases_value >= 3)

                        any_embedded_guidance = model_def.get("embedded_guidance", False)
                        with gr.Row(visible =any_embedded_guidance or any_audio_guidance) as embedded_guidance_row:
                            audio_guidance_scale = gr.Slider(1.0, 20.0, value=ui_defaults.get("audio_guidance_scale", 4), step=0.5, label="Audio Guidance", visible= any_audio_guidance )
                            embedded_guidance_scale = gr.Slider(1.0, 20.0, value=ui_defaults.get("embedded_guidance", 6.0), step=0.5, label="Embedded Guidance Scale", visible=any_embedded_guidance )

                        sample_solver_choices = model_def.get("sample_solvers", None)
                        with gr.Row(visible = sample_solver_choices is not None or not image_outputs) as sample_solver_row:
                            if sample_solver_choices is None:
                                sample_solver = gr.Dropdown( value="",  choices=[ ("", ""), ], visible= False, label= "Sampler Solver / Scheduler" )
                            else:
                                sample_solver = gr.Dropdown( value=ui_defaults.get("sample_solver", sample_solver_choices[0][1]), 
                                    choices= sample_solver_choices, visible= True, label= "Sampler Solver / Scheduler"
                                )
                            flow_shift = gr.Slider(1.0, 25.0, value=ui_defaults.get("flow_shift",3), step=0.1, label="Shift Scale", visible = not image_outputs) 

                        with gr.Row(visible = vace) as control_net_weights_row:
                            control_net_weight = gr.Slider(0.0, 2.0, value=ui_defaults.get("control_net_weight",1), step=0.1, label="Control Net Weight #1", visible=vace)
                            control_net_weight2 = gr.Slider(0.0, 2.0, value=ui_defaults.get("control_net_weight2",1), step=0.1, label="Control Net Weight #2", visible=vace)
                        negative_prompt = gr.Textbox(label="Negative Prompt (ignored if no Guidance that is if CFG = 1)", value=ui_defaults.get("negative_prompt", ""), visible = not (hunyuan_t2v or hunyuan_i2v or no_negative_prompt)  )
                        with gr.Column(visible = vace or t2v or test_class_i2v(model_type)) as NAG_col:
                            gr.Markdown("<B>NAG enforces Negative Prompt even if no Guidance is set (CFG = 1), set NAG Scale to > 1 to enable it</B>")
                            with gr.Row():
                                NAG_scale = gr.Slider(1.0, 20.0, value=ui_defaults.get("NAG_scale",1), step=0.1, label="NAG Scale", visible = True)
                                NAG_tau = gr.Slider(1.0, 5.0, value=ui_defaults.get("NAG_tau",3.5), step=0.1, label="NAG Tau", visible = True)
                                NAG_alpha = gr.Slider(0.0, 2.0, value=ui_defaults.get("NAG_alpha",.5), step=0.1, label="NAG Alpha", visible = True)
                        with gr.Row():
                            repeat_generation = gr.Slider(1, 25.0, value=ui_defaults.get("repeat_generation",1), step=1, label="Num. of Generated Videos per Prompt", visible = not image_outputs) 
                            multi_images_gen_type = gr.Dropdown( value=ui_defaults.get("multi_images_gen_type",0), 
                                choices=[
                                    ("Generate every combination of images and texts", 0),
                                    ("Match images and text prompts", 1),
                                ], visible= test_class_i2v(model_type), label= "Multiple Images as Texts Prompts"
                            )
                with gr.Tab("Loras"):
                    with gr.Column(visible = True): #as loras_column:
                        gr.Markdown("<B>Loras can be used to create special effects on the video by mentioning a trigger word in the Prompt. You can save Loras combinations in presets.</B>")
                        loras_choices = gr.Dropdown(
                            choices=[
                                (lora_name, str(i) ) for i, lora_name in enumerate(loras_names)
                            ],
                            value= launch_loras,
                            multiselect= True,
                            label="Activated Loras"
                        )
                        loras_multipliers = gr.Textbox(label="Loras Multipliers (1.0 by default) separated by Space chars or CR, lines that start with # are ignored", value=launch_multis_str)
                with gr.Tab("Steps Skipping", visible = any_tea_cache or any_mag_cache) as speed_tab:
                    with gr.Column():
                        gr.Markdown("<B>Tea Cache and Mag Cache accelerate the Video Generation by skipping intelligently some steps, the more steps are skipped the lower the quality of the video.</B>")
                        gr.Markdown("<B>Steps Skipping  consumes also VRAM. It is recommended not to skip at least the first 10% steps.</B>")
                        steps_skipping_choices = [("None", "")]
                        if any_tea_cache: steps_skipping_choices += [("Tea Cache", "tea")]
                        if any_mag_cache: steps_skipping_choices += [("Mag Cache", "mag")]
                        skip_steps_cache_type = gr.Dropdown(
                            choices= steps_skipping_choices,
                            value="" if not (any_tea_cache or any_mag_cache) else ui_defaults.get("skip_steps_cache_type",""),
                            visible=True,
                            label="Skip Steps Cache Type"
                        )
 
                        skip_steps_multiplier = gr.Dropdown(
                            choices=[
                                ("around x1.5 speed up", 1.5), 
                                ("around x1.75 speed up", 1.75), 
                                ("around x2 speed up", 2.0), 
                                ("around x2.25 speed up", 2.25), 
                                ("around x2.5 speed up", 2.5), 
                            ],
                            value=float(ui_defaults.get("skip_steps_multiplier",1.75)),
                            visible=True,
                            label="Skip Steps Cache Global Acceleration"
                        )
                        skip_steps_start_step_perc = gr.Slider(0, 100, value=ui_defaults.get("skip_steps_start_step_perc",0), step=1, label="Skip Steps starting moment in % of generation") 

                with gr.Tab("Post Processing"):
                    

                    with gr.Column():
                        gr.Markdown("<B>Upsampling - postprocessing that may improve fluidity and the size of the video</B>")
                        def gen_upsampling_dropdowns(temporal_upsampling, spatial_upsampling , film_grain_intensity, film_grain_saturation, element_class= None, max_height= None, image_outputs = False):
                            temporal_upsampling = gr.Dropdown(
                                choices=[
                                    ("Disabled", ""),
                                    ("Rife x2 frames/s", "rife2"), 
                                    ("Rife x4 frames/s", "rife4"), 
                                ],
                                value=temporal_upsampling,
                                visible=not image_outputs,
                                scale = 1,
                                label="Temporal Upsampling",
                                elem_classes= element_class
                                # max_height = max_height
                            )
                            spatial_upsampling = gr.Dropdown(
                                choices=[
                                    ("Disabled", ""),
                                    ("Lanczos x1.5", "lanczos1.5"), 
                                    ("Lanczos x2.0", "lanczos2"), 
                                ],
                                value=spatial_upsampling,
                                visible=True,
                                scale = 1,
                                label="Spatial Upsampling",
                                elem_classes= element_class
                                # max_height = max_height
                            )

                            with gr.Row():
                                film_grain_intensity = gr.Slider(0, 1, value=film_grain_intensity, step=0.01, label="Film Grain Intensity (0 = disabled)") 
                                film_grain_saturation = gr.Slider(0.0, 1, value=film_grain_saturation, step=0.01, label="Film Grain Saturation") 

                            return temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation
                        temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation = gen_upsampling_dropdowns(ui_defaults.get("temporal_upsampling", ""), ui_defaults.get("spatial_upsampling", ""), ui_defaults.get("film_grain_intensity", 0), ui_defaults.get("film_grain_saturation", 0.5), image_outputs= image_outputs)

                with gr.Tab("Audio", visible = not image_outputs) as audio_tab:
                    with gr.Column(visible =  server_config.get("mmaudio_enabled", 0) != 0) as mmaudio_col:
                        gr.Markdown("<B>Add a soundtrack based on the content of the Generated Video</B>")
                        with gr.Row():
                            MMAudio_setting = gr.Dropdown(
                                choices=[("Disabled", 0),  ("Enabled", 1), ],
                                value=ui_defaults.get("MMAudio_setting", 0), visible=True, scale = 1, label="MMAudio",
                            )
                            # if MMAudio_seed != None:
                            #     MMAudio_seed = gr.Slider(-1, 999999999, value=MMAudio_seed, step=1, scale=3, label="Seed (-1 for random)") 
                        with gr.Row():
                            MMAudio_prompt = gr.Text(ui_defaults.get("MMAudio_prompt", ""), label="Prompt (1 or 2 keywords)")
                            MMAudio_neg_prompt = gr.Text(ui_defaults.get("MMAudio_neg_prompt", ""), label="Negative Prompt (1 or 2 keywords)")
                            

                    with gr.Column(visible = any_control_video) as audio_prompt_type_remux_row:
                        gr.Markdown("<B>You may transfer the existing audio tracks of a Control Video</B>")
                        audio_prompt_type_remux = gr.Dropdown(
                            choices=[
                                ("No Remux", ""),
                                ("Remux Audio Files from Control Video if any and if no MMAudio / Custom Soundtrack", "R"),
                            ],
                            value=filter_letters(audio_prompt_type_value, "R"),
                            label="Remux Audio Files",
                            visible = True
                        )

                    with gr.Column():
                        gr.Markdown("<B>Add Custom Soundtrack to Video</B>")
                        audio_source = gr.Audio(value= ui_defaults.get("audio_source", None), type="filepath", label="Soundtrack", show_download_button= True)
                        
                any_skip_layer_guidance = model_def.get("skip_layer_guidance", False)
                any_cfg_zero = model_def.get("cfg_zero", False)
                any_cfg_star = model_def.get("cfg_star", False)
                any_apg = model_def.get("adaptive_projected_guidance", False)

                with gr.Tab("Quality", visible = vace and image_outputs or any_skip_layer_guidance or any_cfg_zero or any_cfg_star or any_apg ) as quality_tab:
                        with gr.Column(visible = any_skip_layer_guidance ) as skip_layer_guidance_row:
                            gr.Markdown("<B>Skip Layer Guidance (improves video quality, requires guidance > 1)</B>")
                            with gr.Row():
                                slg_switch = gr.Dropdown(
                                    choices=[
                                        ("OFF", 0),
                                        ("ON", 1), 
                                    ],
                                    value=ui_defaults.get("slg_switch",0),
                                    visible=True,
                                    scale = 1,
                                    label="Skip Layer guidance"
                                )
                                slg_layers = gr.Dropdown(
                                    choices=[
                                        (str(i), i ) for i in range(40)
                                    ],
                                    value=ui_defaults.get("slg_layers", [9]),
                                    multiselect= True,
                                    label="Skip Layers",
                                    scale= 3
                                )
                            with gr.Row():
                                slg_start_perc = gr.Slider(0, 100, value=ui_defaults.get("slg_start_perc",10), step=1, label="Denoising Steps % start") 
                                slg_end_perc = gr.Slider(0, 100, value=ui_defaults.get("slg_end_perc",90), step=1, label="Denoising Steps % end") 

                        with gr.Column(visible= any_apg ) as apg_col:
                            gr.Markdown("<B>Correct Progressive Color Saturation during long Video Generations")
                            apg_switch = gr.Dropdown(
                                choices=[
                                    ("OFF", 0),
                                    ("ON", 1), 
                                ],
                                value=ui_defaults.get("apg_switch",0),
                                visible=True,
                                scale = 1,
                                label="Adaptive Projected Guidance (requires Guidance > 1 or Audio Guidance > 1) " if multitalk else "Adaptive Projected Guidance (requires Guidance > 1)",
                            )

                        with gr.Column(visible = any_cfg_star) as cfg_free_guidance_col:
                            gr.Markdown("<B>Classifier-Free Guidance Zero Star, better adherence to Text Prompt")
                            cfg_star_switch = gr.Dropdown(
                                choices=[
                                    ("OFF", 0),
                                    ("ON", 1), 
                                ],
                                value=ui_defaults.get("cfg_star_switch",0),
                                visible=True,
                                scale = 1,
                                label="Classifier-Free Guidance Star (requires Guidance > 1)"
                            )
                            with gr.Row():
                                cfg_zero_step = gr.Slider(-1, 39, value=ui_defaults.get("cfg_zero_step",-1), step=1, label="CFG Zero below this Layer (Extra Process)", visible = any_cfg_zero) 

                        with gr.Column(visible = (vace or t2v or standin) and image_outputs) as min_frames_if_references_col:
                            gr.Markdown("<B>Generating a single Frame alone may not be sufficient to preserve Reference Image Identity / Control Image Information or simply to get a good Image Quality. A workaround is to generate a short Video and keep the First Frame.")
                            min_frames_if_references = gr.Dropdown(
                                choices=[
                                    ("Disabled, generate only one Frame", 1),
                                    ("Generate a 5 Frames long Video only if any Reference Image / Control Image (x1.5 slower)",5),
                                    ("Generate a 9 Frames long Video only if any Reference Image / Control Image (x2.0 slower)",9),
                                    ("Generate a 13 Frames long Video only if any Reference Image / Control Image (x2.5 slower)",13),
                                    ("Generate a 17 Frames long Video only if any Reference Image / Control Image (x3.0 slower)",17),
                                    ("Generate always a 5 Frames long Video (x1.5 slower)",1005),
                                    ("Generate always a 9 Frames long Video (x2.0 slower)",1009),
                                    ("Generate always a 13 Frames long Video (x2.5 slower)",1013),
                                    ("Generate always a 17 Frames long Video (x3.0 slower)",1017),
                                ],
                                value=ui_defaults.get("min_frames_if_references",9 if vace else 1),
                                visible=True,
                                scale = 1,
                                label="Generate more frames to preserve Reference Image Identity / Control Image Information or improve"
                            )

                with gr.Tab("Sliding Window", visible= sliding_window_enabled and not image_outputs) as sliding_window_tab:

                    with gr.Column():  
                        gr.Markdown("<B>A Sliding Window allows you to generate video with a duration not limited by the Model</B>")
                        gr.Markdown("<B>It is automatically turned on if the number of frames to generate is higher than the Window Size</B>")
                        if diffusion_forcing:
                            sliding_window_size = gr.Slider(37, get_max_frames(257), value=ui_defaults.get("sliding_window_size", 129), step=20, label="  (recommended to keep it at 97)")
                            sliding_window_overlap = gr.Slider(17, 97, value=ui_defaults.get("sliding_window_overlap",17), step=20, label="Windows Frames Overlap (needed to maintain continuity between windows, a higher value will require more windows)")
                            sliding_window_color_correction_strength = gr.Slider(0, 1, visible=False, value =0)                            
                            sliding_window_overlap_noise = gr.Slider(0, 100, value=ui_defaults.get("sliding_window_overlap_noise",20), step=1, label="Noise to be added to overlapped frames to reduce blur effect", visible = True)
                            sliding_window_discard_last_frames = gr.Slider(0, 20, value=ui_defaults.get("sliding_window_discard_last_frames", 0), step=4, visible = False)
                        elif ltxv:
                            sliding_window_size = gr.Slider(41, get_max_frames(257), value=ui_defaults.get("sliding_window_size", 129), step=8, label="Sliding Window Size")
                            sliding_window_overlap = gr.Slider(1, 97, value=ui_defaults.get("sliding_window_overlap",9), step=8, label="Windows Frames Overlap (needed to maintain continuity between windows, a higher value will require more windows)")
                            sliding_window_color_correction_strength = gr.Slider(0, 1, visible=False, value =0)                            
                            sliding_window_overlap_noise = gr.Slider(0, 100, value=ui_defaults.get("sliding_window_overlap_noise",20), step=1, label="Noise to be added to overlapped frames to reduce blur effect", visible = False)
                            sliding_window_discard_last_frames = gr.Slider(0, 20, value=ui_defaults.get("sliding_window_discard_last_frames", 0), step=8, label="Discard Last Frames of a Window (that may have bad quality)",  visible = True)
                        elif hunyuan_video_custom_edit:
                            sliding_window_size = gr.Slider(5, get_max_frames(257), value=ui_defaults.get("sliding_window_size", 129), step=4, label="Sliding Window Size")
                            sliding_window_overlap = gr.Slider(1, 97, value=ui_defaults.get("sliding_window_overlap",5), step=4, label="Windows Frames Overlap (needed to maintain continuity between windows, a higher value will require more windows)")
                            sliding_window_color_correction_strength = gr.Slider(0, 1, visible=False, value =0)                            
                            sliding_window_overlap_noise = gr.Slider(0, 150, value=ui_defaults.get("sliding_window_overlap_noise",20), step=1, label="Noise to be added to overlapped frames to reduce blur effect", visible = False)
                            sliding_window_discard_last_frames = gr.Slider(0, 20, value=ui_defaults.get("sliding_window_discard_last_frames", 0), step=4, label="Discard Last Frames of a Window (that may have bad quality)", visible = True)
                        else: # Vace, Multitalk
                            sliding_window_defaults = model_def.get("sliding_window_defaults", {})                            
                            sliding_window_size = gr.Slider(5, get_max_frames(257), value=ui_defaults.get("sliding_window_size", 129), step=4, label="Sliding Window Size")
                            sliding_window_overlap = gr.Slider(sliding_window_defaults.get("overlap_min", 1), sliding_window_defaults.get("overlap_max", 97), value=ui_defaults.get("sliding_window_overlap",sliding_window_defaults.get("overlap_default", 5)), step=sliding_window_defaults.get("overlap_step", 4), label="Windows Frames Overlap (needed to maintain continuity between windows, a higher value will require more windows)")
                            sliding_window_color_correction_strength = gr.Slider(0, 1, value=ui_defaults.get("sliding_window_color_correction_strength",0), step=0.01, label="Color Correction Strength (match colors of new window with previous one, 0 = disabled)", visible = True)
                            sliding_window_overlap_noise = gr.Slider(0, 150, value=ui_defaults.get("sliding_window_overlap_noise",20 if vace else 0), step=1, label="Noise to be added to overlapped frames to reduce blur effect" , visible = vace)
                            sliding_window_discard_last_frames = gr.Slider(0, 20, value=ui_defaults.get("sliding_window_discard_last_frames", 0), step=4, label="Discard Last Frames of a Window (that may have bad quality)", visible = True)

                        video_prompt_type_alignment = gr.Dropdown(
                            choices=[
                                ("Aligned to the beginning of the Source Video", ""),
                                ("Aligned to the beginning of the First Window of the new Video Sample", "T"),
                            ],
                            value=filter_letters(video_prompt_type_value, "T"),
                            label="Control Video / Control Audio / Positioned Frames Temporal Alignment when any Video to continue",
                            visible = vace or ltxv or t2v or infinitetalk
                        )

                        multi_prompts_gen_type = gr.Dropdown(
                            choices=[
                                ("Will create a new generated Video added to the Generation Queue", 0),
                                ("Will be used for a new Sliding Window of the same Video Generation", 1),
                           ],
                            value=ui_defaults.get("multi_prompts_gen_type",0),
                            visible=True,
                            scale = 1,
                            label="Images & Text Prompts separated by a Carriage Return" if (any_start_image or any_end_image) else "Text Prompts separated by a Carriage Return"
                        )
                        
                with gr.Tab("Misc.", visible = True) as misc_tab:
                    with gr.Column(visible = not (recammaster or ltxv or diffusion_forcing)) as RIFLEx_setting_col:
                        gr.Markdown("<B>With Riflex you can generate videos longer than 5s which is the default duration of videos used to train the model</B>")
                        RIFLEx_setting = gr.Dropdown(
                            choices=[
                                ("Auto (ON if Video longer than 5s)", 0),
                                ("Always ON", 1), 
                                ("Always OFF", 2), 
                            ],
                            value=ui_defaults.get("RIFLEx_setting",0),
                            label="RIFLEx positional embedding to generate long video",
                            visible = True
                        )

                    gr.Markdown("<B>You can change the Default number of Frames Per Second of the output Video, in the absence of Control Video this may create unwanted slow down / acceleration</B>")
                    force_fps_choices =  [(f"Model Default ({fps} fps)", "")]
                    if any_control_video and (any_video_source or recammaster):
                        force_fps_choices +=  [("Auto fps: Source Video if any, or Control Video if any, or Model Default", "auto")]
                    elif any_control_video :
                        force_fps_choices +=  [("Auto fps: Control Video if any, or Model Default", "auto")]
                    elif any_control_video and (any_video_source or recammaster):
                        force_fps_choices +=  [("Auto fps: Source Video if any, or Model Default", "auto")]
                    if any_control_video:
                        force_fps_choices +=  [("Control Video fps", "control")]
                    if any_video_source or recammaster:
                        force_fps_choices +=  [("Source Video fps", "source")]
                    force_fps_choices += [
                            ("15", "15"), 
                            ("16", "16"), 
                            ("23", "23"), 
                            ("24", "24"), 
                            ("25", "25"), 
                            ("30", "30"), 
                        ]
                    
                    force_fps = gr.Dropdown(
                        choices=force_fps_choices,
                        value=ui_defaults.get("force_fps",""),
                        label=f"Override Frames Per Second (model default={fps} fps)"
                    )


                    gr.Markdown("<B>You can set a more agressive Memory Profile if you generate only Short Videos or Images<B>")
                    override_profile = gr.Dropdown(
                        choices=[("Default Memory Profile", -1)] + memory_profile_choices,
                        value=ui_defaults.get("override_profile", -1),
                        label=f"Override Memory Profile"
                    )

            with gr.Row():
                save_settings_btn = gr.Button("Set Settings as Default", visible = not args.lock_config)
                export_settings_from_file_btn = gr.Button("Export Settings to File")
            with gr.Row():
                settings_file = gr.File(height=41,label="Load Settings From Video / Image / JSON")
                settings_base64_output = gr.Text(interactive= False, visible=False, value = "")
                settings_filename =  gr.Text(interactive= False, visible=False, value = "")
            
            mode = gr.Text(value="", visible = False)

        with gr.Column():
            if not update_form:
                gen_status = gr.Text(interactive= False, label = "Status")
                status_trigger = gr.Text(interactive= False, visible=False)
                default_files = []
                output = gr.Gallery(value =default_files, label="Generated videos", preview= True, show_label=False, elem_id="gallery" , columns=[3], rows=[1], object_fit="contain", height=450, selected_index=0, interactive= False)
                output_trigger = gr.Text(interactive= False, visible=False)
                refresh_form_trigger = gr.Text(interactive= False, visible=False)
                fill_wizard_prompt_trigger = gr.Text(interactive= False, visible=False)
                save_form_trigger = gr.Text(interactive= False, visible=False)

            with gr.Accordion("Video Info and Late Post Processing & Audio Remuxing", open=False) as video_info_accordion:
                with gr.Tabs() as video_info_tabs:
                    with gr.Tab("Information", id="video_info"):
                        default_visibility = {} if update_form else {"visible" : False}                        
                        video_info = gr.HTML(visible=True, min_height=100, value=get_default_video_info()) 
                        with gr.Row(**default_visibility) as video_buttons_row:
                            video_info_extract_settings_btn = gr.Button("Extract Settings", min_width= 1, size ="sm")
                            video_info_to_video_source_btn = gr.Button("To Video Source", min_width= 1, size ="sm", visible = any_video_source)
                            video_info_to_control_video_btn = gr.Button("To Control Video", min_width= 1, size ="sm", visible = any_control_video )
                            video_info_eject_video_btn = gr.Button("Eject Video", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility) as image_buttons_row:
                            video_info_extract_image_settings_btn = gr.Button("Extract Settings", min_width= 1, size ="sm")
                            video_info_to_start_image_btn = gr.Button("To Start Image", size ="sm", min_width= 1, visible = any_start_image )
                            video_info_to_end_image_btn = gr.Button("To End Image", size ="sm", min_width= 1, visible = any_end_image)
                            video_info_to_image_mask_btn = gr.Button("To Mask Image", min_width= 1, size ="sm", visible = any_image_mask and False)
                            video_info_to_reference_image_btn = gr.Button("To Reference Image", min_width= 1, size ="sm", visible = any_reference_image)
                            video_info_to_image_guide_btn = gr.Button("To Control Image", min_width= 1, size ="sm", visible = any_control_image )
                            video_info_eject_image_btn = gr.Button("Eject Image", min_width= 1, size ="sm")
                    with gr.Tab("Post Processing", id= "post_processing", visible = True) as video_postprocessing_tab:
                        with gr.Group(elem_classes= "postprocess"):
                            with gr.Column():
                                PP_temporal_upsampling, PP_spatial_upsampling, PP_film_grain_intensity, PP_film_grain_saturation = gen_upsampling_dropdowns("",  "", 0, 0.5, element_class ="postprocess", image_outputs = False)
                        with gr.Row():
                            video_info_postprocessing_btn = gr.Button("Apply Postprocessing", size ="sm", visible=True)
                            video_info_eject_video2_btn = gr.Button("Eject Video", size ="sm", visible=True)
                    with gr.Tab("Audio Remuxing", id= "audio_remuxing", visible = True) as audio_remuxing_tab:
                        with gr.Group(elem_classes= "postprocess"):
                            with gr.Column(visible = server_config.get("mmaudio_enabled", 0) != 0) as PP_MMAudio_col:
                                with gr.Row():
                                    PP_MMAudio_setting = gr.Dropdown(
                                        choices=[("Add Custom Audio Sountrack", 0),  ("Use MMAudio to generate a Soundtrack based on the Video", 1), ],
                                        value=0, visible=True, scale = 1, label="MMAudio", show_label= False, elem_classes= "postprocess",
                                    )
                                with gr.Column(visible = False) as PP_MMAudio_row:
                                    with gr.Row():
                                        PP_MMAudio_prompt = gr.Text("", label="Prompt (1 or 2 keywords)", elem_classes= "postprocess")
                                        PP_MMAudio_neg_prompt = gr.Text("", label="Negative Prompt (1 or 2 keywords)", elem_classes= "postprocess")
                                    PP_MMAudio_seed = gr.Slider(-1, 999999999, value=-1, step=1, label="Seed (-1 for random)") 
                                    PP_repeat_generation = gr.Slider(1, 25.0, value=1, step=1, label="Number of Sample Videos to Generate") 
                            with gr.Row(visible = True) as PP_custom_audio_row:
                                    PP_custom_audio = gr.Audio(label = "Soundtrack", type="filepath", show_download_button= True,)
                        with gr.Row():
                            video_info_remux_audio_btn = gr.Button("Remux Audio", size ="sm", visible=True)
                            video_info_eject_video3_btn = gr.Button("Eject Video", size ="sm", visible=True)
                    with gr.Tab("Add Videos / Images", id= "video_add"):
                        files_to_load = gr.Files(label= "Files to Load in Gallery", height=120)
                        with gr.Row():
                            video_info_add_videos_btn = gr.Button("Add Videos / Images", size ="sm")
 
            if not update_form:
                if tab_id == 'edit':
                    edit_btn = gr.Button("Edit")
                    cancel_btn = gr.Button("Cancel")
                else:
                    generate_btn = gr.Button("Generate")
                    add_to_queue_btn = gr.Button("Add New Prompt To Queue", visible=False)
                generate_trigger = gr.Text(visible = False) 
                add_to_queue_trigger = gr.Text(visible = False)

                with gr.Column(visible= False) as current_gen_column:
                    with gr.Accordion("Preview", open=False):
                        preview = gr.Image(label="Preview", height=200, show_label= False)
                        preview_trigger = gr.Text(visible= False)
                    gen_info = gr.HTML(visible=False, min_height=1) 
                    with gr.Row() as current_gen_buttons_row:
                        onemoresample_btn = gr.Button("One More Sample Please !", visible = True)
                        onemorewindow_btn = gr.Button("Extend this Sample Please !", visible = False)
                        abort_btn = gr.Button("Abort", visible = True)
                with gr.Accordion("Queue Management", open=False) as queue_accordion:
                    with gr.Row():
                        queue_html = gr.HTML(
                            value=generate_queue_html(state_dict["gen"]["queue"]),
                            elem_id="queue_html_container"
                        )
                    queue_action_input = gr.Text(elem_id="queue_action_input", visible=False)
                    queue_action_trigger = gr.Button(elem_id="queue_action_trigger", visible=False)
                    with gr.Row(visible= True):
                        queue_zip_base64_output = gr.Text(visible=False)
                        save_queue_btn = gr.DownloadButton("Save Queue", size="sm")
                        load_queue_btn = gr.UploadButton("Load Queue", file_types=[".zip"], size="sm")
                        clear_queue_btn = gr.Button("Clear Queue", size="sm", variant="stop")
                        quit_button = gr.Button("Save and Quit", size="sm", variant="secondary")
                        with gr.Row(visible=False) as quit_confirmation_row:
                            confirm_quit_button = gr.Button("Confirm", elem_id="comfirm_quit_btn_hidden", size="sm", variant="stop")
                            cancel_quit_button = gr.Button("Cancel", size="sm", variant="secondary")
                        hidden_force_quit_trigger = gr.Button("force_quit", visible=False, elem_id="force_quit_btn_hidden")
                        hidden_countdown_state = gr.Number(value=-1, visible=False, elem_id="hidden_countdown_state_num")
                        single_hidden_trigger_btn = gr.Button("trigger_countdown", visible=False, elem_id="trigger_info_single_btn")

        extra_inputs = prompt_vars + [wizard_prompt, wizard_variables_var, wizard_prompt_activated_var, video_prompt_column, image_prompt_column, image_prompt_type_group, image_prompt_type_radio, image_prompt_type_endcheckbox,
                                      prompt_column_advanced, prompt_column_wizard_vars, prompt_column_wizard, lset_name, save_lset_prompt_drop, advanced_row, speed_tab, audio_tab, mmaudio_col, quality_tab,
                                      sliding_window_tab, misc_tab, prompt_enhancer_row, inference_steps_row, skip_layer_guidance_row, audio_guide_row, RIFLEx_setting_col,
                                      video_prompt_type_video_guide, video_prompt_type_video_guide_alt, video_prompt_type_video_mask, video_prompt_type_image_refs, apg_col, audio_prompt_type_sources,  audio_prompt_type_remux, audio_prompt_type_remux_row,
                                      video_guide_outpainting_col,video_guide_outpainting_top, video_guide_outpainting_bottom, video_guide_outpainting_left, video_guide_outpainting_right,
                                      video_guide_outpainting_checkbox, video_guide_outpainting_row, show_advanced, video_info_to_control_video_btn, video_info_to_video_source_btn, sample_solver_row,
                                      video_buttons_row, image_buttons_row, video_postprocessing_tab, audio_remuxing_tab, PP_MMAudio_row, PP_custom_audio_row, 
                                      video_info_to_start_image_btn, video_info_to_end_image_btn, video_info_to_reference_image_btn, video_info_to_image_guide_btn, video_info_to_image_mask_btn,
                                      NAG_col, remove_background_sound , speakers_locations_row, embedded_guidance_row, guidance_phases_row, guidance_row, resolution_group, cfg_free_guidance_col, control_net_weights_row, guide_selection_row, image_mode_tabs, 
                                      min_frames_if_references_col, video_prompt_type_alignment, prompt_enhancer_btn, tab_inpaint, tab_t2v] + image_start_extra + image_end_extra + image_refs_extra #  presets_column,
        if update_form:
            locals_dict = locals()
            gen_inputs = [state_dict if k=="state" else locals_dict[k]  for k in inputs_names] + [state_dict] + extra_inputs
            return gen_inputs
        else:
            target_state = gr.Text(value = "state", interactive= False, visible= False)
            target_settings = gr.Text(value = "settings", interactive= False, visible= False)
            last_choice = gr.Number(value =-1, interactive= False, visible= False)

            resolution_group.input(fn=change_resolution_group, inputs=[state, resolution_group], outputs=[resolution], show_progress="hidden")
            resolution.change(fn=record_last_resolution, inputs=[state, resolution])


            # video_length.release(fn=refresh_video_length_label, inputs=[state, video_length ], outputs = video_length, trigger_mode="always_last" )
            gr.on(triggers=[video_length.release, force_fps.change, video_guide.change, video_source.change], fn=refresh_video_length_label, inputs=[state, video_length, force_fps, video_guide, video_source] , outputs = video_length, trigger_mode="always_last", show_progress="hidden"  )
            guidance_phases.change(fn=change_guidance_phases, inputs= [state, guidance_phases], outputs =[model_switch_phase, guidance_phases_row, switch_threshold, switch_threshold2, guidance2_scale, guidance3_scale ])
            audio_prompt_type_remux.change(fn=refresh_audio_prompt_type_remux, inputs=[state, audio_prompt_type, audio_prompt_type_remux], outputs=[audio_prompt_type])
            remove_background_sound.change(fn=refresh_remove_background_sound, inputs=[state, audio_prompt_type, remove_background_sound], outputs=[audio_prompt_type])
            audio_prompt_type_sources.change(fn=refresh_audio_prompt_type_sources, inputs=[state, audio_prompt_type, audio_prompt_type_sources], outputs=[audio_prompt_type, audio_guide, audio_guide2, speakers_locations_row, remove_background_sound])
            image_prompt_type_radio.change(fn=refresh_image_prompt_type_radio, inputs=[state, image_prompt_type, image_prompt_type_radio], outputs=[image_prompt_type, image_start_row, image_end_row, video_source, keep_frames_video_source, image_prompt_type_endcheckbox], show_progress="hidden" ) 
            image_prompt_type_endcheckbox.change(fn=refresh_image_prompt_type_endcheckbox, inputs=[state, image_prompt_type, image_prompt_type_radio, image_prompt_type_endcheckbox], outputs=[image_prompt_type, image_end_row] ) 
            video_prompt_type_image_refs.input(fn=refresh_video_prompt_type_image_refs, inputs = [state, video_prompt_type, video_prompt_type_image_refs,image_mode], outputs = [video_prompt_type, image_refs_row, remove_background_images_ref,  image_refs_relative_size, frames_positions,video_guide_outpainting_col], show_progress="hidden")
            video_prompt_type_video_guide.input(fn=refresh_video_prompt_type_video_guide, inputs = [state, video_prompt_type, video_prompt_type_video_guide, image_mode, image_mask_guide, image_guide, image_mask], outputs = [video_prompt_type, video_guide, image_guide, keep_frames_video_guide, denoising_strength, video_guide_outpainting_col, video_prompt_type_video_mask, video_mask, image_mask, image_mask_guide, mask_expand], show_progress="hidden") 
            video_prompt_type_video_guide_alt.input(fn=refresh_video_prompt_type_video_guide_alt, inputs = [state, video_prompt_type, video_prompt_type_video_guide_alt, image_mode], outputs = [video_prompt_type, video_guide, image_guide, image_refs_row, denoising_strength ], show_progress="hidden")
            video_prompt_type_video_mask.input(fn=refresh_video_prompt_type_video_mask, inputs = [state, video_prompt_type, video_prompt_type_video_mask, image_mode, image_mask_guide, image_guide, image_mask], outputs = [video_prompt_type, video_mask, image_mask_guide, image_guide, image_mask, mask_expand], show_progress="hidden")
            video_prompt_type_alignment.input(fn=refresh_video_prompt_type_alignment, inputs = [state, video_prompt_type, video_prompt_type_alignment], outputs = [video_prompt_type])
            multi_prompts_gen_type.select(fn=refresh_prompt_labels, inputs=[multi_prompts_gen_type, image_mode], outputs=[prompt, wizard_prompt, image_end], show_progress="hidden")
            video_guide_outpainting_top.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_top, gr.State(0)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_bottom.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_bottom,gr.State(1)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_left.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_left,gr.State(2)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_right.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_right,gr.State(3)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_checkbox.input(fn=refresh_video_guide_outpainting_row, inputs=[video_guide_outpainting_checkbox, video_guide_outpainting], outputs= [video_guide_outpainting_row,video_guide_outpainting])
            show_advanced.change(fn=switch_advanced, inputs=[state, show_advanced, lset_name], outputs=[advanced_row, preset_buttons_rows, refresh_lora_btn, refresh2_row ,lset_name]).then(
                fn=switch_prompt_type, inputs = [state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars], outputs = [wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, prompt_column_advanced, prompt_column_wizard, prompt_column_wizard_vars, *prompt_vars])
            queue_action_trigger.click(fn=handle_queue_action, inputs=[state, queue_action_input], outputs=[queue_html, main_tabs], show_progress="hidden")
            gr.on( triggers=[output.change, output.select], fn=select_video, inputs=[state, output], outputs=[last_choice, video_info, video_buttons_row, image_buttons_row, video_postprocessing_tab, audio_remuxing_tab], show_progress="hidden")
            preview_trigger.change(refresh_preview, inputs= [state], outputs= [preview], show_progress="hidden")
            PP_MMAudio_setting.change(fn = lambda value : [gr.update(visible = value == 1), gr.update(visible = value == 0)] , inputs = [PP_MMAudio_setting], outputs = [PP_MMAudio_row, PP_custom_audio_row] )
            def refresh_status_async(state, progress=gr.Progress()):
                gen = get_gen_info(state)
                gen["progress"] = progress

                while True: 
                    progress_args= gen.get("progress_args", None)
                    if progress_args != None:
                        progress(*progress_args)
                        gen["progress_args"] = None
                    status= gen.get("status","")
                    if status == None or len(status) > 0:
                        yield status
                        gen["status"]= ""
                    if not gen.get("status_display", False):
                        return
                    time.sleep(0.5)

            def activate_status(state):
                if state.get("validate_success",0) != 1:
                    return
                gen = get_gen_info(state)
                gen["status_display"] = True
                return time.time()

            start_quit_timer_js, cancel_quit_timer_js, trigger_zip_download_js, trigger_settings_download_js, click_brush_js = get_js()

            status_trigger.change(refresh_status_async, inputs= [state] , outputs= [gen_status], show_progress_on= [gen_status])

            if tab_id == 'generate':
                output_trigger.change(refresh_gallery,
                    inputs = [state], 
                    outputs = [output, gen_info, generate_btn, add_to_queue_btn, current_gen_column, current_gen_buttons_row, queue_html, abort_btn, onemorewindow_btn],
                    show_progress="hidden"
                    )


            modal_action_trigger.click(
                fn=show_modal_image,
                inputs=[state, modal_action_input],
                outputs=[modal_html_display, modal_container],
                show_progress="hidden"
            )
            close_modal_button.click(
                fn=lambda: gr.Column(visible=False),
                inputs=[],
                outputs=[modal_container],
                show_progress="hidden"
            )
            abort_btn.click(abort_generation, [state], [ abort_btn] ) #.then(refresh_gallery, inputs = [state, gen_info], outputs = [output, gen_info, queue_html] )
            onemoresample_btn.click(fn=one_more_sample,inputs=[state], outputs= [state])
            onemorewindow_btn.click(fn=one_more_window,inputs=[state], outputs= [state])

            inputs_names= list(inspect.signature(save_inputs).parameters)[1:-1]
            locals_dict = locals()
            gen_inputs = [locals_dict[k] for k in inputs_names] + [state]
            save_settings_btn.click( fn=validate_wizard_prompt, inputs =[state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] , outputs= [prompt]).then(
                save_inputs, inputs =[target_settings] + gen_inputs, outputs = [])

            gr.on( triggers=[video_info_extract_settings_btn.click, video_info_extract_image_settings_btn.click], fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then( fn=use_video_settings, inputs =[state, output, last_choice] , outputs= [model_family, model_choice, refresh_form_trigger])


            prompt_enhancer_btn.click(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then( fn=enhance_prompt, inputs =[state, prompt, prompt_enhancer, multi_images_gen_type, override_profile ] , outputs= [prompt, wizard_prompt])

            save_form_trigger.change(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            )

            main_tabs.select(fn=detect_auto_save_form, inputs= [state], outputs= save_form_trigger, trigger_mode="multiple")


            video_info_add_videos_btn.click(fn=add_videos_to_gallery, inputs =[state, output, last_choice, files_to_load], outputs = [output, files_to_load, video_info_tabs] )
            gr.on(triggers=[video_info_eject_video_btn.click, video_info_eject_video2_btn.click, video_info_eject_video3_btn.click, video_info_eject_image_btn.click], fn=eject_video_from_gallery, inputs =[state, output, last_choice], outputs = [output, video_info, video_buttons_row] )
            video_info_to_control_video_btn.click(fn=video_to_control_video, inputs =[state, output, last_choice], outputs = [video_guide] )
            video_info_to_video_source_btn.click(fn=video_to_source_video, inputs =[state, output, last_choice], outputs = [video_source] )
            video_info_to_start_image_btn.click(fn=image_to_ref_image_add, inputs =[state, output, last_choice, image_start, gr.State("Start Image")], outputs = [image_start] )
            video_info_to_end_image_btn.click(fn=image_to_ref_image_add, inputs =[state, output, last_choice, image_end, gr.State("End Image")], outputs = [image_end] )
            video_info_to_image_guide_btn.click(fn=image_to_ref_image_guide, inputs =[state, output, last_choice], outputs = [image_guide, image_mask_guide]).then(fn=None, inputs=[], outputs=[], js=click_brush_js )
            video_info_to_image_mask_btn.click(fn=image_to_ref_image_set, inputs =[state, output, last_choice, image_mask, gr.State("Image Mask")], outputs = [image_mask] )
            video_info_to_reference_image_btn.click(fn=image_to_ref_image_add, inputs =[state, output, last_choice, image_refs, gr.State("Ref Image")],  outputs = [image_refs] )
            video_info_postprocessing_btn.click(fn=apply_post_processing, inputs =[state, output, last_choice, PP_temporal_upsampling, PP_spatial_upsampling, PP_film_grain_intensity, PP_film_grain_saturation], outputs = [mode, generate_trigger, add_to_queue_trigger ] )
            video_info_remux_audio_btn.click(fn=remux_audio, inputs =[state, output, last_choice, PP_MMAudio_setting, PP_MMAudio_prompt, PP_MMAudio_neg_prompt, PP_MMAudio_seed, PP_repeat_generation, PP_custom_audio], outputs = [mode, generate_trigger, add_to_queue_trigger ] )
            save_lset_btn.click(validate_save_lset, inputs=[state, lset_name], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_save_lset_btn, cancel_lset_btn, save_lset_prompt_drop])
            delete_lset_btn.click(validate_delete_lset, inputs=[state, lset_name], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_delete_lset_btn, cancel_lset_btn ])
            confirm_save_lset_btn.click(fn=validate_wizard_prompt, inputs =[state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars] , outputs= [prompt], show_progress="hidden",).then(
                fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None).then(
                fn=save_lset, inputs=[state, lset_name, loras_choices, loras_multipliers, prompt, save_lset_prompt_drop], outputs=[lset_name, apply_lset_btn,refresh_lora_btn, delete_lset_btn, save_lset_btn, confirm_save_lset_btn, cancel_lset_btn, save_lset_prompt_drop])
            confirm_delete_lset_btn.click(delete_lset, inputs=[state, lset_name], outputs=[lset_name, apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_delete_lset_btn, cancel_lset_btn ])
            cancel_lset_btn.click(cancel_lset, inputs=[], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn, confirm_delete_lset_btn,confirm_save_lset_btn, cancel_lset_btn,save_lset_prompt_drop ])
            apply_lset_btn.click(fn=save_inputs, inputs =[target_state] + gen_inputs, outputs= None).then(fn=apply_lset, 
                inputs=[state, wizard_prompt_activated_var, lset_name,loras_choices, loras_multipliers, prompt], outputs=[wizard_prompt_activated_var, loras_choices, loras_multipliers, prompt, fill_wizard_prompt_trigger, model_family, model_choice, refresh_form_trigger])
            refresh_lora_btn.click(refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])
            refresh_lora_btn2.click(refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])

            lset_name.select(fn=update_lset_type, inputs=[state, lset_name], outputs=save_lset_prompt_drop)
            export_settings_from_file_btn.click(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=export_settings, 
                inputs =[state], 
                outputs= [settings_base64_output, settings_filename]
            ).then(
                fn=None,
                inputs=[settings_base64_output, settings_filename],
                outputs=None,
                js=trigger_settings_download_js
            )
            
            image_mode_tabs.select(fn=record_image_mode_tab, inputs=[state], outputs= None
            ).then(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=switch_image_mode, inputs =[state] , outputs= [refresh_form_trigger], trigger_mode="multiple")

            settings_file.upload(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=load_settings_from_file, inputs =[state, settings_file] , outputs= [model_family, model_choice, refresh_form_trigger, settings_file])


            fill_wizard_prompt_trigger.change(
                fn = fill_wizard_prompt, inputs = [state, wizard_prompt_activated_var, prompt, wizard_prompt], outputs = [ wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, prompt_column_advanced, prompt_column_wizard, prompt_column_wizard_vars, *prompt_vars]
            )

            if tab_id == 'edit':
                edit_inputs_names = list(inspect.signature(edit_task_in_queue).parameters)[:-1]
                edit_inputs_components = [locals_dict[k] for k in edit_inputs_names]
                edit_btn.click(
                    fn=validate_wizard_prompt,
                    inputs=[state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars],
                    outputs=[prompt]
                ).then(
                    fn=edit_task_in_queue,
                    inputs=edit_inputs_components + [state],
                    outputs=[queue_html, main_tabs]
                )
                cancel_btn.click(
                    fn=cancel_edit,
                    inputs=[state],
                    outputs=[main_tabs]
                )

            refresh_form_trigger.change(fn= fill_inputs, 
                inputs=[state],
                outputs=gen_inputs + extra_inputs,
                show_progress= "full" if args.debug_gen_form else "hidden",
            ).then(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars],
                outputs= [prompt],
                show_progress="hidden",
            )                

            model_family.input(fn=change_model_family, inputs=[state, model_family], outputs= [model_choice])

            model_choice.change(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn= change_model,
                inputs=[state, model_choice],
                outputs= [header]
            ).then(fn= fill_inputs, 
                inputs=[state],
                outputs=gen_inputs + extra_inputs,
                show_progress="full" if args.debug_gen_form else "hidden",
            ).then(fn= preload_model_when_switching, 
                inputs=[state],
                outputs=[gen_status])
            
            if tab_id == 'generate':
                generate_btn.click(fn = init_generate, inputs = [state, output, last_choice], outputs=[generate_trigger, mode])
                add_to_queue_btn.click(fn = lambda : (get_unique_id(), ""), inputs = None, outputs=[add_to_queue_trigger, mode])
                # gr.on(triggers=[add_to_queue_btn.click, add_to_queue_trigger.change],fn=validate_wizard_prompt, 
                add_to_queue_trigger.change(fn=validate_wizard_prompt, 
                    inputs =[state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                    outputs= [prompt],
                    show_progress="hidden",
                ).then(fn=save_inputs,
                    inputs =[target_state] + gen_inputs,
                    outputs= None
                ).then(fn=process_prompt_and_add_tasks,
                    inputs = [state, model_choice],
                    outputs=[queue_html, queue_accordion],
                    show_progress="hidden",
                ).then(
                    fn=update_status,
                    inputs = [state],
                )

                generate_trigger.change(fn=validate_wizard_prompt,
                    inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                    outputs= [prompt],
                    show_progress="hidden",
                ).then(fn=save_inputs,
                    inputs =[target_state] + gen_inputs,
                    outputs= None
                ).then(fn=process_prompt_and_add_tasks,
                    inputs = [state, model_choice],
                    outputs= [queue_html, queue_accordion],
                    show_progress="hidden",
                ).then(fn=prepare_generate_video,
                    inputs= [state],
                    outputs= [generate_btn, add_to_queue_btn, current_gen_column, current_gen_buttons_row]
                ).then(fn=activate_status,
                    inputs= [state],
                    outputs= [status_trigger],             
                ).then(fn=process_tasks,
                    inputs= [state],
                    outputs= [preview_trigger, output_trigger], 
                    show_progress="hidden",
                ).then(finalize_generation,
                    inputs= [state], 
                    outputs= [output, abort_btn, generate_btn, add_to_queue_btn, current_gen_column, gen_info]
                ).then(
                    fn=lambda s: gr.Accordion(open=False) if len(get_gen_info(s).get("queue", [])) <= 1 else gr.update(),
                    inputs=[state],
                    outputs=[queue_accordion]
                ).then(unload_model_if_needed,
                    inputs= [state], 
                    outputs= []
                )

                gr.on(triggers=[load_queue_btn.upload, main.load],
                    fn=load_queue_action,
                    inputs=[load_queue_btn, state],
                    outputs=[queue_html]
                ).then(
                     fn=lambda s: (gr.update(visible=bool(get_gen_info(s).get("queue",[]))), gr.Accordion(open=True)) if bool(get_gen_info(s).get("queue",[])) else (gr.update(visible=False), gr.update()),
                     inputs=[state],
                     outputs=[current_gen_column, queue_accordion]
                ).then(
                    fn=init_process_queue_if_any,
                    inputs=[state],
                    outputs=[generate_btn, add_to_queue_btn, current_gen_column, ]
                ).then(fn=activate_status,
                    inputs= [state],
                    outputs= [status_trigger],             
                ).then(
                    fn=process_tasks,
                    inputs=[state],
                    outputs=[preview_trigger, output_trigger],
                    trigger_mode="once"
                ).then(
                    fn=finalize_generation_with_state,
                    inputs=[state],
                    outputs=[output, abort_btn, generate_btn, add_to_queue_btn, current_gen_column, gen_info, queue_accordion, state],
                    trigger_mode="always_last"
                ).then(
                    unload_model_if_needed,
                     inputs= [state],
                     outputs= []
                )



            single_hidden_trigger_btn.click(
                fn=show_countdown_info_from_state,
                inputs=[hidden_countdown_state],
                outputs=[hidden_countdown_state]
            )
            quit_button.click(
                fn=start_quit_process,
                inputs=[],
                outputs=[hidden_countdown_state, quit_button, quit_confirmation_row]
            ).then(
                fn=None, inputs=None, outputs=None, js=start_quit_timer_js
            )

            confirm_quit_button.click(
                fn=quit_application,
                inputs=[],
                outputs=[]
            ).then(
                fn=None, inputs=None, outputs=None, js=cancel_quit_timer_js
            )

            cancel_quit_button.click(
                fn=cancel_quit_process,
                inputs=[],
                outputs=[hidden_countdown_state, quit_button, quit_confirmation_row]
            ).then(
                fn=None, inputs=None, outputs=None, js=cancel_quit_timer_js
            )

            hidden_force_quit_trigger.click(
                fn=quit_application,
                inputs=[],
                outputs=[]
            )

            save_queue_btn.click(
                fn=save_queue_action,
                inputs=[state],
                outputs=[queue_zip_base64_output]
            ).then(
                fn=None,
                inputs=[queue_zip_base64_output],
                outputs=None,
                js=trigger_zip_download_js
            )

            clear_queue_btn.click(
                fn=clear_queue_action,
                inputs=[state],
                outputs=[queue_html]
            ).then(
                 fn=lambda: (gr.update(visible=False), gr.Accordion(open=False)),
                 inputs=None,
                 outputs=[current_gen_column, queue_accordion]
            )

    if tab_id == 'edit':
        locals_dict = locals()
        gen_inputs = [locals_dict[k] for k in inputs_names] + [state] + extra_inputs
        return gen_inputs
    else:
        return ( state, loras_choices, lset_name, resolution, refresh_form_trigger, save_form_trigger,
                #  video_guide, image_guide, video_mask, image_mask, image_refs,   
                ) 
 

def generate_download_tab(lset_name,loras_choices, state):
    with gr.Row():
        with gr.Row(scale =2):
            gr.Markdown("<I>WanGP's Lora Festival ! Press the following button to download i2v <B>Remade_AI</B> Loras collection (and bonuses Loras).")
        with gr.Row(scale =1):
            download_loras_btn = gr.Button("---> Let the Lora's Festival Start !", scale =1)
        with gr.Row(scale =1):
            gr.Markdown("")
    with gr.Row() as download_status_row: 
        download_status = gr.Markdown()

    download_loras_btn.click(fn=download_loras, inputs=[], outputs=[download_status_row, download_status]).then(fn=refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])

    
def generate_configuration_tab(state, blocks, header, model_family, model_choice, resolution, refresh_form_trigger):
    gr.Markdown("Please click Apply Changes at the bottom so that the changes are effective. Some choices below may be locked if the app has been launched by specifying a config preset.")
    with gr.Column():
        with gr.Tabs():
            # with gr.Row(visible=advanced_ui) as advanced_row:
            with gr.Tab("General"):
                dropdown_families, dropdown_choices = get_sorted_dropdown(displayed_model_types, None)

                transformer_types_choices = gr.Dropdown(
                    choices= dropdown_choices,
                    value= transformer_types,
                    label= "Selectable Generative Models (keep empty to get All of them)",
                    scale= 2,
                    multiselect= True
                    )

                fit_canvas_choice = gr.Dropdown(
                    choices=[
                        ("Dimensions correspond to the Pixels Budget (as the Prompt Image/Video will be Resized to match this pixels Budget, output video height or width may exceed the requested dimensions )", 0),
                        ("Dimensions correspond to the Maximum Width and Height (as the Prompt Image/Video will be Resized to fit into these dimensions, the output video may be smaller)", 1),
                        ("Dimensions correspond to the Output Width and Height (as the Prompt Image/Video will be Cropped to fit exactly these dimensions)", 2),
                    ],
                    value= server_config.get("fit_canvas", 0),
                    label="Generated Video Dimensions when Prompt contains an Image or a Video",
                    interactive= not lock_ui_attention
                 )


                def check(mode): 
                    if not mode in attention_modes_installed:
                        return " (NOT INSTALLED)"
                    elif not mode in attention_modes_supported:
                        return " (NOT SUPPORTED)"
                    else:
                        return ""
                attention_choice = gr.Dropdown(
                    choices=[
                        ("Auto : pick sage2 > sage > sdpa depending on what is installed", "auto"),
                        ("Scale Dot Product Attention: default, always available", "sdpa"),
                        ("Flash" + check("flash")+ ": good quality - requires additional install (usually complex to set up on Windows without WSL)", "flash"),
                        ("Xformers" + check("xformers")+ ": good quality - requires additional install (usually complex, may consume less VRAM to set up on Windows without WSL)", "xformers"),
                        ("Sage" + check("sage")+ ": 30% faster but slightly worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage"),
                        ("Sage2/2++" + check("sage2")+ ": 40% faster but slightly worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage2"),
                        ("Sage3" + check("sage3")+ ": x2 faster but worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage3"),
                    ],
                    value= attention_mode,
                    label="Attention Type",
                    interactive= not lock_ui_attention
                 )


                metadata_choice = gr.Dropdown(
                    choices=[
                        ("Export JSON files", "json"),
                        ("Embed metadata (Exif tag)", "metadata"),
                        ("Neither", "none")
                    ],
                    value=server_config.get("metadata_type", "metadata"),
                    label="Metadata Handling"
                )
                preload_model_policy_choice = gr.CheckboxGroup([("Preload Model while Launching the App","P"), ("Preload Model while Switching Model", "S"), ("Unload Model when Queue is Done", "U")],
                    value=server_config.get("preload_model_policy",[]),
                    label="RAM Loading / Unloading Model Policy (in any case VRAM will be freed once the queue has been processed)"
                )

                clear_file_list_choice = gr.Dropdown(
                    choices=[
                        ("None", 0),
                        ("Keep the last video", 1),
                        ("Keep the last 5 videos", 5),
                        ("Keep the last 10 videos", 10),
                        ("Keep the last 20 videos", 20),
                        ("Keep the last 30 videos", 30),
                    ],
                    value=server_config.get("clear_file_list", 5),
                    label="Keep Previously Generated Videos when starting a new Generation Batch"
                )

                display_stats_choice = gr.Dropdown(
                    choices=[
                        ("Disabled", 0),
                        ("Enabled", 1),
                    ],
                    value=server_config.get("display_stats", 0),
                    label="Display in real time available RAM / VRAM and other stats (needs a restart)"
                )

                max_frames_multiplier_choice = gr.Dropdown(
                    choices=[
                        ("Default", 1),
                        ("x2", 2),
                        ("x3", 3),
                        ("x4", 4),
                        ("x5", 5),
                        ("x6", 7),
                        ("x7", 7),
                    ],
                    value=server_config.get("max_frames_multiplier", 1),
                    label="Increase the Max Number of Frames (needs more RAM and VRAM, usually the longer the worse the quality, needs an App restart)"
                )

                UI_theme_choice = gr.Dropdown(
                    choices=[
                        ("Blue Sky", "default"),
                        ("Classic Gradio", "gradio"),
                    ],
                    value=server_config.get("UI_theme", "default"),
                    label="User Interface Theme. You will need to restart the App the see new Theme."
                )


            with gr.Tab("Performance"):

                quantization_choice = gr.Dropdown(
                    choices=[
                        ("Scaled Int8 Quantization (recommended)", "int8"),
                        ("16 bits (no quantization)", "bf16"),
                    ],
                    value= transformer_quantization,
                    label="Transformer Model Quantization Type (if available)",
                )                

                transformer_dtype_policy_choice = gr.Dropdown(
                    choices=[
                        ("Best Supported Data Type by Hardware", ""),
                        ("FP16", "fp16"),
                        ("BF16", "bf16"),
                    ],
                    value= server_config.get("transformer_dtype_policy", ""),
                    label="Transformer Data Type (if available)"
                )

                mixed_precision_choice = gr.Dropdown(
                    choices=[
                        ("16 bits only, requires less VRAM", "0"),
                        ("Mixed 16 / 32 bits, slightly more VRAM needed but better Quality mainly for 1.3B models", "1"),
                    ],
                    value= server_config.get("mixed_precision", "0"),
                    label="Transformer Engine Calculation"
                )


                text_encoder_quantization_choice = gr.Dropdown(
                    choices=[
                        ("16 bits - unquantized text encoder, better quality uses more RAM", "bf16"),
                        ("8 bits - quantized text encoder, slightly worse quality but uses less RAM", "int8"),
                    ],
                    value= text_encoder_quantization,
                    label="Text Encoder model"
                )

                VAE_precision_choice = gr.Dropdown(
                    choices=[
                        ("16 bits, requires less VRAM and faster", "16"),
                        ("32 bits, requires twice more VRAM and slower but recommended with Window Sliding", "32"),
                    ],
                    value= server_config.get("vae_precision", "16"),
                    label="VAE Encoding / Decoding precision"
                )

                gr.Text("Beware: when restarting the server or changing a resolution or video duration, the first step of generation for a duration / resolution may last a few minutes due to recompilation", interactive= False, show_label= False )
                compile_choice = gr.Dropdown(
                    choices=[
                        ("On (requires to have Triton installed)", "transformer"),
                        ("Off", "" ),
                    ],
                    value= compile,
                    label="Compile Transformer : up to 10-20% faster, useful only if multiple gens at same frames no / resolution",
                    interactive= not lock_ui_compile
                )              

                depth_anything_v2_variant_choice = gr.Dropdown(
                    choices=[
                        ("Large (more precise but 2x slower)", "vitl"),
                        ("Big (less precise, less VRAM needed but faster)", "vitb"),
                    ],
                    value= server_config.get("depth_anything_v2_variant", "vitl"),
                    label="Depth Anything v2 Vace Preprocessor Model type",
                )              

                vae_config_choice = gr.Dropdown(
                    choices=[
                ("Auto", 0),
                ("Disabled (faster but may require up to 22 GB of VRAM)", 1),
                ("256 x 256 : If at least 8 GB of VRAM", 2),
                ("128 x 128 : If at least 6 GB of VRAM", 3),
                    ],
                    value= vae_config,
                    label="VAE Tiling - reduce the high VRAM requirements for VAE decoding and VAE encoding (if enabled it will be slower)"
                )

                boost_choice = gr.Dropdown(
                    choices=[
                        # ("Auto (ON if Video longer than 5s)", 0),
                        ("ON", 1), 
                        ("OFF", 2), 
                    ],
                    value=boost,
                    label="Boost: Give a 10% speedup without losing quality at the cost of a litle VRAM (up to 1GB at max frames and resolution)"
                )

                profile_choice = gr.Dropdown(
                    choices = memory_profile_choices,
                    value= default_profile,
                    label="Profile (for power users only, not needed to change it)"
                )
                preload_in_VRAM_choice = gr.Slider(0, 40000, value=server_config.get("preload_in_VRAM", 0), step=100, label="Number of MB of Models that are Preloaded in VRAM (0 will use Profile default)")

                release_RAM_btn  = gr.Button("Force Release RAM")
                release_RAM_btn.click(fn=release_RAM)

            with gr.Tab("Extensions"):
                enhancer_enabled_choice = gr.Dropdown(
                    choices=[
                        ("Off", 0),
                        ("Florence 2 1.6B + LLava 3.2 3.5B", 1),
                        ("Florence 2 1.6B + LLama Joy Caption (uncensored but needs more VRAM) 9,3B", 2),
                    ],
                    value=server_config.get("enhancer_enabled", 0),
                    label="Prompt Enhancer (if enabled, from 8 GB to 14 GB of extra models will be downloaded)"
                )

                enhancer_mode_choice = gr.Dropdown(
                    choices=[
                        ("Automatically triggered when Generating a Video", 0),
                        ("On Demand Only", 1),
                    ],
                    value=server_config.get("enhancer_mode", 0),
                    label="Prompt Enhancer Usage"
                )

                mmaudio_enabled_choice = gr.Dropdown(
                    choices=[
                        ("Off", 0),
                        ("Turned On but unloaded from RAM after usage", 1),
                        ("Turned On and kept in RAM for fast loading", 2),
                    ],
                    value=server_config.get("mmaudio_enabled", 0),
                    label="MMAudio (if enabled, 10 GB of extra models will be downloaded)"
                )

            with gr.Tab("Outputs"):

                video_output_codec_choice = gr.Dropdown(
                    choices=[
                        ("x265 Balanced Quality (CRF 28)", 'libx265_28'),
                        ("x264 Balanced Quality (Level 8)", 'libx264_8'),
                        ("x265 High Quality (CRF 8)", 'libx265_8'),
                        ("x264 High Quality (Level 10)", 'libx264_10'),
                        ("x264 Lossless", 'libx264_lossless'),
                    ],
                    value=server_config.get("video_output_codec", "libx264_8"),
                    label="Video Codec to use"
                )

                image_output_codec_choice = gr.Dropdown(
                    choices=[
                        ("JPEG Quality 85", 'jpeg_85'),
                        ("WEBP Quality 85", 'webp_85'),
                        ("JPEG Quality 95", 'jpeg_95'),
                        ("WEBP Quality 95", 'webp_95'),
                        ("WEBP Lossless", 'webp_lossless'),
                        ("PNG Lossless", 'png'),
                    ],
                    value=server_config.get("image_output_codec", "jpeg_95"),
                    label="Image Codec to use"
                )

                audio_output_codec_choice = gr.Dropdown(
                    choices=[
                        ("AAC 128 kbit", 'aac_128'),
                    ],
                    value=server_config.get("audio_output_codec", "aac_128"),
                    visible = False,
                    label="Audio Codec to use"
                )

                video_save_path_choice = gr.Textbox(
                    label="Output Folder for Generated Videos (need to restart app to be taken into account)",
                    value=server_config.get("save_path", save_path)
                )

                image_save_path_choice = gr.Textbox(
                    label="Output Folder for Generated Images (need to restart app to be taken into account)",
                    value=server_config.get("image_save_path", image_save_path)
                )


            with gr.Tab("Notifications"):
                gr.Markdown("### Notification Settings")
                notification_sound_enabled_choice = gr.Dropdown(
                    choices=[
                        ("On", 1),
                        ("Off", 0),
                    ],
                    value=server_config.get("notification_sound_enabled", 0),
                    label="Notification Sound Enabled"
                )

                notification_sound_volume_choice = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=server_config.get("notification_sound_volume", 50),
                    step=5,
                    label="Notification Sound Volume (0 = silent, 100 = very loud)"
                )


        
        msg = gr.Markdown() 
        apply_btn  = gr.Button("Apply Changes")
        apply_btn.click(
                fn=apply_changes,
                inputs=[
                    state,
                    transformer_types_choices,
                    transformer_dtype_policy_choice,
                    text_encoder_quantization_choice,
                    VAE_precision_choice,
                    mixed_precision_choice,
                    video_save_path_choice,
                    image_save_path_choice,
                    attention_choice,
                    compile_choice,                            
                    profile_choice,
                    vae_config_choice,
                    metadata_choice,
                    quantization_choice,
                    boost_choice,
                    clear_file_list_choice,
                    preload_model_policy_choice,
                    UI_theme_choice,
                    enhancer_enabled_choice,
                    enhancer_mode_choice,
                    mmaudio_enabled_choice,
                    fit_canvas_choice,
                    preload_in_VRAM_choice,
                    depth_anything_v2_variant_choice,
                    notification_sound_enabled_choice,
                    notification_sound_volume_choice,
                    max_frames_multiplier_choice,
                    display_stats_choice,
                    video_output_codec_choice,
                    image_output_codec_choice,
                    audio_output_codec_choice,
                    resolution,
                ],
                outputs= [msg , header, model_family, model_choice, refresh_form_trigger]
        )

def generate_about_tab():
    gr.Markdown("<H2>WanGP - AI Generative Models for the GPU Poor by <B>DeepBeepMeep</B> (<A HREF='https://github.com/deepbeepmeep/Wan2GP'>GitHub</A>)</H2>")
    gr.Markdown("Many thanks to:")
    gr.Markdown("- <B>Alibaba Wan Team</B> for the best open source video generators (https://github.com/Wan-Video/Wan2.1)")
    gr.Markdown("- <B>Alibaba Vace, Multitalk and Fun Teams</B> for their incredible control net models (https://github.com/ali-vilab/VACE), (https://github.com/MeiGen-AI/MultiTalk) and  (https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP) ")
    gr.Markdown("- <B>Tencent</B> for the impressive Hunyuan Video models (https://github.com/Tencent-Hunyuan/HunyuanVideo)")
    gr.Markdown("- <B>Blackforest Labs</B> for the innovative Flux image generators (https://github.com/black-forest-labs/flux)")
    gr.Markdown("- <B>Alibaba Qwen Team</B> for their state of the art Qwen Image generators (https://github.com/QwenLM/Qwen-Image)")
    gr.Markdown("- <B>Lightricks</B> for their super fast LTX Video models (https://github.com/Lightricks/LTX-Video)")
    gr.Markdown("- <B>Hugging Face</B> for providing hosting for the models and developing must have open source libraries such as Tranformers, Diffusers, Accelerate and Gradio (https://huggingface.co/)")
    gr.Markdown("<BR>Huge acknowledgments to these great open source projects used in WanGP:")
    gr.Markdown("- <B>Rife</B>: temporal upsampler (https://github.com/hzwer/ECCV2022-RIFE)")
    gr.Markdown("- <B>DwPose</B>: Open Pose extractor (https://github.com/IDEA-Research/DWPose)")
    gr.Markdown("- <B>DepthAnything</B> & <B>Midas</B>: Depth extractors (https://github.com/DepthAnything/Depth-Anything-V2) and (https://github.com/isl-org/MiDaS")
    gr.Markdown("- <B>Matanyone</B> and <B>SAM2</B>: Mask Generation (https://github.com/pq-yang/MatAnyone) and (https://github.com/facebookresearch/sam2)")
    gr.Markdown("- <B>Pyannote</B>: speaker diarization (https://github.com/pyannote/pyannote-audio)")

    gr.Markdown("<BR>Special thanks to the following people for their support:")
    gr.Markdown("- <B>Cocktail Peanuts</B> : QA dpand simple installation via Pinokio.computer")
    gr.Markdown("- <B>Tophness</B> : created (former) multi tabs and queuing frameworks")
    gr.Markdown("- <B>AmericanPresidentJimmyCarter</B> : added original support for Skip Layer Guidance")
    gr.Markdown("- <B>Remade_AI</B> : for their awesome Loras collection")
    gr.Markdown("- <B>Reevoy24</B> : for his repackaging / completing the documentation")
    gr.Markdown("- <B>Redtash1</B> : for designing the protype of the RAM / VRAM stats viewer")

def generate_info_tab():


    with open("docs/VACE.md", "r", encoding="utf-8") as reader:
        vace= reader.read()

    with open("docs/MODELS.md", "r", encoding="utf-8") as reader:
        models = reader.read()

    with open("docs/LORAS.md", "r", encoding="utf-8") as reader:
        loras = reader.read()

    with open("docs/FINETUNES.md", "r", encoding="utf-8") as reader:
        finetunes = reader.read()

    with gr.Tabs() :
        with gr.Tab("Models", id="models"):
            gr.Markdown(models)
        with gr.Tab("Loras", id="loras"):
            gr.Markdown(loras)
        with gr.Tab("Vace", id="vace"):
            gr.Markdown(vace)
        with gr.Tab("Finetunes", id="finetunes"):
            gr.Markdown(finetunes)

def compact_name(family_name, model_name):
    if model_name.startswith(family_name):
        return model_name[len(family_name):].strip()
    return model_name

def get_sorted_dropdown(dropdown_types, current_model_family):
    models_families = [get_model_family(type, for_ui= True) for type in dropdown_types] 
    families = {}
    for family in models_families:
        if family not in families: families[family] = 1

    families_orders = [  families_infos[family][0]  for family in families ]
    families_labels = [  families_infos[family][1]  for family in families ]
    sorted_familes = [ info[1:] for info in sorted(zip(families_orders, families_labels, families), key=lambda c: c[0])]
    if current_model_family is None:
        dropdown_choices = [ (families_infos[family][0], get_model_name(model_type), model_type) for model_type, family in zip(dropdown_types, models_families)]
    else:
        dropdown_choices = [ (families_infos[family][0], compact_name(families_infos[family][1], get_model_name(model_type)), model_type) for model_type, family in zip( dropdown_types, models_families) if family == current_model_family]
    dropdown_choices = sorted(dropdown_choices, key=lambda c: (c[0], c[1]))
    dropdown_choices = [model[1:] for model in dropdown_choices] 
    return sorted_familes, dropdown_choices

def generate_dropdown_model_list(current_model_type):
    dropdown_types= transformer_types if len(transformer_types) > 0 else displayed_model_types 
    if current_model_type not in dropdown_types:
        dropdown_types.append(current_model_type)
    current_model_family = get_model_family(current_model_type, for_ui= True)
    sorted_familes, dropdown_choices = get_sorted_dropdown(dropdown_types, current_model_family)

    dropdown_families = gr.Dropdown(
        choices= sorted_familes,
        value= current_model_family,
        show_label= False,
        scale= 1,
        elem_id="family_list",
        min_width=50
        )

    return dropdown_families, gr.Dropdown(
        choices= dropdown_choices,
        value= current_model_type,
        show_label= False,
        scale= 4,
        elem_id="model_list",
        )

def change_model_family(state, current_model_family):
    dropdown_types= transformer_types if len(transformer_types) > 0 else displayed_model_types 
    current_family_name = families_infos[current_model_family][1]
    models_families = [get_model_family(type, for_ui= True) for type in dropdown_types] 
    dropdown_choices = [ (compact_name(current_family_name,  get_model_name(model_type)), model_type) for model_type, family in zip(dropdown_types, models_families) if family == current_model_family ]
    dropdown_choices = sorted(dropdown_choices, key=lambda c: c[0])
    last_model_per_family = state.get("last_model_per_family", {})
    model_type = last_model_per_family.get(current_model_family, "")
    if len(model_type) == "" or model_type not in [choice[1] for choice in dropdown_choices] :  model_type = dropdown_choices[0][1]
    return gr.Dropdown(choices= dropdown_choices, value = model_type )

def set_new_tab(tab_state, new_tab_no):
    global vmc_event_handler    

    tab_video_mask_creator = 2

    old_tab_no = tab_state.get("tab_no",0)
    # print(f"old tab {old_tab_no}, new tab {new_tab_no}")
    if old_tab_no == tab_video_mask_creator:
        vmc_event_handler(False)
    elif new_tab_no == tab_video_mask_creator:
        if gen_in_progress:
            gr.Info("Unable to access this Tab while a Generation is in Progress. Please come back later")
            tab_state["tab_no"] = 0
            return gr.Tabs(selected="video_gen") 
        else:
            vmc_event_handler(True)
    tab_state["tab_no"] = new_tab_no
    return gr.Tabs() 

def select_tab(tab_state, evt:gr.SelectData):
    old_tab_no = tab_state.get("tab_no",0)
    if old_tab_no == 0:
        saveform_trigger = get_unique_id()
    else:
        saveform_trigger = gr.update()
    return set_new_tab(tab_state, evt.index), saveform_trigger

def get_js():
    start_quit_timer_js = """
    () => {
        function findAndClickGradioButton(elemId) {
            const gradioApp = document.querySelector('gradio-app') || document;
            const button = gradioApp.querySelector(`#${elemId}`);
            if (button) { button.click(); }
        }

        if (window.quitCountdownTimeoutId) clearTimeout(window.quitCountdownTimeoutId);

        let js_click_count = 0;
        const max_clicks = 5;

        function countdownStep() {
            if (js_click_count < max_clicks) {
                findAndClickGradioButton('trigger_info_single_btn');
                js_click_count++;
                window.quitCountdownTimeoutId = setTimeout(countdownStep, 1000);
            } else {
                findAndClickGradioButton('force_quit_btn_hidden');
            }
        }

        countdownStep();
    }
    """

    cancel_quit_timer_js = """
    () => {
        if (window.quitCountdownTimeoutId) {
            clearTimeout(window.quitCountdownTimeoutId);
            window.quitCountdownTimeoutId = null;
            console.log("Quit countdown cancelled (single trigger).");
        }
    }
    """

    trigger_zip_download_js = """
    (base64String) => {
        if (!base64String) {
        console.log("No base64 zip data received, skipping download.");
        return;
        }
        try {
        const byteCharacters = atob(base64String);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'application/zip' });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'queue.zip';
        document.body.appendChild(a);
        a.click();

        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        console.log("Zip download triggered.");
        } catch (e) {
        console.error("Error processing base64 data or triggering download:", e);
        }
    }
    """

    trigger_settings_download_js = """
    (base64String, filename) => {
        if (!base64String) {
        console.log("No base64 settings data received, skipping download.");
        return;
        }
        try {
        const byteCharacters = atob(base64String);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'application/text' });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();

        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        console.log("settings download triggered.");
        } catch (e) {
        console.error("Error processing base64 data or triggering download:", e);
        }
    }
    """

    click_brush_js = """
    () => {
        setTimeout(() => {
            const brushButton = document.querySelector('button[aria-label="Brush"]');
            if (brushButton) {
                brushButton.click();
                console.log('Brush button clicked');
            } else {
                console.log('Brush button not found');
            }
        }, 1000);
    }    """

    return start_quit_timer_js, cancel_quit_timer_js, trigger_zip_download_js, trigger_settings_download_js, click_brush_js

def create_ui():
    global vmc_event_handler    
    css = """
        .postprocess div,  
        .postprocess span,    
        .postprocess label,
        .postprocess input,
        .postprocess select,
        .postprocess textarea {
            font-size: 12px !important;
            padding: 0px !important;
            border:  5px !important;
            border-radius: 0px !important;
            --form-gap-width: 0px !important;
            box-shadow: none !important;
            --layout-gap: 0px !important;
        }    
        .postprocess span {margin-top:4px;margin-bottom:4px} 
        #model_list, #family_list{
        background-color:black;
        padding:1px}

        #model_list input, #family_list input {
        font-size:25px}

        #family_list div div {
        border-radius: 4px 0px 0px 4px;
        }

        #model_list div div {
        border-radius: 0px 4px 4px 0px;
        }

        .title-with-lines {
            display: flex;
            align-items: center;
            margin: 25px 0;
        }
        .line {
            flex-grow: 1;
            height: 1px;
            background-color: #333;
        }
        h2 {
            margin: 0 20px;
            white-space: nowrap;
        }
        #queue_html_container table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            table-layout: fixed;
        }
        #queue_html_container th {
            text-align: left;
            padding: 10px 8px;
            border-bottom: 2px solid #4a5568;
            font-weight: bold;
            font-size: 11px;
            text-transform: uppercase;
            color: #a0aec0;
            white-space: nowrap;
        }
        #queue_html_container td {
            padding: 8px;
            border-bottom: 1px solid #2d3748;
            vertical-align: middle;
        }
        #queue_html_container tr:hover td {
            background-color: rgba(255, 255, 255, 0.04);
        }
        #queue_html_container .prompt-cell {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        #queue_html_container .action-button {
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1.3em;
            padding: 0;
            color: #718096;
            transition: color 0.2s;
            line-height: 1;
        }
        #queue_html_container .action-button:hover {
            color: #e2e8f0;
        }
        #queue_html_container .center-align {
            text-align: center;
        }
        #queue_html_container .text-left {
            text-align: left;
        }
        #queue_html_container .hover-image img {
            max-width: 50px;
            max-height: 50px;
            object-fit: contain;
            display: block;
            margin: auto;
        }
        #image-modal-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            justify-content: center;
            align-items: center;
            z-index: 1000;
            padding: 20px;
            box-sizing: border-box;
        }
        #image-modal-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            justify-content: center;
            align-items: center;
            z-index: 1000;
            padding: 20px;
            box-sizing: border-box;
        }
        #image-modal-container > div {
             background-color: white;
             padding: 15px;
             border-radius: 8px;
             max-width: 90%;
             max-height: 90%;
             overflow: auto;
             position: relative;
             display: flex;
             flex-direction: column;
        }
         #image-modal-container img {
             max-width: 100%;
             max-height: 80vh;
             object-fit: contain;
             margin-top: 10px;
         }
         #image-modal-close-button-row {
             display: flex;
             justify-content: flex-end;
         }
         #image-modal-close-button-row button {
            cursor: pointer;
         }
        .progress-container-custom {
            width: 100%;
            background-color: #e9ecef;
            border-radius: 0.375rem;
            overflow: hidden;
            height: 25px;
            position: relative;
            margin-top: 5px;
            margin-bottom: 5px;
        }
        .progress-bar-custom {
            height: 100%;
            background-color: #0d6efd;
            transition: width 0.3s ease-in-out;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.9em;
            font-weight: bold;
            white-space: nowrap;
            overflow: hidden;
        }
        .progress-bar-custom.idle {
            background-color: #6c757d;
        }
        .progress-bar-text {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            mix-blend-mode: difference;
            font-size: 0.9em;
            font-weight: bold;
            white-space: nowrap;
            z-index: 2;
            pointer-events: none;
        }

        .hover-image {
        cursor: pointer;
        position: relative;
        display: inline-block; /* Important for positioning */
        }

        .hover-image .tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 4px 6px;
        border-radius: 2px;
        font-size: 14px;
        white-space: nowrap;
        pointer-events: none;
        z-index: 9999;         
        transition: visibility 0s linear 1s, opacity 0.3s linear 1s; /* Delay both properties */
        }
        div.compact_tab , span.compact_tab 
        { padding: 0px !important;
        } 
        .hover-image .tooltip2 {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            top: 50%; /* Center vertically with the image */
            left: 0; /* Position to the left of the image */
            transform: translateY(-50%); /* Center vertically */
            margin-left: -10px; /* Small gap to the left of image */
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
            white-space: nowrap;
            pointer-events: none;
            z-index: 9999;         
            transition: visibility 0s linear 1s, opacity 0.3s linear 1s;
        }
                
        .hover-image:hover .tooltip, .hover-image:hover .tooltip2 {
        visibility: visible;
        opacity: 1;
        transition: visibility 0s linear 1s, opacity 0.3s linear 1s; /* 1s delay before showing */
        }
        .btn_centered {margin-top:10px; text-wrap-mode: nowrap;}
    """
    UI_theme = server_config.get("UI_theme", "default")
    UI_theme  = args.theme if len(args.theme) > 0 else UI_theme
    if UI_theme == "gradio":
        theme = None
    else:
        theme = gr.themes.Soft(font=["Verdana"], primary_hue="sky", neutral_hue="slate", text_size="md")

    js = """
    function() {
        window.updateAndTrigger = function(action) {
            const hiddenTextbox = document.querySelector('#queue_action_input textarea');
            const hiddenButton = document.querySelector('#queue_action_trigger');
            
            if (hiddenTextbox && hiddenButton) {
                hiddenTextbox.value = action;
                const inputEvent = new Event('input', { bubbles: true });
                hiddenTextbox.dispatchEvent(inputEvent);
                hiddenButton.click();
            } else {
                console.error("Could not find hidden queue action elements.");
            }
        };
        console.log('updateAndTrigger function for queue is ready.');

        window.showImageModal = function(action) {
            const hiddenTextbox = document.querySelector('#modal_action_input textarea');
            const hiddenButton = document.querySelector('#modal_action_trigger');
            if (hiddenTextbox && hiddenButton) {
                hiddenTextbox.value = action;
                hiddenTextbox.dispatchEvent(new Event('input', { bubbles: true }));
                hiddenButton.click();
            } else {
                console.error("Could not find hidden modal action elements.");
            }
        };

        window.closeImageModal = function() {
            const modal = document.querySelector('#image-modal-container');
            if (modal) {
                modal.style.display = 'none';
            }
        };

        // cancel wheel usage inside image editor    
        const hit = n => n?.id === "img_editor" || n?.classList?.contains("wheel-pass");
        addEventListener("wheel", e => {
            const path = e.composedPath?.() || (() => { let a=[],n=e.target; for(;n;n=n.parentNode||n.host) a.push(n); return a; })();
            if (path.some(hit)) e.stopImmediatePropagation();
        }, { capture: true, passive: true });

        }    

    """
    if server_config.get("display_stats", 0) == 1:
        from shared.utils.stats import SystemStatsApp
        stats_app = SystemStatsApp() 
    else:
        stats_app = None

    with gr.Blocks(css=css, js=js,  theme=theme, title= "WanGP") as main:
        gr.Markdown(f"<div align=center><H1>Wan<SUP>GP</SUP> v{WanGP_version} <FONT SIZE=4>by <I>DeepBeepMeep</I></FONT> <FONT SIZE=3>") # (<A HREF='https://github.com/deepbeepmeep/Wan2GP'>Updates</A>)</FONT SIZE=3></H1></div>")
        global model_list

        tab_state = gr.State({ "tab_no":0 }) 

        with gr.Tabs(selected="video_gen", ) as main_tabs:
            with gr.Tab("Video Generator", id="video_gen") as video_generator_tab:
                with gr.Row():
                    if args.lock_model:    
                        gr.Markdown("<div class='title-with-lines'><div class=line></div><h2>" + get_model_name(transformer_type) + "</h2><div class=line></div>")
                        model_family = gr.Dropdown(visible=False, value= "")
                        model_choice = gr.Dropdown(visible=False, value= transformer_type, choices= [transformer_type])
                    else:
                        gr.Markdown("<div class='title-with-lines'><div class=line width=100%></div></div>")
                        model_family, model_choice = generate_dropdown_model_list(transformer_type)
                        gr.Markdown("<div class='title-with-lines'><div class=line width=100%></div></div>")
                with gr.Row():
                    header = gr.Markdown(generate_header(transformer_type, compile, attention_mode), visible= True)
                    if stats_app is not None:
                        stats_element = stats_app.get_gradio_element()

                with gr.Row():
                    (   state, loras_choices, lset_name, resolution, refresh_form_trigger, save_form_trigger
                        # video_guide, image_guide, video_mask, image_mask, image_refs, 
                    ) = generate_video_tab(model_family=model_family, model_choice=model_choice, header=header, main = main, main_tabs =main_tabs, tab_id='generate')
            with gr.Tab("Edit", id="edit") as edit_tab:
                gr.Markdown("<div align=center><H2>Edit Task</H2></div>")
                edit_tab_components = generate_video_tab(
                    update_form=False, 
                    state_dict=state.value, 
                    ui_defaults=get_default_settings(transformer_type), 
                    model_family=model_family, 
                    model_choice=model_choice, 
                    header=header, 
                    main=main, 
                    main_tabs=main_tabs, 
                    tab_id='edit'
                )
                def fill_inputs_for_edit(state):
                    editing_task_index = state.get("editing_task_index", None)
                    if editing_task_index is None:
                        return [gr.update()] * len(edit_tab_components)

                    gen = get_gen_info(state)
                    queue = gen.get("queue", [])
                    task_to_edit_index = editing_task_index + 1
                    
                    if task_to_edit_index >= len(queue):
                        gr.Warning("Task to edit not found in queue.")
                        state["editing_task_index"] = None
                        return [gr.update()] * len(edit_tab_components)

                    task = queue[task_to_edit_index]
                    ui_defaults = task['params']

                    image_list_keys = ['image_start', 'image_end', 'image_refs']
                    for key in image_list_keys:
                        value = ui_defaults.get(key)
                        if value is not None and not isinstance(value, list):
                            ui_defaults[key] = [value]

                    if ui_defaults.get('model_type') != state["model_type"]:
                        gr.Warning(f"Editing a task for a different model ({ui_defaults.get('model_type')}). Some settings may not apply.")

                    return generate_video_tab(update_form=True, state_dict=state, ui_defaults=ui_defaults)
                edit_tab.select(
                    fn=fill_inputs_for_edit,
                    inputs=[state],
                    outputs=edit_tab_components
                )
            with gr.Tab("Guides", id="info") as info_tab:
                generate_info_tab()
            with gr.Tab("Video Mask Creator", id="video_mask_creator") as video_mask_creator:
                matanyone_app.display(main_tabs, tab_state, state, refresh_form_trigger, server_config, get_current_model_settings) #, video_guide, image_guide, video_mask, image_mask, image_refs)
            if not args.lock_config:
                with gr.Tab("Downloads", id="downloads") as downloads_tab:
                    generate_download_tab(lset_name, loras_choices, state)
                with gr.Tab("Configuration", id="configuration") as configuration_tab:
                    generate_configuration_tab(state, main, header, model_family, model_choice, resolution, refresh_form_trigger)
            with gr.Tab("About"):
                generate_about_tab()
        if stats_app is not None:
            stats_app.setup_events(main, state)
        main_tabs.select(fn=select_tab, inputs= [tab_state], outputs= [main_tabs, save_form_trigger], trigger_mode="multiple")
        return main

if __name__ == "__main__":
    atexit.register(autosave_queue)
    download_ffmpeg()
    # threading.Thread(target=runner, daemon=True).start()
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_port = int(args.server_port)
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if server_port == 0:
        server_port = int(os.getenv("SERVER_PORT", "7860"))
    server_name = args.server_name
    if args.listen:
        server_name = "0.0.0.0"
    if len(server_name) == 0:
        server_name = os.getenv("SERVER_NAME", "localhost")      
    demo = create_ui()
    if args.open_browser:
        import webbrowser 
        if server_name.startswith("http"):
            url = server_name 
        else:
            url = "http://" + server_name 
        webbrowser.open(url + ":" + str(server_port), new = 0, autoraise = True)
    demo.launch(favicon_path="favicon.png", server_name=server_name, server_port=server_port, share=args.share, allowed_paths=list({save_path, image_save_path}))
