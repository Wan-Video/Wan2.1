import argparse
import os
import os.path as osp
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import tempfile
import imageio
import torch
import decord
from PIL import Image
import numpy as np
from rembg import remove, new_session
import random
import ffmpeg
import os
import tempfile
import subprocess
import json
from functools import lru_cache
os.environ["U2NET_HOME"] = os.path.join(os.getcwd(), "ckpts", "rembg")


from PIL import Image
video_info_cache = []
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def has_video_file_extension(filename):
    extension = os.path.splitext(filename)[-1].lower()
    return extension in [".mp4"]

def has_image_file_extension(filename):
    extension = os.path.splitext(filename)[-1].lower()
    return extension in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"]

def resample(video_fps, video_frames_count, max_target_frames_count, target_fps, start_target_frame ):
    import math

    video_frame_duration = 1 /video_fps
    target_frame_duration = 1 / target_fps 
    
    target_time = start_target_frame * target_frame_duration
    frame_no = math.ceil(target_time / video_frame_duration)  
    cur_time = frame_no * video_frame_duration
    frame_ids =[]
    while True:
        if max_target_frames_count != 0 and len(frame_ids) >= max_target_frames_count :
            break
        diff = round( (target_time -cur_time) / video_frame_duration , 5)
        add_frames_count = math.ceil( diff)
        frame_no += add_frames_count
        if frame_no >= video_frames_count:             
            break
        frame_ids.append(frame_no)
        cur_time += add_frames_count * video_frame_duration
        target_time += target_frame_duration
    frame_ids = frame_ids[:max_target_frames_count]
    return frame_ids

import os
from datetime import datetime

def get_file_creation_date(file_path):
    # On Windows
    if os.name == 'nt':
        return datetime.fromtimestamp(os.path.getctime(file_path))
    # On Unix/Linux/Mac (gets last status change, not creation)
    else:
        stat = os.stat(file_path)
    return datetime.fromtimestamp(stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_mtime)

def truncate_for_filesystem(s, max_bytes=255):
    if len(s.encode('utf-8')) <= max_bytes: return s
    l, r = 0, len(s)
    while l < r:
        m = (l + r + 1) // 2
        if len(s[:m].encode('utf-8')) <= max_bytes: l = m
        else: r = m - 1
    return s[:l]

@lru_cache(maxsize=100)
def get_video_info(video_path):
    global video_info_cache
    import cv2
    cap = cv2.VideoCapture(video_path)
    
    # Get FPS
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    # Get resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap.release()
    
    return fps, width, height, frame_count

def get_video_frame(file_name: str, frame_no: int, return_last_if_missing: bool = False, target_fps = None,  return_PIL = True) -> torch.Tensor:
    """Extract nth frame from video as PyTorch tensor normalized to [-1, 1]."""
    cap = cv2.VideoCapture(file_name)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {file_name}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if target_fps is not None:
        frame_no = round(target_fps * frame_no /fps)

    # Handle out of bounds
    if frame_no >= total_frames or frame_no < 0:
        if return_last_if_missing:
            frame_no = total_frames - 1
        else:
            cap.release()
            raise IndexError(f"Frame {frame_no} out of bounds (0-{total_frames-1})")
    
    # Get frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read frame {frame_no}")
    
    # Convert BGR->RGB, reshape to (C,H,W), normalize to [-1,1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if return_PIL:
          return Image.fromarray(frame)
    else:
        return (torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5) - 1.0
# def get_video_frame(file_name, frame_no):
#     decord.bridge.set_bridge('torch')
#     reader = decord.VideoReader(file_name)

#     frame = reader.get_batch([frame_no]).squeeze(0)
#     img = Image.fromarray(frame.numpy().astype(np.uint8))
#     return img

def convert_image_to_video(image):
    if image is None:
        return None
    
    # Convert PIL/numpy image to OpenCV format if needed
    if isinstance(image, np.ndarray):
        # Gradio images are typically RGB, OpenCV expects BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # Handle PIL Image
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    height, width = img_bgr.shape[:2]
    
    # Create temporary video file (auto-cleaned by Gradio)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video.name, fourcc, 30.0, (width, height))
        out.write(img_bgr)
        out.release()
        return temp_video.name
    
def resize_lanczos(img, h, w):
    img = (img + 1).float().mul_(127.5)
    img = Image.fromarray(np.clip(img.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8))
    img = img.resize((w,h), resample=Image.Resampling.LANCZOS) 
    img = torch.from_numpy(np.array(img).astype(np.float32)).movedim(-1, 0)
    img = img.div(127.5).sub_(1)
    return img

def remove_background(img, session=None):
    if session ==None:
        session = new_session() 
    img = Image.fromarray(np.clip(255. * img.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8))
    img = remove(img, session=session, alpha_matting = True, bgcolor=[255, 255, 255, 0]).convert('RGB')
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).movedim(-1, 0)


def convert_image_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32)).div_(127.5).sub_(1.).movedim(-1, 0)

def convert_tensor_to_image(t, frame_no = 0, mask_levels = False):
    if len(t.shape) == 4:
        t = t[:, frame_no] 
    if t.shape[0]== 1:
        t = t.expand(3,-1,-1)
    if mask_levels:
        return Image.fromarray(t.clone().mul_(255).permute(1,2,0).to(torch.uint8).cpu().numpy())
    else:
        return Image.fromarray(t.clone().add_(1.).mul_(127.5).permute(1,2,0).to(torch.uint8).cpu().numpy())

def save_image(tensor_image, name, frame_no = -1):
    convert_tensor_to_image(tensor_image, frame_no).save(name)

def get_outpainting_full_area_dimensions(frame_height,frame_width, outpainting_dims):
    outpainting_top, outpainting_bottom, outpainting_left, outpainting_right= outpainting_dims
    frame_height = int(frame_height * (100 + outpainting_top + outpainting_bottom) / 100)
    frame_width =  int(frame_width * (100 + outpainting_left + outpainting_right) / 100)
    return frame_height, frame_width  

def rgb_bw_to_rgba_mask(img, thresh=127):
    a = img.convert('L').point(lambda p: 255 if p > thresh else 0)  # alpha
    out = Image.new('RGBA', img.size, (255, 255, 255, 0))           # white, transparent
    out.putalpha(a)                                                 # white where alpha=255
    return out
                        


def  get_outpainting_frame_location(final_height, final_width,  outpainting_dims, block_size = 8):
    outpainting_top, outpainting_bottom, outpainting_left, outpainting_right= outpainting_dims
    raw_height = int(final_height / ((100 + outpainting_top + outpainting_bottom) / 100))
    height = int(raw_height / block_size) * block_size
    extra_height = raw_height - height
          
    raw_width = int(final_width / ((100 + outpainting_left + outpainting_right) / 100)) 
    width = int(raw_width / block_size) * block_size
    extra_width = raw_width - width  
    margin_top = int(outpainting_top/(100 + outpainting_top + outpainting_bottom) * final_height)
    if extra_height != 0 and (outpainting_top + outpainting_bottom) != 0:
        margin_top += int(outpainting_top / (outpainting_top + outpainting_bottom) * extra_height)
    if (margin_top + height) > final_height or outpainting_bottom == 0: margin_top = final_height - height
    margin_left = int(outpainting_left/(100 + outpainting_left + outpainting_right) * final_width)
    if extra_width != 0 and (outpainting_left + outpainting_right) != 0:
        margin_left += int(outpainting_left / (outpainting_left + outpainting_right) * extra_height)
    if (margin_left + width) > final_width or outpainting_right == 0: margin_left = final_width - width
    return height, width, margin_top, margin_left

def rescale_and_crop(img, w, h):
    ow, oh = img.size
    target_ratio = w / h
    orig_ratio = ow / oh
    
    if orig_ratio > target_ratio:
        # Crop width first
        nw = int(oh * target_ratio)
        img = img.crop(((ow - nw) // 2, 0, (ow + nw) // 2, oh))
    else:
        # Crop height first
        nh = int(ow / target_ratio)
        img = img.crop((0, (oh - nh) // 2, ow, (oh + nh) // 2))
    
    return img.resize((w, h), Image.LANCZOS)

def calculate_new_dimensions(canvas_height, canvas_width, image_height, image_width, fit_into_canvas,  block_size = 16):
    if fit_into_canvas == None or fit_into_canvas == 2:
        # return image_height, image_width
        return canvas_height, canvas_width
    if fit_into_canvas == 1:
        scale1  = min(canvas_height / image_height, canvas_width / image_width)
        scale2  = min(canvas_width / image_height, canvas_height / image_width)
        scale = max(scale1, scale2) 
    else: #0 or #2 (crop)
        scale = (canvas_height * canvas_width / (image_height * image_width))**(1/2)

    new_height = round( image_height * scale / block_size) * block_size
    new_width = round( image_width * scale / block_size) * block_size
    return new_height, new_width

def calculate_dimensions_and_resize_image(image, canvas_height, canvas_width, fit_into_canvas, fit_crop, block_size = 16):
    if fit_crop:
        image = rescale_and_crop(image, canvas_width, canvas_height)
        new_width, new_height = image.size  
    else:
        image_width, image_height = image.size
        new_height, new_width = calculate_new_dimensions(canvas_height, canvas_width, image_height, image_width, fit_into_canvas, block_size = block_size )
        image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS) 
    return image, new_height, new_width

def resize_and_remove_background(img_list, budget_width, budget_height, rm_background, any_background_ref, fit_into_canvas = 0, block_size= 16, outpainting_dims = None, background_ref_outpainted = True, inpaint_color = 127.5, return_tensor = False ):
    if rm_background:
        session = new_session() 

    output_list =[]
    output_mask_list =[]
    for i, img in enumerate(img_list):
        width, height =  img.size 
        resized_mask = None
        if any_background_ref == 1 and i==0 or any_background_ref == 2:
            if outpainting_dims is not None and background_ref_outpainted:
                resized_image, resized_mask = fit_image_into_canvas(img, (budget_height, budget_width), inpaint_color, full_frame = True, outpainting_dims = outpainting_dims, return_mask= True, return_image= True)
            elif img.size != (budget_width, budget_height):
                resized_image= img.resize((budget_width, budget_height), resample=Image.Resampling.LANCZOS) 
            else:
                resized_image =img
        elif fit_into_canvas == 1:
            white_canvas = np.ones((budget_height, budget_width, 3), dtype=np.uint8) * 255 
            scale = min(budget_height / height, budget_width / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            resized_image= img.resize((new_width,new_height), resample=Image.Resampling.LANCZOS) 
            top = (budget_height - new_height) // 2
            left = (budget_width - new_width) // 2
            white_canvas[top:top + new_height, left:left + new_width] = np.array(resized_image)            
            resized_image = Image.fromarray(white_canvas)  
        else:
            scale = (budget_height * budget_width / (height * width))**(1/2)
            new_height = int( round(height * scale / block_size) * block_size)
            new_width = int( round(width * scale / block_size) * block_size)
            resized_image= img.resize((new_width,new_height), resample=Image.Resampling.LANCZOS) 
        if rm_background  and not (any_background_ref and i==0 or any_background_ref == 2) :
            # resized_image = remove(resized_image, session=session, alpha_matting_erode_size = 1,alpha_matting_background_threshold = 70, alpha_foreground_background_threshold = 100, alpha_matting = True, bgcolor=[255, 255, 255, 0]).convert('RGB')
            resized_image = remove(resized_image, session=session, alpha_matting_erode_size = 1, alpha_matting = True, bgcolor=[255, 255, 255, 0]).convert('RGB')
        if return_tensor:
            output_list.append(convert_image_to_tensor(resized_image).unsqueeze(1)) 
        else:
            output_list.append(resized_image) 
        output_mask_list.append(resized_mask)
    return output_list, output_mask_list

def fit_image_into_canvas(ref_img, image_size, canvas_tf_bg =127.5, device ="cpu", full_frame = False, outpainting_dims = None, return_mask = False, return_image = False):
    from shared.utils.utils import save_image
    inpaint_color = canvas_tf_bg / 127.5 - 1

    ref_width, ref_height = ref_img.size
    if (ref_height, ref_width) == image_size and outpainting_dims  == None:
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
        canvas = torch.zeros_like(ref_img[:1]) if return_mask else None
    else:
        if outpainting_dims != None:
            final_height, final_width = image_size
            canvas_height, canvas_width, margin_top, margin_left =   get_outpainting_frame_location(final_height, final_width,  outpainting_dims, 1)        
        else:
            canvas_height, canvas_width = image_size
        if full_frame:
            new_height = canvas_height
            new_width = canvas_width
            top = left = 0 
        else:
            # if fill_max  and (canvas_height - new_height) < 16:
            #     new_height = canvas_height
            # if fill_max  and (canvas_width - new_width) < 16:
            #     new_width = canvas_width
            scale = min(canvas_height / ref_height, canvas_width / ref_width)
            new_height = int(ref_height * scale)
            new_width = int(ref_width * scale)
            top = (canvas_height - new_height) // 2
            left = (canvas_width - new_width) // 2
        ref_img = ref_img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS) 
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
        if outpainting_dims != None:
            canvas = torch.full((3, 1, final_height, final_width), inpaint_color, dtype= torch.float, device=device) # [-1, 1]
            canvas[:, :, margin_top + top:margin_top + top + new_height, margin_left + left:margin_left + left + new_width] = ref_img 
        else:
            canvas = torch.full((3, 1, canvas_height, canvas_width), inpaint_color, dtype= torch.float, device=device) # [-1, 1]
            canvas[:, :, top:top + new_height, left:left + new_width] = ref_img 
        ref_img = canvas
        canvas = None
        if return_mask:
            if outpainting_dims != None:
                canvas = torch.ones((1, 1, final_height, final_width), dtype= torch.float, device=device) # [-1, 1]
                canvas[:, :, margin_top + top:margin_top + top + new_height, margin_left + left:margin_left + left + new_width] = 0
            else:
                canvas = torch.ones((1, 1, canvas_height, canvas_width), dtype= torch.float, device=device) # [-1, 1]
                canvas[:, :, top:top + new_height, left:left + new_width] = 0
            canvas = canvas.to(device)
    if return_image:
        return convert_tensor_to_image(ref_img), canvas

    return ref_img.to(device), canvas

def prepare_video_guide_and_mask( video_guides, video_masks, pre_video_guide, image_size, current_video_length = 81, latent_size = 4, any_mask = False, any_guide_padding = False, guide_inpaint_color = 127.5, keep_video_guide_frames = [],  inject_frames = [], outpainting_dims = None, device ="cpu"):
    src_videos, src_masks = [], []
    inpaint_color_compressed = guide_inpaint_color/127.5 - 1
    prepend_count = pre_video_guide.shape[1] if pre_video_guide is not None else 0
    for guide_no, (cur_video_guide, cur_video_mask) in enumerate(zip(video_guides, video_masks)):
        src_video, src_mask = cur_video_guide, cur_video_mask
        if pre_video_guide is not None:
            src_video = pre_video_guide if src_video is None else torch.cat( [pre_video_guide, src_video], dim=1)
            if any_mask:
                src_mask = torch.zeros_like(pre_video_guide[:1]) if src_mask is None else torch.cat( [torch.zeros_like(pre_video_guide[:1]), src_mask], dim=1)

        if any_guide_padding:
            if src_video is None:
                src_video = torch.full( (3, current_video_length, *image_size ), inpaint_color_compressed, dtype = torch.float, device= device)
            elif src_video.shape[1] < current_video_length:
                src_video = torch.cat([src_video, torch.full( (3, current_video_length - src_video.shape[1], *src_video.shape[-2:]  ), inpaint_color_compressed, dtype = src_video.dtype, device= src_video.device) ], dim=1)
        elif src_video is not None:
            new_num_frames = (src_video.shape[1] - 1) // latent_size * latent_size + 1 
            src_video = src_video[:, :new_num_frames]

        if any_mask and src_video is not None:
            if src_mask is None:                   
                src_mask = torch.ones_like(src_video[:1])
            elif src_mask.shape[1] < src_video.shape[1]:
                src_mask = torch.cat([src_mask, torch.full( (1, src_video.shape[1]- src_mask.shape[1], *src_mask.shape[-2:]  ), 1, dtype = src_video.dtype, device= src_video.device) ], dim=1)
            else:
                src_mask = src_mask[:, :src_video.shape[1]]                                        

        if src_video is not None :
            for k, keep in enumerate(keep_video_guide_frames):
                if not keep:
                    pos = prepend_count + k
                    src_video[:, pos:pos+1] = inpaint_color_compressed
                    if any_mask: src_mask[:, pos:pos+1] = 1

            for k, frame in enumerate(inject_frames):
                if frame != None:
                    pos = prepend_count + k
                    src_video[:, pos:pos+1], msk = fit_image_into_canvas(frame, image_size, guide_inpaint_color, device, True, outpainting_dims, return_mask= any_mask)
                    if any_mask: src_mask[:, pos:pos+1] = msk
        src_videos.append(src_video)
        src_masks.append(src_mask)
    return src_videos, src_masks


