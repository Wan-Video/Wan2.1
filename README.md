# Wan2.1

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.1">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">Technical Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">Blog</a> &nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat Group</a>&nbsp&nbsp | &nbsp&nbsp 📖 <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>&nbsp&nbsp
<br>

-----

[**Wan: Open and Advanced Large-Scale Video Generative Models**](https://arxiv.org/abs/2503.20314) <be>

In this repository, we present **Wan2.1**, a comprehensive and open suite of video foundation models that pushes the boundaries of video generation. **Wan2.1** offers these key features:
- 👍 **SOTA Performance**: **Wan2.1** consistently outperforms existing open-source models and state-of-the-art commercial solutions across multiple benchmarks.
- 👍 **Supports Consumer-grade GPUs**: The T2V-1.3B model requires only 8.19 GB VRAM, making it compatible with almost all consumer-grade GPUs. It can generate a 5-second 480P video on an RTX 4090 in about 4 minutes (without optimization techniques like quantization). Its performance is even comparable to some closed-source models.
- 👍 **Multiple Tasks**: **Wan2.1** excels in Text-to-Video, Image-to-Video, Video Editing, Text-to-Image, and Video-to-Audio, advancing the field of video generation.
- 👍 **Visual Text Generation**: **Wan2.1** is the first video model capable of generating both Chinese and English text, featuring robust text generation that enhances its practical applications.
- 👍 **Powerful Video VAE**: **Wan-VAE** delivers exceptional efficiency and performance, encoding and decoding 1080P videos of any length while preserving temporal information, making it an ideal foundation for video and image generation.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/4aca6063-60bf-4953-bfb7-e265053f49ef" width="70%" poster=""> </video>
</div>

## 🔥 Latest News!!

* May 14, 2025: 👋 We introduce **Wan2.1** [VACE](https://github.com/ali-vilab/VACE), an all-in-one model for video creation and editing, along with its [inference code](#run-vace), [weights](#model-download), and [technical report](https://arxiv.org/abs/2503.07598)!
* Apr 17, 2025: 👋 We introduce **Wan2.1** [FLF2V](#run-first-last-frame-to-video-generation) with its inference code and weights!
* Mar 21, 2025: 👋 We are excited to announce the release of the **Wan2.1** [technical report](https://files.alicdn.com/tpsservice/5c9de1c74de03972b7aa657e5a54756b.pdf). We welcome discussions and feedback!
* Mar 3, 2025: 👋 **Wan2.1**'s T2V and I2V have been integrated into Diffusers ([T2V](https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan#diffusers.WanPipeline) | [I2V](https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan#diffusers.WanImageToVideoPipeline)). Feel free to give it a try!
* Feb 27, 2025: 👋 **Wan2.1** has been integrated into [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/wan/). Enjoy!
* Feb 25, 2025: 👋 We've released the inference code and weights of **Wan2.1**.

## Community Works
If your work has improved **Wan2.1** and you would like more people to see it, please inform us.
- [LightX2V](https://github.com/ModelTC/LightX2V), a lightweight and efficient video generation framework that integrates **Wan2.1** and **Wan2.2**, supporting multiple engineering acceleration techniques for fast inference. [LightX2V-HuggingFace](https://huggingface.co/lightx2v), offers a variety of Wan-based step-distillation models, quantized models, and lightweight VAE models. Combined with step-distillation models, LightX2V running with multi-GPU parallelism on RTX 5090 can generate a 5-second video in under 5 seconds. With offloading techniques, LightX2V can also run inference on an RTX 4060 (8GB VRAM).
- [DriVerse](https://github.com/shalfun/DriVerse), an autonomous driving world model based on **Wan2.1-14B-I2V**, generates future driving videos conditioned on any scene frame and given trajectory. Refer to the [project page](https://github.com/shalfun/DriVerse/tree/main) for more examples.
- [Training-Free-WAN-Editing](https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing), built on **Wan2.1-T2V-1.3B**, allows training-free video editing with image-based training-free methods, such as [FlowEdit](https://arxiv.org/abs/2412.08629) and [FlowAlign](https://arxiv.org/abs/2505.23145).
- [Wan-Move](https://github.com/ali-vilab/Wan-Move), accepted to NeurIPS 2025, a framework that brings **Wan2.1-I2V-14B** to SOTA fine-grained, point-level motion control! Refer to [their project page](https://wan-move.github.io/) for more information.
- [EchoShot](https://github.com/JoHnneyWang/EchoShot), a native multi-shot portrait video generation model based on **Wan2.1-T2V-1.3B**, allows generation of multiple video clips featuring the same character as well as highly flexible content controllability. Refer to [their project page](https://johnneywang.github.io/EchoShot-webpage/) for more information.
- [AniCrafter](https://github.com/MyNiuuu/AniCrafter), a human-centric animation model based on **Wan2.1-14B-I2V**, controls the Video Diffusion Models with 3DGS Avatars to insert and animate anyone into any scene following given motion sequences. Refer to the [project page](https://myniuuu.github.io/AniCrafter) for more examples.
- [HyperMotion](https://vivocameraresearch.github.io/hypermotion/), a human image animation framework based on **Wan2.1**, addresses the challenge of generating complex human body motions in pose-guided animation. Refer to [their website](https://vivocameraresearch.github.io/magictryon/) for more examples.
- [MagicTryOn](https://vivocameraresearch.github.io/magictryon/), a video virtual try-on framework built upon **Wan2.1-14B-I2V**, addresses the limitations of existing models in expressing garment details and maintaining dynamic stability during human motion. Refer to [their website](https://vivocameraresearch.github.io/magictryon/) for more examples.
- [ATI](https://github.com/bytedance/ATI), built on **Wan2.1-I2V-14B**, is a trajectory-based motion-control framework that unifies object, local, and camera movements in video generation. Refer to [their website](https://anytraj.github.io/) for more examples.
- [Phantom](https://github.com/Phantom-video/Phantom) has developed a unified video generation framework for single and multi-subject references based on both **Wan2.1-T2V-1.3B** and **Wan2.1-T2V-14B**. Please refer to [their examples](https://github.com/Phantom-video/Phantom).
- [UniAnimate-DiT](https://github.com/ali-vilab/UniAnimate-DiT), based on **Wan2.1-14B-I2V**, has trained a Human image animation model and has open-sourced the inference and training code. Feel free to enjoy it!
- [CFG-Zero](https://github.com/WeichenFan/CFG-Zero-star) enhances **Wan2.1** (covering both T2V and I2V models) from the perspective of CFG.
- [TeaCache](https://github.com/ali-vilab/TeaCache) now supports **Wan2.1** acceleration, capable of increasing speed by approximately 2x. Feel free to give it a try!
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) provides more support for **Wan2.1**, including video-to-video, FP8 quantization, VRAM optimization, LoRA training, and more. Please refer to [their examples](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo).


## 📑 Todo List
- Wan2.1 Text-to-Video
    - [x] Multi-GPU Inference code of the 14B and 1.3B models
    - [x] Checkpoints of the 14B and 1.3B models
    - [x] Gradio demo
    - [x] ComfyUI integration
    - [x] Diffusers integration
    - [ ] Diffusers + Multi-GPU Inference
- Wan2.1 Image-to-Video
    - [x] Multi-GPU Inference code of the 14B model
    - [x] Checkpoints of the 14B model
    - [x] Gradio demo
    - [x] ComfyUI integration
    - [x] Diffusers integration
    - [ ] Diffusers + Multi-GPU Inference
- Wan2.1 First-Last-Frame-to-Video
    - [x] Multi-GPU Inference code of the 14B model
    - [x] Checkpoints of the 14B model
    - [x] Gradio demo
    - [ ] ComfyUI integration
    - [ ] Diffusers integration
    - [ ] Diffusers + Multi-GPU Inference
- Wan2.1 VACE
    - [x] Multi-GPU Inference code of the 14B and 1.3B models
    - [x] Checkpoints of the 14B and 1.3B models
    - [x] Gradio demo
    - [x] ComfyUI integration
    - [ ] Diffusers integration
    - [ ] Diffusers + Multi-GPU Inference

## Quickstart

#### Installation
Clone the repo:
```sh
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1
```

Install dependencies:
```sh
# Ensure torch >= 2.4.0
pip install -r requirements.txt
```


#### Model Download

| Models       | Download Link                                                                                                                                           |    Notes                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| T2V-14B      | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)      🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)             | Supports both 480P and 720P
| I2V-14B-720P | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)     | Supports 720P
| I2V-14B-480P | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)     | Supports 480P
| T2V-1.3B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)     🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)            | Supports 480P
| FLF2V-14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P)     🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-FLF2V-14B-720P) | Supports 720P
| VACE-1.3B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B)     🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B)          | Supports 480P
| VACE-14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)     🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B)        | Supports both 480P and 720P

> 💡Note: 
> * The 1.3B model is capable of generating videos at 720P resolution. However, due to limited training at this resolution, the results are generally less stable compared to 480P. For optimal performance, we recommend using 480P resolution. 
> * For the first-last frame to video generation, we train our model primarily on Chinese text-video pairs. Therefore, we recommend using Chinese prompt to achieve better results.


Download models using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./Wan2.1-T2V-14B
```
#### Run Text-to-Video Generation

This repository supports two Text-to-Video models (1.3B and 14B) and two resolutions (480P and 720P). The parameters and configurations for these models are as follows:

<table>
    <thead>
        <tr>
            <th rowspan="2">Task</th>
            <th colspan="2">Resolution</th>
            <th rowspan="2">Model</th>
        </tr>
        <tr>
            <th>480P</th>
            <th>720P</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>t2v-14B</td>
            <td style="color: green;">✔️</td>
            <td style="color: green;">✔️</td>
            <td>Wan2.1-T2V-14B</td>
        </tr>
        <tr>
            <td>t2v-1.3B</td>
            <td style="color: green;">✔️</td>
            <td style="color: red;">❌</td>
            <td>Wan2.1-T2V-1.3B</td>
        </tr>
    </tbody>
</table>


##### (1) Without Prompt Extension

To facilitate implementation, we will start with a basic version of the inference process that skips the [prompt extension](#2-using-prompt-extention) step.

- Single-GPU inference

``` sh
python generate.py  --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

If you encounter OOM (Out-of-Memory) issues, you can use the `--offload_model True` and `--t5_cpu` options to reduce GPU memory usage. For example, on an RTX 4090 GPU:

``` sh
python generate.py  --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> 💡Note: If you are using the `T2V-1.3B` model, we recommend setting the parameter `--sample_guide_scale 6`. The `--sample_shift parameter` can be adjusted within the range of 8 to 12 based on the performance.


- Multi-GPU inference using FSDP + xDiT USP

  We use FSDP and [xDiT](https://github.com/xdit-project/xDiT) USP to accelerate  inference.

  * Ulysess Strategy

    If you want to use [`Ulysses`](https://arxiv.org/abs/2309.14509) strategy, you should set `--ulysses_size $GPU_NUMS`. Note that the `num_heads` should be divisible by `ulysses_size` if you wish to use `Ulysess` strategy. For the 1.3B model, the `num_heads` is `12` which can't be divided by 8 (as most multi-GPU machines have 8 GPUs). Therefore, it is recommended to use `Ring Strategy` instead.

  * Ring Strategy

    If you want to use [`Ring`](https://arxiv.org/pdf/2310.01889) strategy, you should set `--ring_size $GPU_NUMS`. Note that the `sequence length` should be divisible by `ring_size` when using the `Ring` strategy.

  Of course, you can also combine the use of `Ulysses` and `Ring` strategies.


``` sh
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=8 generate.py --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```


##### (2) Using Prompt Extension

Extending the prompts can effectively enrich the details in the generated videos, further enhancing the video quality. Therefore, we recommend enabling prompt extension. We provide the following two methods for prompt extension:

- Use the Dashscope API for extension.
  - Apply for a `dashscope.api_key` in advance ([EN](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen) | [CN](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen)).
  - Configure the environment variable `DASH_API_KEY` to specify the Dashscope API key. For users of Alibaba Cloud's international site, you also need to set the environment variable `DASH_API_URL` to 'https://dashscope-intl.aliyuncs.com/api/v1'. For more detailed instructions, please refer to the [dashscope document](https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-qwen-by-calling-api?spm=a2c63.p38356.0.i1).
  - Use the `qwen-plus` model for text-to-video tasks and `qwen-vl-max` for image-to-video tasks.
  - You can modify the model used for extension with the parameter `--prompt_extend_model`. For example:
```sh
DASH_API_KEY=your_key python generate.py  --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
```

- Using a local model for extension.

  - By default, the Qwen model on HuggingFace is used for this extension. Users can choose Qwen models or other models based on the available GPU memory size.
  - For text-to-video tasks, you can use models like `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`.
  - For image-to-video or first-last-frame-to-video tasks, you can use models like `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2.5-VL-3B-Instruct`.
  - Larger models generally provide better extension results but require more GPU memory.
  - You can modify the model used for extension with the parameter `--prompt_extend_model` , allowing you to specify either a local model path or a Hugging Face model. For example:

``` sh
python generate.py  --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
```


##### (3) Running with Diffusers

You can easily inference **Wan2.1**-T2V using Diffusers with the following command:
``` python
import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = scheduler
pipe.to("cuda")

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
     prompt=prompt,
     negative_prompt=negative_prompt,
     height=720,
     width=1280,
     num_frames=81,
     guidance_scale=5.0,
    ).frames[0]
export_to_video(output, "output.mp4", fps=16)
```
> 💡Note: Please note that this example does not integrate Prompt Extension and distributed inference. We will soon update with the integrated prompt extension and multi-GPU version of Diffusers.


##### (4) Running local gradio

``` sh
cd gradio
# if one uses dashscope’s API for prompt extension
DASH_API_KEY=your_key python t2v_14B_singleGPU.py --prompt_extend_method 'dashscope' --ckpt_dir ./Wan2.1-T2V-14B

# if one uses a local model for prompt extension
python t2v_14B_singleGPU.py --prompt_extend_method 'local_qwen' --ckpt_dir ./Wan2.1-T2V-14B
```



#### Run Image-to-Video Generation

Similar to Text-to-Video, Image-to-Video is also divided into processes with and without the prompt extension step. The specific parameters and their corresponding settings are as follows:
<table>
    <thead>
        <tr>
            <th rowspan="2">Task</th>
            <th colspan="2">Resolution</th>
            <th rowspan="2">Model</th>
        </tr>
        <tr>
            <th>480P</th>
            <th>720P</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>i2v-14B</td>
            <td style="color: green;">❌</td>
            <td style="color: green;">✔️</td>
            <td>Wan2.1-I2V-14B-720P</td>
        </tr>
        <tr>
            <td>i2v-14B</td>
            <td style="color: green;">✔️</td>
            <td style="color: red;">❌</td>
            <td>Wan2.1-T2V-14B-480P</td>
        </tr>
    </tbody>
</table>


##### (1) Without Prompt Extension

- Single-GPU inference
```sh
python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> 💡For the Image-to-Video task, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.


- Multi-GPU inference using FSDP + xDiT USP

```sh
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=8 generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

##### (2) Using Prompt Extension


The process of prompt extension can be referenced [here](#2-using-prompt-extention).

Run with local prompt extension using `Qwen/Qwen2.5-VL-7B-Instruct`:
```
python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG --use_prompt_extend --prompt_extend_model Qwen/Qwen2.5-VL-7B-Instruct --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

Run with remote prompt extension using `dashscope`:
```
DASH_API_KEY=your_key python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG --use_prompt_extend --prompt_extend_method 'dashscope' --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```


##### (3) Running with Diffusers

You can easily inference **Wan2.1**-I2V using Diffusers with the following command:
``` python
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

# Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))
prompt = (
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height, width=width,
    num_frames=81,
    guidance_scale=5.0
).frames[0]
export_to_video(output, "output.mp4", fps=16)

```
> 💡Note: Please note that this example does not integrate Prompt Extension and distributed inference. We will soon update with the integrated prompt extension and multi-GPU version of Diffusers.


##### (4) Running local gradio

```sh
cd gradio
# if one only uses 480P model in gradio
DASH_API_KEY=your_key python i2v_14B_singleGPU.py --prompt_extend_method 'dashscope' --ckpt_dir_480p ./Wan2.1-I2V-14B-480P

# if one only uses 720P model in gradio
DASH_API_KEY=your_key python i2v_14B_singleGPU.py --prompt_extend_method 'dashscope' --ckpt_dir_720p ./Wan2.1-I2V-14B-720P

# if one uses both 480P and 720P models in gradio
DASH_API_KEY=your_key python i2v_14B_singleGPU.py --prompt_extend_method 'dashscope' --ckpt_dir_480p ./Wan2.1-I2V-14B-480P --ckpt_dir_720p ./Wan2.1-I2V-14B-720P
```


#### Run First-Last-Frame-to-Video Generation

First-Last-Frame-to-Video is also divided into processes with and without the prompt extension step. Currently, only 720P is supported. The specific parameters and corresponding settings are as follows:
<table>
    <thead>
        <tr>
            <th rowspan="2">Task</th>
            <th colspan="2">Resolution</th>
            <th rowspan="2">Model</th>
        </tr>
        <tr>
            <th>480P</th>
            <th>720P</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>flf2v-14B</td>
            <td style="color: green;">❌</td>
            <td style="color: green;">✔️</td>
            <td>Wan2.1-FLF2V-14B-720P</td>
        </tr>
    </tbody>
</table>


##### (1) Without Prompt Extension

- Single-GPU inference
```sh
python generate.py --task flf2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-FLF2V-14B-720P --first_frame examples/flf2v_input_first_frame.png --last_frame examples/flf2v_input_last_frame.png --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird’s feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
```

> 💡Similar to Image-to-Video, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.


- Multi-GPU inference using FSDP + xDiT USP

```sh
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=8 generate.py --task flf2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-FLF2V-14B-720P --first_frame examples/flf2v_input_first_frame.png --last_frame examples/flf2v_input_last_frame.png --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird’s feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
```

##### (2) Using Prompt Extension


The process of prompt extension can be referenced [here](#2-using-prompt-extention).

Run with local prompt extension using `Qwen/Qwen2.5-VL-7B-Instruct`:
```
python generate.py --task flf2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-FLF2V-14B-720P --first_frame examples/flf2v_input_first_frame.png --last_frame examples/flf2v_input_last_frame.png --use_prompt_extend --prompt_extend_model Qwen/Qwen2.5-VL-7B-Instruct --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird’s feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
```

Run with remote prompt extension using `dashscope`:
```
DASH_API_KEY=your_key python generate.py --task flf2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-FLF2V-14B-720P --first_frame examples/flf2v_input_first_frame.png --last_frame examples/flf2v_input_last_frame.png --use_prompt_extend --prompt_extend_method 'dashscope' --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird’s feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
```


##### (3) Running local gradio

```sh
cd gradio
# use 720P model in gradio
DASH_API_KEY=your_key python flf2v_14B_singleGPU.py --prompt_extend_method 'dashscope' --ckpt_dir_720p ./Wan2.1-FLF2V-14B-720P
```


#### Run VACE

[VACE](https://github.com/ali-vilab/VACE) now supports two models (1.3B and 14B) and two main resolutions (480P and 720P). 
The input supports any resolution, but to achieve optimal results, the video size should fall within a specific range.
The parameters and configurations for these models are as follows:

<table>
    <thead>
        <tr>
            <th rowspan="2">Task</th>
            <th colspan="2">Resolution</th>
            <th rowspan="2">Model</th>
        </tr>
        <tr>
            <th>480P(~81x480x832)</th>
            <th>720P(~81x720x1280)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>VACE</td>
            <td style="color: green; text-align: center; vertical-align: middle;">✔️</td>
            <td style="color: green; text-align: center; vertical-align: middle;">✔️</td>
            <td>Wan2.1-VACE-14B</td>
        </tr>
        <tr>
            <td>VACE</td>
            <td style="color: green; text-align: center; vertical-align: middle;">✔️</td>
            <td style="color: red; text-align: center; vertical-align: middle;">❌</td>
            <td>Wan2.1-VACE-1.3B</td>
        </tr>
    </tbody>
</table>

In VACE, users can input text prompt and optional video, mask, and image for video generation or editing. Detailed instructions for using VACE can be found in the [User Guide](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md).
The execution process is as follows:

##### (1) Preprocessing

User-collected materials needs to be preprocessed into VACE-recognizable inputs, including `src_video`, `src_mask`, `src_ref_images`, and `prompt`.
For R2V (Reference-to-Video Generation), you may skip this preprocessing, but for V2V (Video-to-Video Editing) and MV2V (Masked Video-to-Video Editing) tasks, additional preprocessing is required to obtain video with conditions such as depth, pose or masked regions.
For more details, please refer to [vace_preproccess](https://github.com/ali-vilab/VACE/blob/main/vace/vace_preproccess.py).

##### (2) cli inference

- Single-GPU inference
```sh
python generate.py --task vace-1.3B --size 832*480 --ckpt_dir ./Wan2.1-VACE-1.3B --src_ref_images examples/girl.png,examples/snake.png --prompt "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
```

- Multi-GPU inference using FSDP + xDiT USP

```sh
torchrun --nproc_per_node=8 generate.py --task vace-14B --size 1280*720 --ckpt_dir ./Wan2.1-VACE-14B --dit_fsdp --t5_fsdp --ulysses_size 8 --src_ref_images examples/girl.png,examples/snake.png --prompt "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
```

##### (3) Running local gradio
- Single-GPU inference
```sh
python gradio/vace.py --ckpt_dir ./Wan2.1-VACE-1.3B
```

- Multi-GPU inference using FSDP + xDiT USP
```sh
python gradio/vace.py --mp --ulysses_size 8 --ckpt_dir ./Wan2.1-VACE-14B/
```

#### Run Text-to-Image Generation

Wan2.1 is a unified model for both image and video generation. Since it was trained on both types of data, it can also generate images. The command for generating images is similar to video generation, as follows:

##### (1) Without Prompt Extension

- Single-GPU inference
```sh
python generate.py --task t2i-14B --size 1024*1024 --ckpt_dir ./Wan2.1-T2V-14B  --prompt '一个朴素端庄的美人'
```

- Multi-GPU inference using FSDP + xDiT USP

```sh
torchrun --nproc_per_node=8 generate.py --dit_fsdp --t5_fsdp --ulysses_size 8 --base_seed 0 --frame_num 1 --task t2i-14B  --size 1024*1024 --prompt '一个朴素端庄的美人' --ckpt_dir ./Wan2.1-T2V-14B
```

##### (2) With Prompt Extention

- Single-GPU inference
```sh
python generate.py --task t2i-14B --size 1024*1024 --ckpt_dir ./Wan2.1-T2V-14B  --prompt '一个朴素端庄的美人' --use_prompt_extend
```

- Multi-GPU inference using FSDP + xDiT USP
```sh
torchrun --nproc_per_node=8 generate.py --dit_fsdp --t5_fsdp --ulysses_size 8 --base_seed 0 --frame_num 1 --task t2i-14B  --size 1024*1024 --ckpt_dir ./Wan2.1-T2V-14B --prompt '一个朴素端庄的美人' --use_prompt_extend
```


## Manual Evaluation

##### (1) Text-to-Video Evaluation

Through manual evaluation, the results generated after prompt extension are superior to those from both closed-source and open-source models.

<div align="center">
    <img src="assets/t2v_res.jpg" alt="" style="width: 80%;" />
</div>


##### (2) Image-to-Video Evaluation

We also conducted extensive manual evaluations to evaluate the performance of the Image-to-Video model, and the results are presented in the table below. The results clearly indicate that **Wan2.1** outperforms both closed-source and open-source models.

<div align="center">
    <img src="assets/i2v_res.png" alt="" style="width: 80%;" />
</div>


## Computational Efficiency on Different GPUs

We test the computational efficiency of different **Wan2.1** models on different GPUs in the following table. The results are presented in the format: **Total time (s) / peak GPU memory (GB)**.


<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

> The parameter settings for the tests presented in this table are as follows:
> (1) For the 1.3B model on 8 GPUs, set `--ring_size 8` and `--ulysses_size 1`;
> (2) For the 14B model on 1 GPU, use `--offload_model True`;
> (3) For the 1.3B model on a single 4090 GPU, set `--offload_model True --t5_cpu`;
> (4) For all testings, no prompt extension was applied, meaning `--use_prompt_extend` was not enabled.

> 💡Note: T2V-14B is slower than I2V-14B because the former samples 50 steps while the latter uses 40 steps.


-------

## Introduction of Wan2.1

**Wan2.1**  is designed on the mainstream diffusion transformer paradigm, achieving significant advancements in generative capabilities through a series of innovations. These include our novel spatio-temporal variational autoencoder (VAE), scalable training strategies, large-scale data construction, and automated evaluation metrics. Collectively, these contributions enhance the model’s performance and versatility.


##### (1) 3D Variational Autoencoders
We propose a novel 3D causal VAE architecture, termed **Wan-VAE** specifically designed for video generation. By combining multiple strategies, we improve spatio-temporal compression, reduce memory usage, and ensure temporal causality. **Wan-VAE** demonstrates significant advantages in performance efficiency compared to other open-source VAEs. Furthermore, our **Wan-VAE** can encode and decode unlimited-length 1080P videos without losing historical temporal information, making it particularly well-suited for video generation tasks.


<div align="center">
    <img src="assets/video_vae_res.jpg" alt="" style="width: 80%;" />
</div>


##### (2) Video Diffusion DiT

**Wan2.1** is designed using the Flow Matching framework within the paradigm of mainstream Diffusion Transformers. Our model's architecture uses the T5 Encoder to encode multilingual text input, with cross-attention in each transformer block embedding the text into the model structure. Additionally, we employ an MLP with a Linear layer and a SiLU layer to process the input time embeddings and predict six modulation parameters individually. This MLP is shared across all transformer blocks, with each block learning a distinct set of biases. Our experimental findings reveal a significant performance improvement with this approach at the same parameter scale.

<div align="center">
    <img src="assets/video_dit_arch.jpg" alt="" style="width: 80%;" />
</div>


| Model  | Dimension | Input Dimension | Output Dimension | Feedforward Dimension | Frequency Dimension | Number of Heads | Number of Layers |
|--------|-----------|-----------------|------------------|-----------------------|---------------------|-----------------|------------------|
| 1.3B   | 1536      | 16              | 16               | 8960                  | 256                 | 12              | 30               |
| 14B   | 5120       | 16              | 16               | 13824                 | 256                 | 40              | 40               |



##### Data

We curated and deduplicated a candidate dataset comprising a vast amount of image and video data. During the data curation process, we designed a four-step data cleaning process, focusing on fundamental dimensions, visual quality and motion quality. Through the robust data processing pipeline, we can easily obtain high-quality, diverse, and large-scale training sets of images and videos.

![figure1](assets/data_for_diff_stage.jpg "figure1")


##### Comparisons to SOTA
We compared **Wan2.1** with leading open-source and closed-source models to evaluate the performance. Using our carefully designed set of 1,035 internal prompts, we tested across 14 major dimensions and 26 sub-dimensions. We then compute the total score by performing a weighted calculation on the scores of each dimension, utilizing weights derived from human preferences in the matching process. The detailed results are shown in the table below. These results demonstrate our model's superior performance compared to both open-source and closed-source models.

![figure1](assets/vben_vs_sota.png "figure1")


## Citation
If you find our work helpful, please cite us.

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License Agreement
The models in this repository are licensed under the Apache 2.0 License. We claim no rights over the your generated contents, granting you the freedom to use them while ensuring that your usage complies with the provisions of this license. You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations. For a complete list of restrictions and details regarding your rights, please refer to the full text of the [license](LICENSE.txt).


## Acknowledgements

We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research.



## Contact Us
If you would like to leave a message to our research or product teams, feel free to join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)!
