# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os.path as osp
import os
import sys
import warnings

import gradio as gr

warnings.filterwarnings('ignore')

# Model
sys.path.insert(0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image

# Global Var
prompt_expander = None
wan_t2i = None


# Button Func
def prompt_enc(prompt, tar_lang):
    global prompt_expander
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
    if prompt_output.status == False:
        return prompt
    else:
        return prompt_output.prompt


def t2i_generation(txt2img_prompt, resolution, sd_steps, guide_scale,
                   shift_scale, seed, n_prompt):
    global wan_t2i
    # print(f"{txt2img_prompt},{resolution},{sd_steps},{guide_scale},{shift_scale},{seed},{n_prompt}")

    W = int(resolution.split("*")[0])
    H = int(resolution.split("*")[1])
    video = wan_t2i.generate(
        txt2img_prompt,
        size=(W, H),
        frame_num=1,
        shift=shift_scale,
        sampling_steps=sd_steps,
        guide_scale=guide_scale,
        n_prompt=n_prompt,
        seed=seed,
        offload_model=True)

    cache_image(
        tensor=video.squeeze(1)[None],
        save_file="example.png",
        nrow=1,
        normalize=True,
        value_range=(-1, 1))

    return "example.png"


# Interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        Wan2.1 (T2I-14B)
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        Wan: Open and Advanced Large-Scale Video Generative Models.
                    </div>
                    """)

        with gr.Row():
            with gr.Column():
                txt2img_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate",
                )
                tar_lang = gr.Radio(
                    choices=["ZH", "EN"],
                    label="Target language of prompt enhance",
                    value="ZH")
                run_p_button = gr.Button(value="Prompt Enhance")

                with gr.Accordion("Advanced Options", open=True):
                    resolution = gr.Dropdown(
                        label='Resolution(Width*Height)',
                        choices=[
                            '720*1280', '1280*720', '960*960', '1088*832',
                            '832*1088', '480*832', '832*480', '624*624',
                            '704*544', '544*704'
                        ],
                        value='720*1280')

                    with gr.Row():
                        sd_steps = gr.Slider(
                            label="Diffusion steps",
                            minimum=1,
                            maximum=1000,
                            value=50,
                            step=1)
                        guide_scale = gr.Slider(
                            label="Guide scale",
                            minimum=0,
                            maximum=20,
                            value=5.0,
                            step=1)
                    with gr.Row():
                        shift_scale = gr.Slider(
                            label="Shift scale",
                            minimum=0,
                            maximum=10,
                            value=5.0,
                            step=1)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1)
                    n_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Describe the negative prompt you want to add"
                    )

                run_t2i_button = gr.Button("Generate Image")

            with gr.Column():
                result_gallery = gr.Image(
                    label='Generated Image', interactive=False, height=600)

        run_p_button.click(
            fn=prompt_enc,
            inputs=[txt2img_prompt, tar_lang],
            outputs=[txt2img_prompt])

        run_t2i_button.click(
            fn=t2i_generation,
            inputs=[
                txt2img_prompt, resolution, sd_steps, guide_scale, shift_scale,
                seed, n_prompt
            ],
            outputs=[result_gallery],
        )

    return demo


# Main
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image from a text prompt or image using Gradio")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="cache",
        help="The path to the checkpoint directory.")
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

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = _parse_args()

    print("Step1: Init prompt_expander...", end='', flush=True)
    if args.prompt_extend_method == "dashscope":
        prompt_expander = DashScopePromptExpander(
            model_name=args.prompt_extend_model, is_vl=False)
    elif args.prompt_extend_method == "local_qwen":
        prompt_expander = QwenPromptExpander(
            model_name=args.prompt_extend_model, is_vl=False, device=0)
    else:
        raise NotImplementedError(
            f"Unsupport prompt_extend_method: {args.prompt_extend_method}")
    print("done", flush=True)

    print("Step2: Init 14B t2i model...", end='', flush=True)
    cfg = WAN_CONFIGS['t2i-14B']
    wan_t2i = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
    )
    print("done", flush=True)

    demo = gradio_interface()
    demo.launch(server_name="0.0.0.0", share=False, server_port=7860)
