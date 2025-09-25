import torch
import gradio as gr


def get_qwen_text_encoder_filename(text_encoder_quantization):
    text_encoder_filename = "ckpts/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct_bf16.safetensors"
    if text_encoder_quantization =="int8":
        text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_bf16_int8") 
    return text_encoder_filename

class family_handler():
    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {
            "image_outputs" : True,
            "sample_solvers":[
                            ("Default", "default"),
                            ("Lightning", "lightning")],
            "guidance_max_phases" : 1,
            "fit_into_canvas_image_refs": 0,
        }

        if base_model_type in ["qwen_image_edit_20B", "qwen_image_edit_plus_20B"]: 
            extra_model_def["inpaint_support"] = True
            extra_model_def["image_ref_choices"] = {
            "choices": [
                ("None", ""),
                ("Conditional Images is first Main Subject / Landscape and may be followed by People / Objects", "KI"),
                ("Conditional Images are People / Objects", "I"),
                ],
            "letters_filter": "KI",
            }
            extra_model_def["background_removal_label"]= "Remove Backgrounds only behind People / Objects except main Subject / Landscape" 
            extra_model_def["video_guide_outpainting"] = [2]
            extra_model_def["model_modes"] = {
                        "choices": [
                            ("Lora Inpainting: Inpainted area completely unrelated to occulted content", 1),
                            ("Masked Denoising : Inpainted area may reuse some content that has been occulted", 0),
                            ],
                        "default": 1,
                        "label" : "Inpainting Method",
                        "image_modes" : [2],
            }

        if base_model_type in ["qwen_image_edit_plus_20B"]: 
            extra_model_def["guide_preprocessing"] = {
                    "selection": ["", "PV", "SV", "CV"],
                }

            extra_model_def["mask_preprocessing"] = {
                    "selection": ["", "A"],
                    "visible": False,
                }
        return extra_model_def

    @staticmethod
    def query_supported_types():
        return ["qwen_image_20B", "qwen_image_edit_20B", "qwen_image_edit_plus_20B"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "qwen"

    @staticmethod
    def query_family_infos():
        return {"qwen":(40, "Qwen")}

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        text_encoder_filename = get_qwen_text_encoder_filename(text_encoder_quantization)    
        return  {  
            "repoId" : "DeepBeepMeep/Qwen_image", 
            "sourceFolderList" :  ["", "Qwen2.5-VL-7B-Instruct"],
            "fileList" : [ ["qwen_vae.safetensors", "qwen_vae_config.json"], ["merges.txt", "tokenizer_config.json", "config.json", "vocab.json", "video_preprocessor_config.json", "preprocessor_config.json"] + computeList(text_encoder_filename)  ]
            }

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized = False, submodel_no_list = None):
        from .qwen_main import model_factory
        from mmgp import offload

        pipe_processor = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type = model_type, 
            model_def = model_def,
            base_model_type=base_model_type,
            text_encoder_filename= get_qwen_text_encoder_filename(text_encoder_quantization),
            quantizeTransformer = quantizeTransformer,
            dtype = dtype,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer,
            save_quantized = save_quantized
        )

        pipe = {"tokenizer" : pipe_processor.tokenizer, "transformer" : pipe_processor.transformer, "text_encoder" : pipe_processor.text_encoder, "vae" : pipe_processor.vae}

        return pipe_processor, pipe


    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if ui_defaults.get("sample_solver", "") == "": 
            ui_defaults["sample_solver"] = "default"

        if settings_version < 2.32:
            ui_defaults["denoising_strength"] = 1.
                            
    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "guidance_scale":  4,
            "sample_solver": "default",
        })            
        if base_model_type in ["qwen_image_edit_20B"]: 
            ui_defaults.update({
                "video_prompt_type": "KI",
                "denoising_strength" : 1.,
                "model_mode" : 0,
            })
        elif base_model_type in ["qwen_image_edit_plus_20B"]: 
            ui_defaults.update({
                "video_prompt_type": "I",
                "denoising_strength" : 1.,
                "model_mode" : 0,
            })

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if base_model_type in ["qwen_image_edit_20B", "qwen_image_edit_plus_20B"]:
            model_mode = inputs["model_mode"]
            denoising_strength= inputs["denoising_strength"]
            video_guide_outpainting= inputs["video_guide_outpainting"]
            from wgp import get_outpainting_dims
            outpainting_dims = get_outpainting_dims(video_guide_outpainting)

            if denoising_strength < 1 and model_mode == 1:
                gr.Info("Denoising Strength will be ignored while using Lora Inpainting")
            if outpainting_dims is not None and model_mode == 0 :
                return "Outpainting is not supported with Masked Denoising  "
            
    @staticmethod
    def get_rgb_factors(base_model_type ):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("qwen")
        return latent_rgb_factors, latent_rgb_factors_bias
