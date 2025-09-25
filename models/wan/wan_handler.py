import torch
import numpy as np
import gradio as gr

def test_class_i2v(base_model_type):    
    return base_model_type in ["i2v", "i2v_2_2", "fun_inp_1.3B", "fun_inp", "flf2v_720p",  "fantasy",  "multitalk", "infinitetalk", "i2v_2_2_multitalk", "animate" ]

def text_oneframe_overlap(base_model_type):
    return test_class_i2v(base_model_type) and not (test_multitalk(base_model_type) or base_model_type in ["animate"]) or test_wan_5B(base_model_type)

def test_class_1_3B(base_model_type):    
    return base_model_type in [ "vace_1.3B", "t2v_1.3B", "recam_1.3B","phantom_1.3B","fun_inp_1.3B"]

def test_multitalk(base_model_type):
    return base_model_type in ["multitalk", "vace_multitalk_14B", "i2v_2_2_multitalk", "infinitetalk"]

def test_standin(base_model_type):
    return base_model_type in ["standin", "vace_standin_14B"]

def test_wan_5B(base_model_type):
    return base_model_type in ["ti2v_2_2", "lucy_edit"]
class family_handler():

    @staticmethod
    def set_cache_parameters(cache_type, base_model_type, model_def, inputs, skip_steps_cache):
        i2v =  test_class_i2v(base_model_type)

        resolution = inputs["resolution"]
        width, height = resolution.split("x")
        pixels = int(width) * int(height)

        if cache_type == "mag":
            skip_steps_cache.update({     
            "magcache_thresh" : 0,
            "magcache_K" : 2,
            })
            if base_model_type in ["t2v"] and "URLs2" in model_def:
                def_mag_ratios = [1.00124, 1.00155, 0.99822, 0.99851, 0.99696, 0.99687, 0.99703, 0.99732, 0.9966, 0.99679, 0.99602, 0.99658, 0.99578, 0.99664, 0.99484, 0.9949, 0.99633, 0.996, 0.99659, 0.99683, 0.99534, 0.99549, 0.99584, 0.99577, 0.99681, 0.99694, 0.99563, 0.99554, 0.9944, 0.99473, 0.99594, 0.9964, 0.99466, 0.99461, 0.99453, 0.99481, 0.99389, 0.99365, 0.99391, 0.99406, 0.99354, 0.99361, 0.99283, 0.99278, 0.99268, 0.99263, 0.99057, 0.99091, 0.99125, 0.99126, 0.65523, 0.65252, 0.98808, 0.98852, 0.98765, 0.98736, 0.9851, 0.98535, 0.98311, 0.98339, 0.9805, 0.9806, 0.97776, 0.97771, 0.97278, 0.97286, 0.96731, 0.96728, 0.95857, 0.95855, 0.94385, 0.94385, 0.92118, 0.921, 0.88108, 0.88076, 0.80263, 0.80181]
            elif base_model_type in ["i2v_2_2"]:
                def_mag_ratios = [0.99191, 0.99144, 0.99356, 0.99337, 0.99326, 0.99285, 0.99251, 0.99264, 0.99393, 0.99366, 0.9943, 0.9943, 0.99276, 0.99288, 0.99389, 0.99393, 0.99274, 0.99289, 0.99316, 0.9931, 0.99379, 0.99377, 0.99268, 0.99271, 0.99222, 0.99227, 0.99175, 0.9916, 0.91076, 0.91046, 0.98931, 0.98933, 0.99087, 0.99088, 0.98852, 0.98855, 0.98895, 0.98896, 0.98806, 0.98808, 0.9871, 0.98711, 0.98613, 0.98618, 0.98434, 0.98435, 0.983, 0.98307, 0.98185, 0.98187, 0.98131, 0.98131, 0.9783, 0.97835, 0.97619, 0.9762, 0.97264, 0.9727, 0.97088, 0.97098, 0.96568, 0.9658, 0.96045, 0.96055, 0.95322, 0.95335, 0.94579, 0.94594, 0.93297, 0.93311, 0.91699, 0.9172, 0.89174, 0.89202, 0.8541, 0.85446, 0.79823, 0.79902]
            elif test_wan_5B(base_model_type):
                if inputs.get("image_start", None) is not None and inputs.get("video_source", None) is not None : # t2v
                    def_mag_ratios = [0.99505, 0.99389, 0.99441, 0.9957, 0.99558, 0.99551, 0.99499, 0.9945, 0.99534, 0.99548, 0.99468, 0.9946, 0.99463, 0.99458, 0.9946, 0.99453, 0.99408, 0.99404, 0.9945, 0.99441, 0.99409, 0.99398, 0.99403, 0.99397, 0.99382, 0.99377, 0.99349, 0.99343, 0.99377, 0.99378, 0.9933, 0.99328, 0.99303, 0.99301, 0.99217, 0.99216, 0.992, 0.99201, 0.99201, 0.99202, 0.99133, 0.99132, 0.99112, 0.9911, 0.99155, 0.99155, 0.98958, 0.98957, 0.98959, 0.98958, 0.98838, 0.98835, 0.98826, 0.98825, 0.9883, 0.98828, 0.98711, 0.98709, 0.98562, 0.98561, 0.98511, 0.9851, 0.98414, 0.98412, 0.98284, 0.98282, 0.98104, 0.98101, 0.97981, 0.97979, 0.97849, 0.97849, 0.97557, 0.97554, 0.97398, 0.97395, 0.97171, 0.97166, 0.96917, 0.96913, 0.96511, 0.96507, 0.96263, 0.96257, 0.95839, 0.95835, 0.95483, 0.95475, 0.94942, 0.94936, 0.9468, 0.94678, 0.94583, 0.94594, 0.94843, 0.94872, 0.96949, 0.97015]
                else: # i2v
                    def_mag_ratios = [0.99512, 0.99559, 0.99559, 0.99561, 0.99595, 0.99577, 0.99512, 0.99512, 0.99546, 0.99534, 0.99543, 0.99531, 0.99496, 0.99491, 0.99504, 0.99499, 0.99444, 0.99449, 0.99481, 0.99481, 0.99435, 0.99435, 0.9943, 0.99431, 0.99411, 0.99406, 0.99373, 0.99376, 0.99413, 0.99405, 0.99363, 0.99359, 0.99335, 0.99331, 0.99244, 0.99243, 0.99229, 0.99229, 0.99239, 0.99236, 0.99163, 0.9916, 0.99149, 0.99151, 0.99191, 0.99192, 0.9898, 0.98981, 0.9899, 0.98987, 0.98849, 0.98849, 0.98846, 0.98846, 0.98861, 0.98861, 0.9874, 0.98738, 0.98588, 0.98589, 0.98539, 0.98534, 0.98444, 0.98439, 0.9831, 0.98309, 0.98119, 0.98118, 0.98001, 0.98, 0.97862, 0.97859, 0.97555, 0.97558, 0.97392, 0.97388, 0.97152, 0.97145, 0.96871, 0.9687, 0.96435, 0.96434, 0.96129, 0.96127, 0.95639, 0.95638, 0.95176, 0.95175, 0.94446, 0.94452, 0.93972, 0.93974, 0.93575, 0.9359, 0.93537, 0.93552, 0.96655, 0.96616]
            elif test_class_1_3B(base_model_type): #text 1.3B
                def_mag_ratios = [1.0124, 1.02213, 1.00166, 1.0041, 0.99791, 1.00061, 0.99682, 0.99762, 0.99634, 0.99685, 0.99567, 0.99586, 0.99416, 0.99422, 0.99578, 0.99575, 0.9957, 0.99563, 0.99511, 0.99506, 0.99535, 0.99531, 0.99552, 0.99549, 0.99541, 0.99539, 0.9954, 0.99536, 0.99489, 0.99485, 0.99518, 0.99514, 0.99484, 0.99478, 0.99481, 0.99479, 0.99415, 0.99413, 0.99419, 0.99416, 0.99396, 0.99393, 0.99388, 0.99386, 0.99349, 0.99349, 0.99309, 0.99304, 0.9927, 0.9927, 0.99228, 0.99226, 0.99171, 0.9917, 0.99137, 0.99135, 0.99068, 0.99063, 0.99005, 0.99003, 0.98944, 0.98942, 0.98849, 0.98849, 0.98758, 0.98757, 0.98644, 0.98643, 0.98504, 0.98503, 0.9836, 0.98359, 0.98202, 0.98201, 0.97977, 0.97978, 0.97717, 0.97718, 0.9741, 0.97411, 0.97003, 0.97002, 0.96538, 0.96541, 0.9593, 0.95933, 0.95086, 0.95089, 0.94013, 0.94019, 0.92402, 0.92414, 0.90241, 0.9026, 0.86821, 0.86868, 0.81838, 0.81939]#**(0.5)# In our papaer, we utilize the sqrt to smooth the ratio, which has little impact on the performance and can be deleted.
            elif i2v:
                if pixels >= 1280*720:
                    def_mag_ratios = [0.99428, 0.99498, 0.98588, 0.98621, 0.98273, 0.98281, 0.99018, 0.99023, 0.98911, 0.98917, 0.98646, 0.98652, 0.99454, 0.99456, 0.9891, 0.98909, 0.99124, 0.99127, 0.99102, 0.99103, 0.99215, 0.99212, 0.99515, 0.99515, 0.99576, 0.99572, 0.99068, 0.99072, 0.99097, 0.99097, 0.99166, 0.99169, 0.99041, 0.99042, 0.99201, 0.99198, 0.99101, 0.99101, 0.98599, 0.98603, 0.98845, 0.98844, 0.98848, 0.98851, 0.98862, 0.98857, 0.98718, 0.98719, 0.98497, 0.98497, 0.98264, 0.98263, 0.98389, 0.98393, 0.97938, 0.9794, 0.97535, 0.97536, 0.97498, 0.97499, 0.973, 0.97301, 0.96827, 0.96828, 0.96261, 0.96263, 0.95335, 0.9534, 0.94649, 0.94655, 0.93397, 0.93414, 0.91636, 0.9165, 0.89088, 0.89109, 0.8679, 0.86768]
                else:
                    def_mag_ratios =  [0.98783, 0.98993, 0.97559, 0.97593, 0.98311, 0.98319, 0.98202, 0.98225, 0.9888, 0.98878, 0.98762, 0.98759, 0.98957, 0.98971, 0.99052, 0.99043, 0.99383, 0.99384, 0.98857, 0.9886, 0.99065, 0.99068, 0.98845, 0.98847, 0.99057, 0.99057, 0.98957, 0.98961, 0.98601, 0.9861, 0.98823, 0.98823, 0.98756, 0.98759, 0.98808, 0.98814, 0.98721, 0.98724, 0.98571, 0.98572, 0.98543, 0.98544, 0.98157, 0.98165, 0.98411, 0.98413, 0.97952, 0.97953, 0.98149, 0.9815, 0.9774, 0.97742, 0.97825, 0.97826, 0.97355, 0.97361, 0.97085, 0.97087, 0.97056, 0.97055, 0.96588, 0.96587, 0.96113, 0.96124, 0.9567, 0.95681, 0.94961, 0.94969, 0.93973, 0.93988, 0.93217, 0.93224, 0.91878, 0.91896, 0.90955, 0.90954, 0.92617, 0.92616]
            else: # text 14B
                def_mag_ratios = [1.02504, 1.03017, 1.00025, 1.00251, 0.9985, 0.99962, 0.99779, 0.99771, 0.9966, 0.99658, 0.99482, 0.99476, 0.99467, 0.99451, 0.99664, 0.99656, 0.99434, 0.99431, 0.99533, 0.99545, 0.99468, 0.99465, 0.99438, 0.99434, 0.99516, 0.99517, 0.99384, 0.9938, 0.99404, 0.99401, 0.99517, 0.99516, 0.99409, 0.99408, 0.99428, 0.99426, 0.99347, 0.99343, 0.99418, 0.99416, 0.99271, 0.99269, 0.99313, 0.99311, 0.99215, 0.99215, 0.99218, 0.99215, 0.99216, 0.99217, 0.99163, 0.99161, 0.99138, 0.99135, 0.98982, 0.9898, 0.98996, 0.98995, 0.9887, 0.98866, 0.98772, 0.9877, 0.98767, 0.98765, 0.98573, 0.9857, 0.98501, 0.98498, 0.9838, 0.98376, 0.98177, 0.98173, 0.98037, 0.98035, 0.97678, 0.97677, 0.97546, 0.97543, 0.97184, 0.97183, 0.96711, 0.96708, 0.96349, 0.96345, 0.95629, 0.95625, 0.94926, 0.94929, 0.93964, 0.93961, 0.92511, 0.92504, 0.90693, 0.90678, 0.8796, 0.87945, 0.86111, 0.86189]
            skip_steps_cache.def_mag_ratios = def_mag_ratios
        else:
            if i2v:
                if pixels >= 1280*720:
                    coefficients= [-114.36346466,   65.26524496,  -18.82220707,    4.91518089,   -0.23412683]
                else:
                    coefficients= [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01,  5.87348440e+00, -2.01973289e-01]
            else:
                if test_class_1_3B(base_model_type):
                    coefficients= [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
                else: 
                    coefficients= [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
            skip_steps_cache.coefficients = coefficients

    @staticmethod
    def get_wan_text_encoder_filename(text_encoder_quantization):
        text_encoder_filename = "ckpts/umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
        if text_encoder_quantization =="int8":
            text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_int8") 
        return text_encoder_filename

    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {}
        if "URLs2" in model_def:
            extra_model_def["no_steps_skipping"] = True
        i2v =  test_class_i2v(base_model_type)
        extra_model_def["i2v_class"] = i2v
        extra_model_def["multitalk_class"] = test_multitalk(base_model_type)
        extra_model_def["standin_class"] = test_standin(base_model_type)
        vace_class = base_model_type in ["vace_14B", "vace_1.3B", "vace_multitalk_14B", "vace_standin_14B"] 
        extra_model_def["vace_class"] = vace_class

        if base_model_type in ["animate"]:
            fps = 30
        elif test_multitalk(base_model_type):
            fps = 25
        elif base_model_type in ["fantasy"]:
            fps = 23
        elif test_wan_5B(base_model_type):
            fps = 24
        else:
            fps = 16
        extra_model_def["fps"] =fps
        multiple_submodels = "URLs2" in model_def
        if vace_class: 
            frames_minimum, frames_steps =  17, 4
        else:
            frames_minimum, frames_steps = 5, 4
        extra_model_def.update({
        "frames_minimum" : frames_minimum,
        "frames_steps" : frames_steps, 
        "sliding_window" : base_model_type in ["multitalk", "infinitetalk", "t2v", "fantasy", "animate"] or test_class_i2v(base_model_type) or test_wan_5B(base_model_type) or vace_class,  #"ti2v_2_2",
        "multiple_submodels" : multiple_submodels,
        "guidance_max_phases" : 3,
        "skip_layer_guidance" : True,        
        "cfg_zero" : True,
        "cfg_star" : True,
        "adaptive_projected_guidance" : True,  
        "tea_cache" : not (base_model_type in ["i2v_2_2"] or test_wan_5B(base_model_type) or multiple_submodels),
        "mag_cache" : True,
        "keep_frames_video_guide_not_supported": base_model_type in ["infinitetalk"],
        "sample_solvers":[
                            ("unipc", "unipc"),
                            ("euler", "euler"),
                            ("dpm++", "dpm++"),
                            ("flowmatch causvid", "causvid"), ]
        })


        if base_model_type in ["t2v"]: 
            extra_model_def["guide_custom_choices"] = {
                "choices":[("Use Text Prompt Only", ""),("Video to Video guided by Text Prompt", "GUV")],
                "default": "",
                "letters_filter": "GUV",
                "label": "Video to Video"
            }

        if base_model_type in ["infinitetalk"]: 
            extra_model_def["no_background_removal"] = True
            extra_model_def["all_image_refs_are_background_ref"] = True
            extra_model_def["guide_custom_choices"] = {
            "choices":[
                ("Images to Video, each Reference Image will start a new shot with a new Sliding Window - Sharp Transitions", "QKI"),
                ("Images to Video, each Reference Image will start a new shot with a new Sliding Window - Smooth Transitions", "KI"),
                ("Sparse Video to Video, one Image will by extracted from Video for each new Sliding Window - Sharp Transitions", "QRUV"),
                ("Sparse Video to Video, one Image will by extracted from Video for each new Sliding Window - Smooth Transitions", "RUV"),
                ("Video to Video, amount of motion transferred depends on Denoising Strength - Sharp Transitions", "GQUV"),
                ("Video to Video, amount of motion transferred depends on Denoising Strength - Smooth Transitions", "GUV"),
            ],
            "default": "KI",
            "letters_filter": "RGUVQKI",
            "label": "Video to Video",
            "show_label" : False,
            }

            # extra_model_def["at_least_one_image_ref_needed"] = True
        if base_model_type in ["lucy_edit"]:
            extra_model_def["keep_frames_video_guide_not_supported"] = True
            extra_model_def["guide_preprocessing"] = {
                    "selection": ["UV"],
                    "labels" : { "UV": "Control Video"},
                    "visible": False,
                }

        if base_model_type in ["animate"]:
            extra_model_def["guide_custom_choices"] = {
            "choices":[
                ("Animate Person in Reference Image using Motion of Person in Control Video", "PVBXAKI"),
                ("Replace Person in Control Video Person in Reference Image", "PVBAI"),
            ],
            "default": "KI",
            "letters_filter": "PVBXAKI",
            "label": "Type of Process",
            "show_label" : False,
            }
            extra_model_def["video_guide_outpainting"] = [0,1]
            extra_model_def["keep_frames_video_guide_not_supported"] = True
            extra_model_def["extract_guide_from_window_start"] = True
            extra_model_def["forced_guide_mask_inputs"] = True
            extra_model_def["background_removal_label"]= "Remove Backgrounds behind People (Animate Mode Only)"
            extra_model_def["background_ref_outpainted"] = False
            extra_model_def["return_image_refs_tensor"] = True
            extra_model_def["guide_inpaint_color"] = 0



        if vace_class:
            extra_model_def["guide_preprocessing"] = {
                    "selection": ["", "UV", "PV", "DV", "SV", "LV", "CV", "MV", "V", "PDV", "PSV", "PLV" , "DSV", "DLV", "SLV"],
                    "labels" : { "V": "Use Vace raw format"}
                }
            extra_model_def["mask_preprocessing"] = {
                    "selection": ["", "A", "NA", "XA", "XNA", "YA", "YNA", "WA", "WNA", "ZA", "ZNA"],
                }

            extra_model_def["image_ref_choices"] = {
                    "choices": [("None", ""),
                    ("People / Objects", "I"),
                    ("Landscape followed by People / Objects (if any)", "KI"),
                    ("Positioned Frames followed by People / Objects (if any)", "FI"),
                    ],
                    "letters_filter":  "KFI",
            }

            extra_model_def["background_removal_label"]= "Remove Backgrounds behind People / Objects, keep it for Landscape or Positioned Frames"
            extra_model_def["video_guide_outpainting"] = [0,1]
            extra_model_def["pad_guide_video"] = True
            extra_model_def["guide_inpaint_color"] = 127.5
            extra_model_def["forced_guide_mask_inputs"] = True
            extra_model_def["return_image_refs_tensor"] = True
            
        if base_model_type in ["standin"]: 
            extra_model_def["fit_into_canvas_image_refs"] = 0
            extra_model_def["image_ref_choices"] = {
                "choices": [
                    ("No Reference Image", ""),
                    ("Reference Image is a Person Face", "I"),
                    ],
                "letters_filter":"I",
            }

        if base_model_type in ["phantom_1.3B", "phantom_14B"]: 
            extra_model_def["image_ref_choices"] = {
                "choices": [("Reference Image", "I")],
                "letters_filter":"I",
                "visible": False,
            }

        if base_model_type in ["recam_1.3B"]: 
            extra_model_def["keep_frames_video_guide_not_supported"] = True
            extra_model_def["model_modes"] = {
                        "choices": [
                            ("Pan Right", 1),
                            ("Pan Left", 2),
                            ("Tilt Up", 3),
                            ("Tilt Down", 4),
                            ("Zoom In", 5),
                            ("Zoom Out", 6),
                            ("Translate Up (with rotation)", 7),
                            ("Translate Down (with rotation)", 8),
                            ("Arc Left (with rotation)", 9),
                            ("Arc Right (with rotation)", 10),
                        ],
                        "default": 1,
                        "label" : "Camera Movement Type"
            }
            extra_model_def["guide_preprocessing"] = {
                    "selection": ["UV"],
                    "labels" : { "UV": "Control Video"},
                    "visible" : False,
                }

        if vace_class or base_model_type in ["animate"]:
            image_prompt_types_allowed = "TVL"
        elif base_model_type in ["infinitetalk"]:
            image_prompt_types_allowed = "TSVL"
        elif base_model_type in ["ti2v_2_2"]:
            image_prompt_types_allowed = "TSVL"
        elif base_model_type in ["lucy_edit"]:
            image_prompt_types_allowed = "TVL"
        elif test_multitalk(base_model_type) or base_model_type in ["fantasy"]:
            image_prompt_types_allowed = "SVL"
        elif i2v:
            image_prompt_types_allowed = "SEVL"
        else:
            image_prompt_types_allowed = ""
        extra_model_def["image_prompt_types_allowed"] = image_prompt_types_allowed

        if text_oneframe_overlap(base_model_type):
            extra_model_def["sliding_window_defaults"] = { "overlap_min" : 1, "overlap_max" : 1, "overlap_step": 0, "overlap_default": 1}

        # if base_model_type in ["phantom_1.3B", "phantom_14B"]: 
        #     extra_model_def["one_image_ref_needed"] = True


        return extra_model_def
        
    @staticmethod
    def query_supported_types():
        return ["multitalk", "infinitetalk", "fantasy", "vace_14B", "vace_multitalk_14B", "vace_standin_14B",
                    "t2v_1.3B", "standin", "t2v", "vace_1.3B", "phantom_1.3B", "phantom_14B", 
                    "recam_1.3B", "animate",
                    "i2v", "i2v_2_2", "i2v_2_2_multitalk", "ti2v_2_2", "lucy_edit", "flf2v_720p", "fun_inp_1.3B", "fun_inp"]


    @staticmethod
    def query_family_maps():

        models_eqv_map = {
            "flf2v_720p" : "i2v",
            "t2v_1.3B" : "t2v",
        }

        models_comp_map = { 
                    "vace_14B" : [ "vace_multitalk_14B", "vace_standin_14B"],
                    "t2v" : [ "vace_14B", "vace_1.3B" "vace_multitalk_14B", "t2v_1.3B", "phantom_1.3B","phantom_14B", "standin"],
                    "i2v" : [ "fantasy", "multitalk", "flf2v_720p" ],
                    "i2v_2_2" : ["i2v_2_2_multitalk"],
                    "fantasy": ["multitalk"],
                    }
        return models_eqv_map, models_comp_map

    @staticmethod
    def query_model_family():
        return "wan"
    
    @staticmethod
    def query_family_infos():
        return {"wan":(0, "Wan2.1"), "wan2_2":(1, "Wan2.2") }

    @staticmethod
    def get_vae_block_size(base_model_type):
        return 32 if test_wan_5B(base_model_type) else 16

    @staticmethod
    def get_rgb_factors(base_model_type ):
        from shared.RGB_factors import get_rgb_factors
        if test_wan_5B(base_model_type): base_model_type = "ti2v_2_2"
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("wan", base_model_type)
        return latent_rgb_factors, latent_rgb_factors_bias
    
    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        text_encoder_filename = family_handler.get_wan_text_encoder_filename(text_encoder_quantization)

        download_def  = [{
            "repoId" : "DeepBeepMeep/Wan2.1", 
            "sourceFolderList" :  ["xlm-roberta-large", "umt5-xxl", ""  ],
            "fileList" : [ [ "models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"], ["special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"] + computeList(text_encoder_filename) , ["Wan2.1_VAE.safetensors",  "fantasy_proj_model.safetensors" ] +  computeList(model_filename)  ]   
        }]

        if test_wan_5B(base_model_type):
            download_def += [    {
                "repoId" : "DeepBeepMeep/Wan2.2", 
                "sourceFolderList" :  [""],
                "fileList" : [ [ "Wan2.2_VAE.safetensors" ]  ]
            }]

        return download_def


    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized= False, submodel_no_list = None):
        from .configs import WAN_CONFIGS

        if test_class_i2v(base_model_type):
            cfg = WAN_CONFIGS['i2v-14B']
        else:
            cfg = WAN_CONFIGS['t2v-14B']
            # cfg = WAN_CONFIGS['t2v-1.3B']    
        from . import WanAny2V
        wan_model = WanAny2V(
            config=cfg,
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            submodel_no_list = submodel_no_list,
            model_type = model_type,        
            model_def = model_def,
            base_model_type=base_model_type,
            text_encoder_filename= family_handler.get_wan_text_encoder_filename(text_encoder_quantization),
            quantizeTransformer = quantizeTransformer,
            dtype = dtype,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer,
            save_quantized = save_quantized
        )

        pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model, "vae": wan_model.vae.model }
        if hasattr(wan_model,"model2") and wan_model.model2 is not None:
            pipe["transformer2"] = wan_model.model2
        if hasattr(wan_model, "clip"):
            pipe["text_encoder_2"] = wan_model.clip.model
        return wan_model, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if ui_defaults.get("sample_solver", "") == "": 
            ui_defaults["sample_solver"] = "unipc"

        if settings_version < 2.24:
            if (model_def.get("multiple_submodels", False) or ui_defaults.get("switch_threshold", 0) > 0) and ui_defaults.get("guidance_phases",0)<2:
                ui_defaults["guidance_phases"] = 2

        if settings_version == 2.24 and ui_defaults.get("guidance_phases",0) ==2:
            mult = model_def.get("loras_multipliers","")
            if len(mult)> 1 and len(mult[0].split(";"))==3: ui_defaults["guidance_phases"] = 3

        if settings_version < 2.27:
            if base_model_type in "infinitetalk":
                guidance_scale = ui_defaults.get("guidance_scale", None)
                if guidance_scale == 1:
                    ui_defaults["audio_guidance_scale"]= 1
                video_prompt_type = ui_defaults.get("video_prompt_type", "")
                if "I" in video_prompt_type:
                    video_prompt_type = video_prompt_type.replace("KI", "QKI")
                    ui_defaults["video_prompt_type"] = video_prompt_type 

        if settings_version < 2.28:
            if base_model_type in "infinitetalk":
                video_prompt_type = ui_defaults.get("video_prompt_type", "")
                if "U" in video_prompt_type:
                    video_prompt_type = video_prompt_type.replace("U", "RU")
                    ui_defaults["video_prompt_type"] = video_prompt_type 

        if settings_version < 2.31:
            if base_model_type in "recam_1.3B":
                video_prompt_type = ui_defaults.get("video_prompt_type", "")
                if not "V" in video_prompt_type:
                    video_prompt_type += "UV"
                    ui_defaults["video_prompt_type"] = video_prompt_type 
                    ui_defaults["image_prompt_type"] = ""

            if text_oneframe_overlap(base_model_type):
                ui_defaults["sliding_window_overlap"] = 1

        if settings_version < 2.32:
            image_prompt_type = ui_defaults.get("image_prompt_type", "")
            if test_class_i2v(base_model_type) and len(image_prompt_type) == 0:
                ui_defaults["image_prompt_type"] = "S" 

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "sample_solver": "unipc",
        })
        if test_class_i2v(base_model_type) and "S" in model_def["image_prompt_types_allowed"]:
            ui_defaults["image_prompt_type"] = "S" 

        if base_model_type in ["fantasy"]:
            ui_defaults.update({
                "audio_guidance_scale": 5.0,
                "sliding_window_overlap" : 1,
            })

        elif base_model_type in ["multitalk"]:
            ui_defaults.update({
                "guidance_scale": 5.0,
                "flow_shift": 7, # 11 for 720p
                "sliding_window_discard_last_frames" : 4,
                "sample_solver" : "euler",
                "adaptive_switch" : 1,
            })

        elif base_model_type in ["infinitetalk"]:
            ui_defaults.update({
                "guidance_scale": 5.0,
                "flow_shift": 7, # 11 for 720p
                "sliding_window_overlap" : 9,
                "sliding_window_size": 81, 
                "sample_solver" : "euler",
                "video_prompt_type": "QKI",
                "remove_background_images_ref" : 0,
                "adaptive_switch" : 1,
            })

        elif base_model_type in ["standin"]:
            ui_defaults.update({
                "guidance_scale": 5.0,
                "flow_shift": 7, # 11 for 720p
                "sliding_window_overlap" : 9,
                "video_prompt_type": "I",
                "remove_background_images_ref" : 1,
            })
        elif base_model_type in ["phantom_1.3B", "phantom_14B"]:
            ui_defaults.update({
                "guidance_scale": 7.5,
                "flow_shift": 5,
                "remove_background_images_ref": 1,
                "video_prompt_type": "I",
                # "resolution": "1280x720" 
            })

        elif base_model_type in ["vace_14B", "vace_multitalk_14B"]:
            ui_defaults.update({
                "sliding_window_discard_last_frames": 0,
            })

        elif base_model_type in ["ti2v_2_2"]:
            ui_defaults.update({
                "image_prompt_type": "T", 
            })

        if base_model_type in ["recam_1.3B", "lucy_edit"]: 
            ui_defaults.update({
                "video_prompt_type": "UV", 
            })
        elif base_model_type in ["animate"]: 
            ui_defaults.update({
                "video_prompt_type": "PVBXAKI", 
                "mask_expand": 20,
                "audio_prompt_type": "R",
            })

        if text_oneframe_overlap(base_model_type):
            ui_defaults["sliding_window_overlap"] = 1
            ui_defaults["color_correction_strength"]= 0

        if test_multitalk(base_model_type):
            ui_defaults["audio_guidance_scale"] = 4

        if model_def.get("multiple_submodels", False):
            ui_defaults["guidance_phases"] = 2
    
    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if base_model_type in ["infinitetalk"]:
            video_source = inputs["video_source"]
            image_refs = inputs["image_refs"]
            video_prompt_type = inputs["video_prompt_type"]
            image_prompt_type = inputs["image_prompt_type"]
            if ("V" in image_prompt_type or "L" in image_prompt_type) and image_refs is None:
                video_prompt_type = video_prompt_type.replace("I", "").replace("K","")
                inputs["video_prompt_type"] = video_prompt_type 


        if base_model_type in ["vace_standin_14B"]:
            image_refs = inputs["image_refs"]
            video_prompt_type = inputs["video_prompt_type"]
            if image_refs is not None and  len(image_refs) == 1 and "K" in video_prompt_type:
                gr.Info("Warning, Ref Image for Standin Missing: if 'Landscape and then People or Objects' is selected beside the Landscape Image Ref there should be another Image Ref that contains a Face.")
                    
