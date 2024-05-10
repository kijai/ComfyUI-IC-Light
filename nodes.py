import torch
import folder_paths
import os
import types
from comfy.utils import load_torch_file
from .utils.convert_unet import convert_iclight_unet
from .utils.patches import calculate_weight_adjust_channel
from comfy.model_patcher import ModelPatcher

class LoadAndApplyICLightUnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_path": (folder_paths.get_filename_list("unet"), )
            } 
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "IC-Light"
    DESCRIPTION = """
  
Bit hacky (but currently working) way to load the diffusers IC-Light models available here:  
https://huggingface.co/lllyasviel/ic-light/tree/main  
  
Used with ICLightConditioning -node  
"""

    def load(self, model, model_path):
        print("LoadAndApplyICLightUnet: Checking IC-Light Unet path")
        model_full_path = folder_paths.get_full_path("unet", model_path)
        if not os.path.exists(model_full_path):
            raise Exception("Invalid model path")
        else:
            print("LoadAndApplyICLightUnet: Loading IC-Light Unet weights")
            model_clone = model.clone()

            iclight_state_dict = load_torch_file(model_full_path)             
            for key, value in iclight_state_dict.items():
                if key.startswith('conv_in.weight'):
                    in_channels = value.shape[1]
                    break

            # Add weights as patches
            new_keys_dict = convert_iclight_unet(iclight_state_dict)

            print("LoadAndApplyICLightUnet: Attempting to add patches with IC-Light Unet weights")
            model_clone.unpatch_model()
            try:
                for key in new_keys_dict:
                    model_clone.add_patches({key: (new_keys_dict[key],)}, 1.0, 1.0)
            except:
                raise Exception("Could not patch model")
            print("LoadAndApplyICLightUnet: Added LoadICLightUnet patches")

            # # Create a new Conv2d layer with 8 or 12 input channels   
            # original_conv_layer = model_clone.model.diffusion_model.input_blocks[0][0]

            # print(f"LoadAndApplyICLightUnet: Input channels in currently loaded model: {original_conv_layer.in_channels}")
          
            # print("LoadAndApplyICLightUnet: Settings in_channels to: ", in_channels)
            
            # if model_clone.model.diffusion_model.input_blocks[0][0].in_channels != in_channels:
            #     num_channels_to_copy = min(in_channels, original_conv_layer.in_channels)
            #     new_conv_layer = torch.nn.Conv2d(in_channels, original_conv_layer.out_channels, kernel_size=original_conv_layer.kernel_size, stride=original_conv_layer.stride, padding=original_conv_layer.padding)
            #     new_conv_layer.weight.zero_()
            #     new_conv_layer.weight[:, :num_channels_to_copy, :, :].copy_(original_conv_layer.weight[:, :num_channels_to_copy, :, :])
            #     new_conv_layer.bias = original_conv_layer.bias
            #     new_conv_layer = new_conv_layer.to(model_clone.model.diffusion_model.dtype)
            #     original_conv_layer.conv_in = new_conv_layer
            #     # Replace the old layer with the new one
            #     model_clone.model.diffusion_model.input_blocks[0][0] = new_conv_layer
            #     # Verify the change
            #     print(f"LoadAndApplyICLightUnet: New number of input channels: {model_clone.model.diffusion_model.input_blocks[0][0].in_channels}")

            #Patch ComfyUI's LoRA weight application to accept multi-channel inputs. Thanks @huchenlei
            ModelPatcher.calculate_weight = calculate_weight_adjust_channel(ModelPatcher.calculate_weight)   
            # Mimic the existing IP2P class to enable extra_conds
            def bound_extra_conds(self, **kwargs):
                 return ICLight.extra_conds(self, **kwargs)
            new_extra_conds = types.MethodType(bound_extra_conds, model_clone.model)
            model_clone.add_object_patch("extra_conds", new_extra_conds)

            return (model_clone, )

import comfy
class ICLight:
    def extra_conds(self, **kwargs):
        out = {}
        
        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros_like(noise)

        if image.shape[1:] != noise.shape[1:]:
            image = comfy.utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")

        image = comfy.utils.resize_to_batch_size(image, noise.shape[0])

        process_image_in = lambda image: image
        out['c_concat'] = comfy.conds.CONDNoiseShape(process_image_in(image))
        
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = comfy.conds.CONDRegular(adm)
        return out

class ICLightConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "foreground": ("LATENT", ),
                             "multiplier": ("FLOAT", {"default": 0.18215, "min": 0.0, "max": 1.0, "step": 0.001}),
                             },
                "optional": {
                     "opt_background": ("LATENT", ),
                     },
                }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("positive", "negative", "empty_latent")
    FUNCTION = "encode"
    CATEGORY = "IC-Light"
    DESCRIPTION = """
  
Conditioning for the IC-Light model.  
To use the "opt_background" input, you also need to use the  
"fbc" version of the IC-Light models.  
  
"""

    def encode(self, positive, negative, vae, foreground, multiplier, opt_background=None):
        samples_1 = foreground["samples"]

        if opt_background is not None:
            samples_2 = opt_background["samples"]

            concat_latent = torch.cat((samples_1, samples_2), dim=1)
        else:
            concat_latent = samples_1
        print("ICLightConditioning: concat_latent shape: ", concat_latent.shape)

        out_latent = {}
        out_latent["samples"] = torch.zeros_like(concat_latent)

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = concat_latent * multiplier
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], negative, out_latent)
    
NODE_CLASS_MAPPINGS = {
    "LoadAndApplyICLightUnet": LoadAndApplyICLightUnet,
    "ICLightConditioning": ICLightConditioning
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAndApplyICLightUnet": "Load And Apply IC-Light",
    "ICLightConditioning": "IC-Light Conditioning"
}