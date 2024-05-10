import torch
import folder_paths
import os
import types
import numpy as np
from comfy.utils import load_torch_file
from .utils.convert_unet import convert_iclight_unet
from .utils.patches import calculate_weight_adjust_channel
from .utils.image import generate_gradient_image, LightPosition
from nodes import MAX_RESOLUTION
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
            # for key, value in iclight_state_dict.items():
            #     if key.startswith('conv_in.weight'):
            #         in_channels = value.shape[1]
            #         break

            # Add weights as patches
            new_keys_dict = convert_iclight_unet(iclight_state_dict)

            print("LoadAndApplyICLightUnet: Attempting to add patches with IC-Light Unet weights")
            #model_clone.unpatch_model()
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

        out_latent = torch.zeros_like(samples_1)
        print(out_latent.shape)

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = concat_latent * multiplier
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], {"samples": out_latent})

class LightSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "light_position": (["Left Light", "Right Light", "Top Light", "Bottom Light",'Top Left Light', 'Top Right Light', 'Bottom Left Light', 'Bottom Right Light'],),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "multiplier": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, }),
                "color": ("STRING", {"default": "#FFFFFF"})
            } 
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "IC-Light"
    DESCRIPTION = """
Generates a gradient image that can be used  
as a simple light source.  The color can be  
specified in RGB or hex format.  
"""

    def execute(self, width, height, light_position, multiplier, color):
        if color.startswith('#') and len(color) == 7:  # e.g. "#RRGGBB"
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
        else:
            r, g, b = map(int, color.split(','))
        
        lightPosition = LightPosition(light_position)
        image = generate_gradient_image(width, height, lightPosition)
        image = image * [r / 255.0, g / 255.0, b / 255.0]
        # Convert a numpy array to a tensor and scale its values from 0-255 to 0-1
        image = image.astype(np.float32) / 255.0
        image = image * multiplier
        image = torch.from_numpy(image)[None,]
        return (image,)

class CalculateNormalsFromImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "sigma": ("FLOAT", { "default": 10.0, "min": 0.01, "max": 100.0, "step": 0.01, }),
                "center_input_range": ("BOOLEAN", { "default": False, }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal", )
    FUNCTION = "execute"
    CATEGORY = "IC-Light"
    DESCRIPTION = """
Calculates normal map from different directional exposures.  
Takes in 4 images as a batch:  
left, right, bottom, top  

"""

    def execute(self, images, sigma, center_input_range, mask=None):
        print(images.min(), images.max())
        if center_input_range:
            images = images * 0.5 + 0.5
        images_np = images.numpy().astype(np.float32)
        left = images_np[0]
        right = images_np[1]
        bottom = images_np[2]
        top = images_np[3]

        ambient = (left + right + bottom + top) / 4.0
        h, w, _ = ambient.shape
       
        def safa_divide(a, b):
            e = 1e-5
            return ((a + e) / (b + e)) - 1.0

        left = safa_divide(left, ambient)
        right = safa_divide(right, ambient)
        bottom = safa_divide(bottom, ambient)
        top = safa_divide(top, ambient)

        u = (right - left) * 0.5
        v = (top - bottom) * 0.5

        u = np.mean(u, axis=2)
        v = np.mean(v, axis=2)
        h = (1.0 - u ** 2.0 - v ** 2.0).clip(0, 1e5) ** (0.5 * sigma)
        z = np.zeros_like(h)

        normal = np.stack([u, v, h], axis=2)
        normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
        if mask is not None:
            matting = mask.numpy().astype(np.float32)
            matting = matting[..., np.newaxis] 
            normal = normal * matting + np.stack([z, z, 1 - z], axis=2) * (1 - matting)
            normal = torch.from_numpy(normal)
        else:
            normal = normal + np.stack([z, z, 1 - z], axis=2)
            normal = torch.from_numpy(normal).unsqueeze(0)

        print(normal.min(), normal.max())
        normal = (normal + 1.0) / 2.0
        normal = torch.clamp(normal, 0, 1)
        print(normal.min(), normal.max())
   
        return (normal,)
        
NODE_CLASS_MAPPINGS = {
    "LoadAndApplyICLightUnet": LoadAndApplyICLightUnet,
    "ICLightConditioning": ICLightConditioning,
    "LightSource": LightSource,
    "CalculateNormalsFromImages": CalculateNormalsFromImages
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAndApplyICLightUnet": "Load And Apply IC-Light",
    "ICLightConditioning": "IC-Light Conditioning",
    "LightSource": "Simple Light Source",
    "CalculateNormalsFromImages": "Calculate Normals From Images"
}