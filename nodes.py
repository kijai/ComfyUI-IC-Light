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
    
### Light Source
import numpy as np
from enum import Enum
from nodes import MAX_RESOLUTION

class LightPosition(Enum):
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    TOP_LEFT = "Top Left Light"
    TOP_RIGHT = "Top Right Light"
    BOTTOM_LEFT = "Bottom Left Light"
    BOTTOM_RIGHT = "Bottom Right Light"

def toRgb(color):
    if color.startswith('#') and len(color) == 7:  # e.g. "#RRGGBB"
        color_rgb =tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    else:  # e.g. "255,255,255"
        color_rgb = tuple(int(i) for i in color.split(','))
    return color_rgb

def generate_gradient_image(width:int, height:int, start_color: tuple, end_color: tuple, multiplier: float, lightPosition:LightPosition):
    """
    Generate a gradient image with a light source effect.

    Parameters:
    width (int): Width of the image.
    height (int): Height of the image.
    start_color: Starting color RGB of the gradient.
    end_color: Ending color RGB of the gradient.
    multiplier: Weight of light.
    lightPosition (LightPosition): Position of the light source.

    Returns:
    np.array: 2D gradient image array.
    """
    # Create a gradient from 0 to 1 and apply multiplier
    if lightPosition == LightPosition.LEFT:
        gradient = np.tile(np.linspace(0, 1, width)**multiplier, (height, 1))
    elif lightPosition == LightPosition.RIGHT:
        gradient = np.tile(np.linspace(1, 0, width)**multiplier, (height, 1))
    elif lightPosition == LightPosition.TOP:
        gradient = np.tile(np.linspace(0, 1, height)**multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM:
        gradient = np.tile(np.linspace(1, 0, height)**multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM_RIGHT:
        x = np.linspace(1, 0, width)**multiplier
        y = np.linspace(1, 0, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.BOTTOM_LEFT:
        x = np.linspace(0, 1, width)**multiplier
        y = np.linspace(1, 0, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_RIGHT:
        x = np.linspace(1, 0, width)**multiplier
        y = np.linspace(0, 1, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_LEFT:
        x = np.linspace(0, 1, width)**multiplier
        y = np.linspace(0, 1, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unsupported position. Choose from {', '.join([member.value for member in LightPosition])}.")

    # Interpolate between start_color and end_color based on the gradient
    gradient_img = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(3):
        gradient_img[..., i] = start_color[i] + (end_color[i] - start_color[i]) * gradient
    
    gradient_img = np.clip(gradient_img, 0, 255).astype(np.uint8)
    return gradient_img


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a tensor and scale its values from 0-255 to 0-1."""
    array = array.astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


class LightSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "light_position": ([member.value for member in LightPosition],),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "start_color": ("STRING", {"default": "#FFFFFF"}),
                "end_color": ("STRING", {"default": "#000000"}),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, })
            } 
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "IC-Light"
    DESCRIPTION = """Simple Light Source"""

    def execute(self, light_position, multiplier, start_color, end_color, width, height):
        lightPosition = LightPosition(light_position)
        start_color_rgb = toRgb(start_color)
        end_color_rgb = toRgb(end_color)
        
        image = generate_gradient_image(width, height, start_color_rgb, end_color_rgb, multiplier, lightPosition)
        image = numpy_to_tensor(image)
        return (image,)

    
NODE_CLASS_MAPPINGS = {
    "LoadAndApplyICLightUnet": LoadAndApplyICLightUnet,
    "ICLightConditioning": ICLightConditioning,
    "LightSource": LightSource
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAndApplyICLightUnet": "Load And Apply IC-Light",
    "ICLightConditioning": "IC-Light Conditioning",
    "LightSource": "Simple Light Source"
}