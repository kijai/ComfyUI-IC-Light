import torch
import folder_paths
import os
import types
import numpy as np
import torch.nn.functional as F
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
  
Loads and applies the diffusers SD1.5 IC-Light models available here:  
https://huggingface.co/lllyasviel/ic-light/tree/main  
  
Used with ICLightConditioning -node  
"""

    def load(self, model, model_path):
        type_str = str(type(model.model.model_config).__name__)
        if "SD15" not in type_str:
            raise Exception(f"Attempted to load {type_str} model, IC-Light is only compatible with SD 1.5 models.")

        print("LoadAndApplyICLightUnet: Checking IC-Light Unet path")
        model_full_path = folder_paths.get_full_path("unet", model_path)
        if not os.path.exists(model_full_path):
            raise Exception("Invalid model path")
        else:
            print("LoadAndApplyICLightUnet: Loading IC-Light Unet weights")
            model_clone = model.clone()

            iclight_state_dict = load_torch_file(model_full_path)
            
            print("LoadAndApplyICLightUnet: Attempting to add patches with IC-Light Unet weights")
            try:          
                if 'conv_in.weight' in iclight_state_dict:
                    iclight_state_dict = convert_iclight_unet(iclight_state_dict)
                    in_channels = iclight_state_dict["diffusion_model.input_blocks.0.0.weight"].shape[1]
                    for key in iclight_state_dict:
                        model_clone.add_patches({key: (iclight_state_dict[key],)}, 1.0, 1.0)
                else:
                    for key in iclight_state_dict:
                        model_clone.add_patches({"diffusion_model." + key: (iclight_state_dict[key],)}, 1.0, 1.0)

                    in_channels = iclight_state_dict["input_blocks.0.0.weight"].shape[1]

            except:
                raise Exception("Could not patch model")
            print("LoadAndApplyICLightUnet: Added LoadICLightUnet patches")

            #Patch ComfyUI's LoRA weight application to accept multi-channel inputs. Thanks @huchenlei
            try:
                ModelPatcher.calculate_weight = calculate_weight_adjust_channel(ModelPatcher.calculate_weight)
            except:
                raise Exception("IC-Light: Could not patch calculate_weight")
            # Mimic the existing IP2P class to enable extra_conds
            def bound_extra_conds(self, **kwargs):
                 return ICLight.extra_conds(self, **kwargs)
            new_extra_conds = types.MethodType(bound_extra_conds, model_clone.model)
            model_clone.add_object_patch("extra_conds", new_extra_conds)
            

            model_clone.model.model_config.unet_config["in_channels"] = in_channels        

            return (model_clone, )

import comfy
class ICLight:
    def extra_conds(self, **kwargs):
        out = {}
        
        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        model_in_channels = self.model_config.unet_config['in_channels']
        input_channels = image.shape[1] + 4

        if model_in_channels != input_channels:
            raise Exception(f"Input channels {input_channels} does not match model in_channels {model_in_channels}, 'opt_background' latent input should be used with the IC-Light 'fbc' model, and only with it")
        
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

            repeats_1 = samples_2.size(0) // samples_1.size(0)
            repeats_2 = samples_1.size(0) // samples_2.size(0)
            if samples_1.shape[1:] != samples_2.shape[1:]:
                samples_2 = comfy.utils.common_upscale(samples_2, samples_1.shape[-1], samples_1.shape[-2], "bilinear", "disabled")

            # Repeat the tensors to match the larger batch size
            if repeats_1 > 1:
                samples_1 = samples_1.repeat(repeats_1, 1, 1, 1)
            if repeats_2 > 1:
                samples_2 = samples_2.repeat(repeats_2, 1, 1, 1)

            concat_latent = torch.cat((samples_1, samples_2), dim=1)
        else:
            concat_latent = samples_1

        out_latent = torch.zeros_like(samples_1)

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
                "light_position": ([member.value for member in LightPosition],),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.001}),
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
    DESCRIPTION = """
Generates a gradient image that can be used  
as a simple light source.  The color can be  
specified in RGB or hex format.  
"""

    def execute(self, light_position, multiplier, start_color, end_color, width, height):
        def toRgb(color):
            if color.startswith('#') and len(color) == 7:  # e.g. "#RRGGBB"
                color_rgb =tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            else:  # e.g. "255,255,255"
                color_rgb = tuple(int(i) for i in color.split(','))
            return color_rgb
        lightPosition = LightPosition(light_position)
        start_color_rgb = toRgb(start_color)
        end_color_rgb = toRgb(end_color)
        image = generate_gradient_image(width, height, start_color_rgb, end_color_rgb, multiplier, lightPosition)
        
        image = image.astype(np.float32) / 255.0
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
    
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("normal", "divided",)
    FUNCTION = "execute"
    CATEGORY = "IC-Light"
    DESCRIPTION = """
Calculates normal map from different directional exposures.  
Takes in 4 images as a batch:  
left, right, bottom, top  

"""

    def execute(self, images, sigma, center_input_range, mask=None):
        if center_input_range:
            images = images * 0.5 + 0.5
        if mask is not None:
            if mask.shape[-2:] != images[0].shape[:-1]:
                mask = mask.unsqueeze(0)
                mask = F.interpolate(mask, size=(images.shape[1], images.shape[2]), mode="bilinear")
                mask = mask.squeeze(0)
        
        images_np = images.numpy().astype(np.float32)
        left = images_np[0]
        right = images_np[1]
        bottom = images_np[2]
        top = images_np[3]

        ambient = (left + right + bottom + top) / 4.0
        h, w, _ = ambient.shape
       
        def safe_divide(a, b):
            e = 1e-5
            return ((a + e) / (b + e)) - 1.0

        left = safe_divide(left, ambient)
        right = safe_divide(right, ambient)
        bottom = safe_divide(bottom, ambient)
        top = safe_divide(top, ambient)

        u = (right - left) * 0.5
        v = (top - bottom) * 0.5

        u = np.mean(u, axis=2)
        v = np.mean(v, axis=2)
        h = (1.0 - u ** 2.0 - v ** 2.0).clip(0, 1e5) ** (0.5 * sigma)
        z = np.zeros_like(h)

        normal = np.stack([u, v, h], axis=2)
        normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
        if mask is not None:
            mask = mask.squeeze(0)
            matting = mask.numpy().astype(np.float32)
            matting = matting[..., np.newaxis]
            normal = normal * matting + np.stack([z, z, 1 - z], axis=2) 
            normal = torch.from_numpy(normal)
            normal = normal.unsqueeze(0)
        else:
            normal = normal + np.stack([z, z, 1 - z], axis=2)
            normal = torch.from_numpy(normal).unsqueeze(0)
        
        #normal = F.normalize(normal * 2 - 1, dim=3) / 2 + 0.5
        normal = (normal - normal.min()) / ((normal.max() - normal.min()))
        
        divided = np.stack([left, right, bottom, top])
        divided = torch.from_numpy(divided)
        divided = (divided - divided.min()) / ((divided.max() - divided.min()))
        divided = torch.max(divided, dim=3, keepdim=True)[0].repeat(1, 1, 1, 3)
   
        return (normal, divided, )

class LoadHDRImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": False}),
                     "exposures": ("STRING", {"default": "-2,-1,0,1,2"}),
                     },
                }

    CATEGORY = "IC-Light"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "loadhdrimage"
    DESCRIPTION = """
Loads a .hdr image from the input directory.  
Output is a batch of LDR images with the selected exposures.  

"""
    def loadhdrimage(self, image, exposures):
        import cv2
        image_path = folder_paths.get_annotated_filepath(image)
        # Load the HDR image
        hdr_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

        exposures = list(map(int, exposures.split(",")))
        if not isinstance(exposures, list):
            exposures = [exposures]  # Example exposure values
        ldr_images_tensors = []

        for exposure in exposures:
            # Scale pixel values to simulate different exposures
            ldr_image = np.clip(hdr_image * (2**exposure), 0, 1)
            # Convert to 8-bit image (LDR) by scaling to 255
            ldr_image_8bit = np.uint8(ldr_image * 255)
            # Convert BGR to RGB
            ldr_image_8bit = cv2.cvtColor(ldr_image_8bit, cv2.COLOR_BGR2RGB)
            # Convert the LDR image to a torch tensor
            tensor_image = torch.from_numpy(ldr_image_8bit).float()
            # Normalize the tensor to the range [0, 1]
            tensor_image = tensor_image / 255.0
            # Change the tensor shape to (C, H, W)
            tensor_image = tensor_image.permute(2, 0, 1)
            # Add the tensor to the list
            ldr_images_tensors.append(tensor_image)

        batch_tensors = torch.stack(ldr_images_tensors)
        batch_tensors = batch_tensors.permute(0, 2, 3, 1)

        return batch_tensors,

class BackgroundScaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "scale": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.001}),
                "invert": ("BOOLEAN", { "default": False, }),
            }
        }

    CATEGORY = "IC-Light"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    DESCRIPTION = """
Sets the masked area color in grayscale range.  
"""

    def apply(self, image: torch.Tensor, mask: torch.Tensor, scale: float, invert: bool):

        # Validate inputs
        if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ValueError("image and mask must be torch.Tensor types.")
        if image.ndim != 4 or mask.ndim not in [3, 4]:
            raise ValueError("image must be a 4D tensor, and mask must be a 3D or 4D tensor.")

        # Adjust mask dimensions if necessary
        if mask.ndim == 3:
            # [B, H, W] => [B, H, W, C=1]
            mask = mask.unsqueeze(-1)

        if invert:
            mask = 1 - mask
        image_out = image * mask + (1 - mask) * scale
        image_out = torch.clamp(image_out, 0, 1).cpu().float()
        
        return (image_out,)
            
NODE_CLASS_MAPPINGS = {
    "LoadAndApplyICLightUnet": LoadAndApplyICLightUnet,
    "ICLightConditioning": ICLightConditioning,
    "LightSource": LightSource,
    "CalculateNormalsFromImages": CalculateNormalsFromImages,
    "LoadHDRImage": LoadHDRImage,
    "BackgroundScaler": BackgroundScaler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAndApplyICLightUnet": "Load And Apply IC-Light",
    "ICLightConditioning": "IC-Light Conditioning",
    "LightSource": "Simple Light Source",
    "CalculateNormalsFromImages": "Calculate Normals From Images",
    "LoadHDRImage": "Load HDR Image",
    "BackgroundScaler": "Background Scaler"
}