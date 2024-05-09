import torch
import folder_paths
import os
import types
from comfy.utils import load_torch_file
from comfy.model_base import IP2P
from .utils.convert_unet import convert_iclight_unet

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
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
LoadICLightUnet: Loads an ICLightUnet model. (Experimental)
WORK IN PROGRESS  
Very hacky (but currently working) way to load the converted IC-Light model available here:  
https://huggingface.co/Kijai/iclight-comfy/blob/main/iclight_fc_converted.safetensors  

Used with InstructPixToPixConditioning -node

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
            try:
                for key in new_keys_dict:
                    model_clone.add_patches({key: (new_keys_dict[key],)}, 1.0, 1.0)
            except:
                raise Exception("Could not patch model")
            print("LoadAndApplyICLightUnet: Added LoadICLightUnet patches")

            # Create a new Conv2d layer with 8 or 12 input channels
            original_conv_layer = model_clone.model.diffusion_model.input_blocks[0][0]
            print(f"LoadAndApplyICLightUnet: Input channels in currently loaded model: {original_conv_layer.in_channels}")
          
            print("LoadAndApplyICLightUnet: Settings in_channels to: ", in_channels)
            
            if model_clone.model.diffusion_model.input_blocks[0][0].in_channels != in_channels:
                new_conv_layer = torch.nn.Conv2d(in_channels, original_conv_layer.out_channels, kernel_size=original_conv_layer.kernel_size, stride=original_conv_layer.stride, padding=original_conv_layer.padding)
                new_conv_layer.weight.zero_()
                new_conv_layer.weight[:, :original_conv_layer.in_channels, :, :].copy_(original_conv_layer.weight)
                new_conv_layer.bias = original_conv_layer.bias
                new_conv_layer = new_conv_layer.to(model_clone.model.diffusion_model.dtype)
                original_conv_layer.conv_in = new_conv_layer
                # Replace the old layer with the new one
                model_clone.model.diffusion_model.input_blocks[0][0] = new_conv_layer
                # Verify the change
                print(f"LoadAndApplyICLightUnet: New number of input channels: {model_clone.model.diffusion_model.input_blocks[0][0].in_channels}")
            
            # Monkey patch because I don't know what I'm doing
            # Dynamically add the extra_conds method from IP2P to the instance of BaseModel
            def bound_extra_conds(self, **kwargs):
                return ICLight.extra_conds(self, **kwargs)
            #model_clone.model.process_ip2p_image_in = lambda image: image
            model_clone.model.extra_conds = types.MethodType(bound_extra_conds, model_clone.model)
           
           

            return (model_clone, )

import comfy
class ICLight:
    def extra_conds(self, **kwargs):
        out = {}

        process_ip2p_image_in = lambda image: image

        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros_like(noise)

        if image.shape[1:] != noise.shape[1:]:
            image = comfy.utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")

        image = comfy.utils.resize_to_batch_size(image, noise.shape[0])

        out['c_concat'] = comfy.conds.CONDNoiseShape(process_ip2p_image_in(image))
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = comfy.conds.CONDRegular(adm)
        return out
    
NODE_CLASS_MAPPINGS = {
    "LoadAndApplyICLightUnet": LoadAndApplyICLightUnet
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAndApplyICLightUnet": "Load And Apply IC-Light"
}