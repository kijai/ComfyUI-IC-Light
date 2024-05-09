import torch
import folder_paths
import os
import types
from comfy.utils import load_torch_file
from comfy.model_base import IP2P

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
        print("LoadICLightUnet: Checking LoadICLightUnet path")
        model_full_path = folder_paths.get_full_path("unet", model_path)
        if not os.path.exists(model_full_path):
            raise Exception("Invalid model path")
        else:
            print("LoadICLightUnet: Loading LoadICLightUnet weights")
            model_clone = model.clone()

            conv_layer = model_clone.model.diffusion_model.input_blocks[0][0]
            print(f"Current number of input channels: {conv_layer.in_channels}")
            
            # Create a new Conv2d layer with 8 or 12 input channels
            if not "fbc" in model_path:
                in_channels = 8
            else:
                in_channels = 12

            new_conv_layer = torch.nn.Conv2d(in_channels, conv_layer.out_channels, kernel_size=conv_layer.kernel_size, stride=conv_layer.stride, padding=conv_layer.padding)
            new_conv_layer.weight.zero_()
            new_conv_layer.weight[:, :4, :, :].copy_(conv_layer.weight)
            new_conv_layer.bias = conv_layer.bias
            new_conv_layer = new_conv_layer.to(model_clone.model.diffusion_model.dtype)
            conv_layer.conv_in = new_conv_layer
            # Replace the old layer with the new one
            model_clone.model.diffusion_model.input_blocks[0][0] = new_conv_layer
            # Verify the change
            print(f"New number of input channels: {model_clone.model.diffusion_model.input_blocks[0][0].in_channels}")
            
            # Monkey patch because I don't know what I'm doing
            # Dynamically add the extra_conds method from IP2P to the instance of BaseModel
            def bound_extra_conds(self, **kwargs):
                return IP2P.extra_conds(self, **kwargs)
            model_clone.model.process_ip2p_image_in = lambda image: image
            model_clone.model.extra_conds = types.MethodType(bound_extra_conds, model_clone.model)
           
            # Some Proper patching (I hope)
            new_state_dict = load_torch_file(model_full_path)
            prefix_to_remove = 'model.'
            new_keys_dict = {key[len(prefix_to_remove):]: new_state_dict[key] for key in new_state_dict if key.startswith(prefix_to_remove)}

            print("LoadICLightUnet: Attempting to add patches with LoadICLightUnet weights")
            try:
                for key in new_keys_dict:
                    model_clone.add_patches({key: (new_keys_dict[key],)}, 1.0, 1.0)
            except:
                raise Exception("Could not patch model")
            print("LoadICLightUnet: Added LoadICLightUnet patches")

            return (model_clone, )
        
NODE_CLASS_MAPPINGS = {
    "LoadAndApplyICLightUnet": LoadAndApplyICLightUnet
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAndApplyICLightUnet": "Load And Apply IC-Light"
}