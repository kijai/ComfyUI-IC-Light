
#credit to huchenlei for this
#from https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/151f7460bbc9d7437d4f0010f21f80178f7a84a6/layered_diffusion.py#L34-L96

import torch
import functools
from comfy.model_patcher import ModelPatcher
import comfy.model_management

def calculate_weight_adjust_channel(func):
    """Patches ComfyUI's LoRA weight application to accept multi-channel inputs."""

    @functools.wraps(func)
    def calculate_weight(patches, weight: torch.Tensor, key: str, intermediate_dtype=torch.float32) -> torch.Tensor:
        weight = func(patches, weight, key, intermediate_dtype)

        for p in patches:
            alpha = p[0]
            v = p[1]

            # The recursion call should be handled in the main func call.
            if isinstance(v, list):
                continue

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]

            if patch_type == "diff":
                w1 = v[0]
                if all(
                    (
                        alpha != 0.0,
                        w1.shape != weight.shape,
                        w1.ndim == weight.ndim == 4,
                    )
                ):
                    new_shape = [max(n, m) for n, m in zip(weight.shape, w1.shape)]
                    print(
                        f"IC-Light: Merged with {key} channel changed from {weight.shape} to {new_shape}"
                    )
                    new_diff = alpha * comfy.model_management.cast_to_device(
                        w1, weight.device, weight.dtype
                    )
                    new_weight = torch.zeros(size=new_shape).to(weight)
                    new_weight[
                        : weight.shape[0],
                        : weight.shape[1],
                        : weight.shape[2],
                        : weight.shape[3],
                    ] = weight
                    new_weight[
                        : new_diff.shape[0],
                        : new_diff.shape[1],
                        : new_diff.shape[2],
                        : new_diff.shape[3],
                    ] += new_diff
                    new_weight = new_weight.contiguous().clone()
                    weight = new_weight
        return weight

    return calculate_weight
