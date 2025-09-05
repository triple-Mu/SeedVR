import torch
import torch.nn.functional as F

def safe_pad_operation(x, padding, mode='constant', value=0.0):
    """Safe padding operation that handles Half precision only for problematic modes"""
    # Modes qui nécessitent le fix Half precision
    problematic_modes = ['replicate', 'reflect', 'circular']
    
    if mode in problematic_modes:
        try:
            return F.pad(x, padding, mode=mode, value=value)
        except RuntimeError as e:
            if "not implemented for 'Half'" in str(e):
                original_dtype = x.dtype
                return F.pad(x.float(), padding, mode=mode, value=value).to(original_dtype)
            else:
                raise e
    else:
        # Pour 'constant' et autres modes compatibles, pas de fix nécessaire
        return F.pad(x, padding, mode=mode, value=value)


def safe_interpolate_operation(x, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    """Safe interpolate operation that handles Half precision for problematic modes"""
    # Modes qui peuvent causer des problèmes avec Half precision
    problematic_modes = ['bilinear', 'bicubic', 'trilinear']
    
    if mode in problematic_modes:
        try:
            return F.interpolate(
                x, 
                size=size, 
                scale_factor=scale_factor, 
                mode=mode, 
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor
            )
        except RuntimeError as e:
            if ("not implemented for 'Half'" in str(e) or 
                "compute_indices_weights" in str(e)):
                original_dtype = x.dtype
                return F.interpolate(
                    x.float(), 
                    size=size, 
                    scale_factor=scale_factor, 
                    mode=mode, 
                    align_corners=align_corners,
                    recompute_scale_factor=recompute_scale_factor
                ).to(original_dtype)
            else:
                raise e
    else:
        # Pour 'nearest' et autres modes compatibles, pas de fix nécessaire
        return F.interpolate(
            x, 
            size=size, 
            scale_factor=scale_factor, 
            mode=mode, 
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor
        )


import pynvml

_HANDLE_: pynvml.c_nvmlDevice_t = None


def current_cuda_mem_used_gb(index: int = None):
    if index is None:
        index = 0
    global _HANDLE_
    if _HANDLE_ is None:
        pynvml.nvmlInit()
        _HANDLE_ = pynvml.nvmlDeviceGetHandleByIndex(index)

    mem_info: pynvml.c_nvmlMemory_t = pynvml.nvmlDeviceGetMemoryInfo(_HANDLE_)
    return mem_info.used / (1 << 30)