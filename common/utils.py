import torch
import torch.nn.functional as F


from contextlib import contextmanager
from pathlib import Path
import inspect
import torch.distributed as dist

wd = Path('/root/paddlejob/workspace/env_run/SeedVR')

@contextmanager
def tiktok(tag: str, only_rank0: bool = True):
    frame = inspect.currentframe()
    line_number = frame.f_back.f_back.f_lineno
    py_file = Path(inspect.getfile(frame.f_back.f_back))
    py_file = py_file.relative_to(wd).as_posix()
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.synchronize()
    start.record(stream)
    yield
    end.record(stream)
    stream.synchronize()
    cost = start.elapsed_time(end)
    if only_rank0:
        if device == 0:
            print(f'{tag} "{py_file}:{line_number}" | {cost:.5f}ms |\n', end='')
    else:
        print(f'CUDA[{device}] | {tag} | {cost:.5f}ms |\n', end='')


def print_rank(*args, **kwargs):
    device = torch.cuda.current_device()
    print(f'CUDA[{device}]: ', *args, **kwargs)


def print_rank0(*args, **kwargs):
    if torch.cuda.current_device() == 0:
        print(*args, **kwargs)


def print_one_by_one(group: dist.ProcessGroup, *args, **kwargs):
    for rank in range(group.size()):
        if rank == group.rank():
            print(*args, **kwargs)
        dist.barrier(group)


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