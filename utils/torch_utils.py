import torch
import numpy as np
from contextlib import contextmanager
from fvcore.nn import FlopCountAnalysis, flop_count_table

def format_size(bytes, suffix="B"):
    """Format bytes to human readable string."""
    factor = 1024
    for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def track_memory(name):
    if torch.cuda.is_available():
        torch.cuda.synchronize() # 等待所有异步操作完成以获得准确读数
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        print(f"[{name}] Allocated: {format_size(allocated)} | Reserved: {format_size(reserved)} | Peak: {format_size(peak)}")
    else:
        print(f"[{name}] CUDA not available")

def _mb(x):  # bytes -> MiB
    return x / (1024 ** 2)

@contextmanager
def cuda_mem_stage(stats: dict, name: str, sync: bool = True):
    if not torch.cuda.is_available():
        yield
        return

    if sync:
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    alloc0 = torch.cuda.memory_allocated()
    rsv0   = torch.cuda.memory_reserved()

    yield

    if sync:
        torch.cuda.synchronize()

    alloc1 = torch.cuda.memory_allocated()
    rsv1   = torch.cuda.memory_reserved()
    peak   = torch.cuda.max_memory_allocated()

    stats[name] = {
        "alloc_MiB": _mb(alloc1),
        "reserved_MiB": _mb(rsv1),
        "delta_alloc_MiB": _mb(alloc1 - alloc0),
        "delta_reserved_MiB": _mb(rsv1 - rsv0),
        "peak_alloc_MiB": _mb(peak),
        "peak_over_start_MiB": _mb(peak - alloc0),
    }


def print_memory_stats(stats: dict):
    print("\n===== CUDA Memory Stats (MiB) =====")
    for k, v in stats.items():
        print(
            f"{k:18s} | "
            f"alloc={v['alloc_MiB']:.1f} | reserved={v['reserved_MiB']:.1f} | "
            f"d_alloc={v['delta_alloc_MiB']:.1f} | d_reserved={v['delta_reserved_MiB']:.1f} | "
            f"peak={v['peak_alloc_MiB']:.1f} | peak_over_start={v['peak_over_start_MiB']:.1f}"
        )
    print("==================================\n")

def _to_device_float_tree(x, device="cuda"):
    if torch.is_tensor(x):
        return x.float().to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device_float_tree(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ _to_device_float_tree(v, device) for v in x ]
    return x

def sanitize_jit_input(x):
    """
    递归把 fvcore/JIT 不支持的类型（尤其 numpy 标量）转换成可 trace 的类型：
    - numpy 标量 -> Python 标量（int/float/bool）
    - numpy 数组 -> torch.Tensor
    - 其它保持不变
    """
    # torch tensor
    if torch.is_tensor(x):
        return x

    # numpy scalar -> python scalar
    if isinstance(x, np.generic):
        return x.item()

    # numpy array -> torch tensor
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)

    # dict
    if isinstance(x, dict):
        return {k: sanitize_jit_input(v) for k, v in x.items()}

    # list/tuple
    if isinstance(x, (list, tuple)):
        t = [sanitize_jit_input(v) for v in x]
        return type(x)(t) if isinstance(x, tuple) else t

    # python scalar / None / str 等：JIT 支持 dict 里带 str key，但 value 里尽量别放太复杂
    return x

@torch.no_grad()
def measure_depthnet_flops(depth_net, sample_inputs, device="cuda"):
    depth_net = depth_net.to(device).eval()
    inputs = sanitize_jit_input(sample_inputs)
    inputs = _to_device_float_tree(sample_inputs, device=device)
    flops = FlopCountAnalysis(depth_net, (inputs,))
    total = flops.total()

    print(f"Total FLOPs: {total:,}")
    print(flop_count_table(flops, max_depth=4))

    # 看看有没有不支持的 op（Transformer/attention 经常在这）
    try:
        unsup = flops.unsupported_ops()
        if len(unsup) > 0:
            print("\n[WARN] Unsupported ops (FLOPs not counted):")
            for k, v in sorted(unsup.items(), key=lambda kv: -kv[1]):
                print(f"  {k}: {v}")
    except Exception:
        pass

    return total

def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size