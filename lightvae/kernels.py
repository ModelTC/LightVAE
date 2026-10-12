"""Device dispatch for the factored Wan decoder; PyTorch never expands TLR."""

import torch
from torch.nn import functional as F

try:
    from . import triton_kernels
except ImportError:
    triton_kernels = None


def use_triton(x, minimum_sm, backend):
    return (backend == "auto" and triton_kernels is not None and x.is_cuda
            and torch.version.hip is None and x.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(x.device)[0] >= minimum_sm)


def rms_silu(x, gamma, bias, scale, silu, backend="auto"):
    # The kernel retains native BF16 arithmetic on SM90+ and uses explicitly
    # rounded FP32 operations on Ampere/Ada (which lack those PTX instructions).
    if use_triton(x, 8, backend):
        return triton_kernels.rms_silu(x, gamma, bias, scale, silu)
    value = F.normalize(x.float(), dim=1, eps=1e-12).to(x.dtype)
    value = (value * scale) * gamma + bias
    return F.silu(value) if silu else value


def torch_history_up(p, weight, cache):
    batch, time, _, height, width = p.shape
    co, rank = weight.shape[:2]
    history = p[:, :, co:]
    if cache is None:
        cache = p.new_zeros(batch, 2, 2 * rank, height, width)
    history = torch.cat((cache, history), dim=1)
    q = history[:, 1:time + 1, :rank] + history[:, :time, rank:]
    correction = F.conv2d(q.reshape(batch * time, rank, height, width), weight)
    return correction.reshape(batch, time, co, height, width) + p[:, :, :co]


def fused_history_up(p, weight, cache, backend="auto"):
    # BF16 tensor-core matmul is available on Ampere and newer NVIDIA GPUs.
    if use_triton(p, 8, backend):
        return triton_kernels.fused_history_up(p, weight, cache)
    return torch_history_up(p, weight, cache)
