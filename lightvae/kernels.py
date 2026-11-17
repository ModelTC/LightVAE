"""BF16 kernel dispatch for Wan and cross-architecture TLR decoders."""

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


def fused_tlr_history(p, weight, current_index, other_indices, backend="auto"):
    """Fuse cross-architecture temporal history projection for BF16 inference."""
    if use_triton(p, 8, backend):
        return triton_kernels.fused_tlr_history(
            p, weight, current_index, other_indices)

    batch, input_time, _, height, width = p.shape
    output_time = input_time - 2
    co, rank = weight.shape[:2]
    base = p[:, current_index:current_index + output_time, :co]
    first, second = other_indices
    left = p[:, first:first + output_time, co:co + rank]
    right = p[:, second:second + output_time, co + rank:co + 2 * rank]
    history = (left + right).reshape(batch * output_time, rank, height, width)
    correction = F.conv2d(history, weight)
    correction = correction.reshape(batch, output_time, co, height, width)
    return base + correction
