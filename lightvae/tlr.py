"""Temporal low-rank convolutions for streaming Wan video decoding."""

import torch
from torch import nn
from torch.nn import functional as F

from utils import frames

from .triton_kernels import fused_history_up


class TLR(nn.Module):
    """Checkpoint factors are folded once on CPU, before conversion to BF16."""

    def __init__(self, shape, padding, rank, bias):
        super().__init__()
        co, ci, kt, kh, kw = shape
        if kt != 3 or not 0 <= rank <= min(co, 2 * ci * kh * kw):
            raise ValueError(f"Invalid TLR shape/rank: {shape}, {rank}")
        self.co, self.rank, self.padding = co, rank, padding
        self.down = nn.Conv2d(ci, co + 2 * rank, (kh, kw), padding=padding, bias=False)
        self.up = nn.Conv2d(rank, co, 1, bias=False) if rank else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(co)) if bias else None
        self.folded = False

    @torch.no_grad()
    def fold(self):
        if self.folded:
            raise RuntimeError("TLR weights have already been folded")
        if self.down.weight.device.type != "cpu" or self.down.weight.dtype != torch.float32:
            raise ValueError("Fold TLR weights on CPU in FP32 before moving the model")
        if self.rank:
            factors = self.down.weight[self.co:]
            correction = torch.einsum(
                "or,rihw->oihw", self.up.weight[:, :, 0, 0],
                factors[:self.rank] + factors[self.rank:],
            )
            self.down.weight[:self.co].sub_(correction)
        bias = None if self.bias is None else F.pad(self.bias.detach(), (0, 2 * self.rank)).clone()
        del self.bias
        self.register_buffer("packed_bias", bias)
        self.folded = True

    def forward(self, x, cache=None):
        batch, _, time, _, _ = x.shape
        projected = F.conv2d(frames(x), self.down.weight, self.packed_bias, padding=self.padding)
        height, width = projected.shape[-2:]
        projected = projected.reshape(batch, time, self.co + 2 * self.rank, height, width)
        if not self.rank:
            return projected.permute(0, 2, 1, 3, 4), None
        output = fused_history_up(projected, self.up.weight, cache)
        z = projected[:, :, self.co:]
        if time >= 2:
            new_cache = z[:, -2:].clone()
        else:
            previous = torch.zeros_like(z) if cache is None else cache[:, -1:]
            new_cache = torch.cat((previous, z), dim=1)
        return output.permute(0, 2, 1, 3, 4), new_cache
