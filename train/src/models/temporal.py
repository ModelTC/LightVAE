"""Autograd-safe streaming packed temporal differences; dense spatial branch.

W0*x[t-2]+W1*x[t-1]+W2*x[t] = S*x[t] + A*(B1*dx1+B2*dx2),
S=sum(W), [W1,W0] ~= A@[B1,B2]. The two history frames are projected
features, not raw input. No detach across chunks during recovery.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .wan21_native import CausalConv3d, Resample, ResidualBlock
from .wan21 import GatedResidualBlock, SearchLoRAConv3d, WanPrunedDecoder
from .streaming import layers, apply as original_apply


def frames(x):
    return x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], *x.shape[-2:])


def unframes(x, b, t):
    return x.reshape(b, t, *x.shape[1:]).permute(0, 2, 1, 3, 4)


def mass_rank(s, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    s = s.detach().double().cpu()
    if s.ndim != 1 or not torch.isfinite(s).all() or (s < 0).any() or (s[1:] > s[:-1]).any():
        raise ValueError("Invalid singular values")
    if s.sum() == 0:
        return 0
    return min(len(s), int(torch.searchsorted(s.cumsum(0), s.sum() * fraction)) + 1)


class PackedTemporalConv(nn.Module):
    def __init__(self, shape, padding, rank, bias):
        super().__init__()
        co, ci, kt, kh, kw = shape
        if kt != 3 or not 0 <= rank <= min(co, 2 * ci * kh * kw):
            raise ValueError("Invalid temporal factorization")
        self.co, self.rank, self.padding = co, rank, padding
        self.down = nn.Conv2d(ci, co + 2 * rank, (kh, kw), padding=padding, bias=False)
        self.up = nn.Conv2d(rank, co, 1, bias=False) if rank else nn.Identity()
        # Temporal projection MUST remain bias-free (including its zero history).
        self.bias = nn.Parameter(torch.zeros(co)) if bias else None

    def forward(self, x, cache=None):
        b, _, t, _, _ = x.shape
        bias = None if self.bias is None else F.pad(self.bias, (0, 2 * self.rank))
        z = unframes(F.conv2d(frames(x), self.down.weight, bias, padding=self.padding), b, t)
        spatial, temporal = z[:, : self.co], z[:, self.co :]
        if not self.rank:
            return spatial, None
        if cache is None:
            cache = temporal.new_zeros((b, 2 * self.rank, 2, *temporal.shape[-2:]))
        history = torch.cat((cache, temporal), 2)
        r = self.rank
        delta = history[:, :r, 1 : t + 1] - temporal[:, :r] + history[:, r:, :t] - temporal[:, r:]
        return spatial + unframes(self.up(frames(delta)), b, t), history[:, :, -2:].clone()


@torch.no_grad()
def convert(component, fraction=0.6, records=None):
    """Convert merged dense weights; records-only mode reconstructs checkpoint shapes."""
    result = []
    candidates = [
        (n, m) for n, m in component.decoder.named_modules() if isinstance(m, CausalConv3d) and m.kernel_size[0] == 3
    ]
    if records is not None and {r["path"] for r in records} != {n for n, _ in candidates}:
        raise ValueError("Packed checkpoint operator paths differ")
    for name, layer in candidates:
        if isinstance(layer, SearchLoRAConv3d):
            raise ValueError("Merge LoRA BEFORE temporal truncation")
        if layer.stride != (1, 1, 1) or layer.dilation != (1, 1, 1) or layer.groups != 1:
            raise ValueError(f"Unsupported convolution: {name}")
        shape = list(layer.weight.shape)
        co, ci, _, kh, kw = shape
        if records is None:
            w = layer.weight.float()
            n = torch.cat((w[:, :, 1].reshape(co, -1), w[:, :, 0].reshape(co, -1)), 1)
            u, s, vh = torch.linalg.svd(n, full_matrices=False)
            rank = mass_rank(s, fraction)
            rec = dict(
                path=name,
                shape=shape,
                padding=layer._padding[0],
                rank=rank,
                bias=layer.bias is not None,
                fraction=fraction,
                retained_sum=float(s[:rank].double().sum() / s.double().sum()) if rank else 1.0,
                singular_values=s.cpu().tolist(),
                spatial_truncation=False,
            )
        else:
            rec = next(r for r in records if r["path"] == name)
            if rec["shape"] != shape or rec["padding"] != layer._padding[0]:
                raise ValueError("Packed checkpoint shape/padding differs")
            rank = rec["rank"]
        op = PackedTemporalConv(shape, rec["padding"], rank, rec["bias"]).to(layer.weight.device)
        if records is None:
            weights = [w.sum(2)]
            if rank:
                root = s[:rank].sqrt()
                a = u[:, :rank] * root
                b = (root[:, None] * vh[:rank]).reshape(rank, 2, ci, kh, kw)
                weights.extend((b[:, 0], b[:, 1]))
                op.up.weight.copy_(a[:, :, None, None])
            op.down.weight.copy_(torch.cat(weights))
            if op.bias is not None:
                op.bias.copy_(layer.bias)
        component.decoder.set_submodule(name, op)
        result.append(rec)
    return result


def cached(layer, x, cache, slot):
    if isinstance(layer, PackedTemporalConv):
        i = slot[0]
        x, cache[i] = layer(x, cache[i])
        slot[0] += 1
        return x
    return original_apply(layer, x, cache, slot)


def apply(layer, x, cache, slot, gate=None):
    if isinstance(layer, (ResidualBlock, GatedResidualBlock)):
        shortcut = layer.shortcut(x)
        for sublayer in layer.residual:
            x = cached(sublayer, x, cache, slot)
        return shortcut + (x if gate is None else x * gate.to(x.dtype).view(-1, 1, 1, 1, 1))
    if isinstance(layer, Resample) and layer.mode == "upsample3d" and isinstance(layer.time_conv, PackedTemporalConv):
        b, c, t, h, w = x.shape
        i = slot[0]
        old = cache[i]
        if old is None:
            cache[i] = "startup"
        else:
            x, new = layer.time_conv(x, None if isinstance(old, str) else old)
            cache[i] = new if new is not None else "rank_zero"
            x = x.reshape(b, 2, c, t, h, w).permute(0, 2, 3, 1, 4, 5).reshape(b, c, t * 2, h, w)
        slot[0] += 1
        return unframes(layer.resample(frames(x)), b, x.shape[2])
    return cached(layer, x, cache, slot)


def decode(component, z, anchor=None, checkpoint_chunks=False):
    projected = component.conv2(component.denormalize_latents(z))
    sequence = layers(component)
    mask = component._search_mask(z.shape[0]) if component.search_config is not None else None
    cache = [None] * 256
    outputs = []
    features = []

    def run(x, gates, *history):
        history = list(history)
        slot = [0]
        feature = None
        selected = []
        for layer, index in sequence:
            gate = gates[:, index] if gates is not None and index is not None else None
            x = apply(layer, x, history, slot, gate)
            if index == anchor and anchor is not None:
                feature = x
            if isinstance(anchor, tuple) and index in anchor:
                selected.append(x)
        if isinstance(anchor, tuple):
            feature = tuple(selected)
        return x, feature, *history

    for i in range(projected.shape[2]):
        args = (projected[:, :, i : i + 1], mask, *cache)
        if checkpoint_chunks and torch.is_grad_enabled():
            x, h, *cache = checkpoint(run, *args, use_reentrant=False)
        else:
            x, h, *cache = run(*args)
        outputs.append(x)
        if h is not None:
            features.append(h)
    return outputs, features


class Student(nn.Module):
    def __init__(self, component):
        super().__init__()
        self.component = component

    def forward(self, z, anchor=None):
        return decode(self.component, z, anchor, checkpoint_chunks=True)


@torch.no_grad()
def merge_search(saved, cfg):
    search = WanPrunedDecoder(search=cfg["search"])
    search.load_state_dict(saved["model"], strict=True)
    search.gate_logits.copy_(saved["gate_ema"])
    search.eval().requires_grad_(False)
    kept = search.selected_layers()
    if kept != saved["kept"]:
        raise ValueError("Saved EMA selection differs")
    fixed = WanPrunedDecoder(kept_residual_indices=kept)
    original = search.state_dict()
    state = {k: original[k].clone() for k in fixed.state_dict()}
    records = []
    for name, layer in search.named_modules():
        if isinstance(layer, SearchLoRAConv3d) and name + ".weight" in state:
            delta = torch.einsum("or,rithw->oithw", layer.lora_B.float()[:, :, 0, 0, 0], layer.lora_A.float())
            state[name + ".weight"] = layer.weight.float() + layer.scale * delta
            records.append(dict(path=name, scale=layer.scale, rank=layer.lora_A.shape[0]))
    assert len(records) == 2 * len(kept)
    fixed.load_state_dict(state, strict=True)
    return search, fixed.eval().requires_grad_(False), records
