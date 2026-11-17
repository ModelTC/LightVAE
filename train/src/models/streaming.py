"""Wan2.1 causal traversal. Cache tensors retain gradients across chunks."""

import torch
from .wan21 import GatedResidualBlock, ResidualShortcut
from .wan21_native import CausalConv3d, ResidualBlock, Resample


def conv(layer, x, cache, index):
    slot = index[0]
    old = cache[slot]
    history = x[:, :, -2:].clone()
    if history.shape[2] < 2 and old is not None:
        history = torch.cat((old[:, :, -1:].to(x.device), history), 2)
    result = layer(x, old)
    cache[slot] = history
    index[0] += 1
    return result


def apply(layer, x, cache, index, gate=None):
    if isinstance(layer, (ResidualBlock, GatedResidualBlock)):
        shortcut = layer.shortcut(x)
        for sub in layer.residual:
            x = conv(sub, x, cache, index) if isinstance(sub, CausalConv3d) else sub(x)
        return shortcut + (x if gate is None else x * gate.to(x.dtype).view(-1, 1, 1, 1, 1))
    if isinstance(layer, Resample):
        return layer(x, cache, index)
    if isinstance(layer, CausalConv3d):
        return conv(layer, x, cache, index)
    return layer(x)


def layers(component):
    index = -1
    result = []
    for layer in component._layers():
        residual = isinstance(layer, (ResidualBlock, GatedResidualBlock, ResidualShortcut))
        if residual:
            index += 1
        result.append((layer, index if residual else None))
    return result
