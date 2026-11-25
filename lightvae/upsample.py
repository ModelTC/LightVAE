"""Factor nearest-neighbor 2x upsampling and a padded 3x3 convolution."""

import torch
from torch import nn
from torch.nn import functional as F


class PolyphaseUpsample(nn.Module):
    """Compute four output phases on the low-resolution grid."""

    def __init__(self, convolution):
        super().__init__()
        source = convolution.weight.detach()
        packed = []
        for phase_y in range(2):
            for phase_x in range(2):
                weights = torch.zeros((*source.shape[:2], 2, 2), device=source.device, dtype=torch.float64)
                for kernel_y in range(3):
                    for kernel_x in range(3):
                        y = (phase_y + kernel_y - 1) // 2 + 1 - phase_y
                        x = (phase_x + kernel_x - 1) // 2 + 1 - phase_x
                        weights[:, :, y, x] += source[:, :, kernel_y, kernel_x].double()
                packed.append(weights.to(source.dtype))
        self.register_buffer("weight", torch.cat(packed))
        bias = convolution.bias.detach().repeat(4) if convolution.bias is not None else None
        self.register_buffer("bias", bias)
        self.channels = convolution.out_channels

    def forward(self, x):
        batch, _, height, width = x.shape
        y = F.conv2d(x, self.weight, self.bias, padding=1)
        c = self.channels
        phases = (
            y[:, :c, :height, :width],
            y[:, c : 2 * c, :height, 1 : width + 1],
            y[:, 2 * c : 3 * c, 1 : height + 1, :width],
            y[:, 3 * c :, 1 : height + 1, 1 : width + 1],
        )
        return F.pixel_shuffle(torch.stack(phases, dim=2).reshape(batch, 4 * c, height, width), 2)


class TransposedUpsample(nn.Module):
    """Apply the same phase filters with one strided transposed convolution."""

    def __init__(self, packed):
        super().__init__()
        channels = packed.channels
        weights = packed.weight.reshape(4, channels, -1, 2, 2)
        weight = weights.new_empty((weights.shape[2], channels, 4, 4))
        for y in range(2):
            for x in range(2):
                for row in range(2):
                    for col in range(2):
                        weight[:, :, 3-y-2*row, 3-x-2*col] = weights[y*2+x, :, :, row, col].T
        self.register_buffer("weight", weight)
        self.register_buffer("bias", None if packed.bias is None else packed.bias[:channels].clone())

    def forward(self, x):
        return F.conv_transpose2d(x, self.weight, self.bias, stride=2, padding=1)


def replace_lightvae_upsamplers(model, mode):
    """Only replace standalone spatial upsampling pairs with matching geometry."""
    # Snapshot before replacing: set_submodule mutates the traversal.
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, nn.Sequential) or len(layer) != 2:
            continue
        resize, convolution = layer
        if not isinstance(resize, nn.Upsample) or not isinstance(convolution, nn.Conv2d):
            continue
        if resize.mode not in ("nearest", "nearest-exact") or resize.scale_factor not in (2, (2, 2)):
            continue
        if convolution.kernel_size != (3, 3) or convolution.padding != (1, 1):
            continue
        if convolution.stride != (1, 1) or convolution.dilation != (1, 1) or convolution.groups != 1 or convolution.padding_mode != "zeros":
            continue
        replacement = PolyphaseUpsample(convolution)
        if mode == "transpose":
            replacement = TransposedUpsample(replacement)
        model.set_submodule(name, replacement)
