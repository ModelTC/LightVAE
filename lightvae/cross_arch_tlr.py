"""TLR for native video VAE convolutions with a three-frame temporal kernel.

The operator replaces an inner ``Conv3d``. Its input keeps the surrounding
model's temporal padding and cache rules. ``current_index=2`` uses the final
frame in each convolution window (causal); ``current_index=1`` uses its center
(noncausal). Only the two other temporal slices are factorized. Rank zero is
the spatial-only ``Conv2d`` used in high-resolution decoder stages.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .kernels import fused_tlr_history

def mass_rank(singular_values, eta):
    """Smallest rank retaining eta of the singular-value sum."""
    if not 0 < eta <= 1:
        raise ValueError('eta must be in (0,1]')
    values = singular_values.detach().double().cpu()
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError('Expected finite singular values')
    if values.sum() == 0:
        return 0
    return min(len(values), int(torch.searchsorted(values.cumsum(0), values.sum() * eta)) + 1)



class TLRConv3d(nn.Module):
    """Trainable packed spatial convolution and shared low-rank temporal correction.

    In BF16 inference, a packed ``Conv2d`` computes spatial and history features;
    a Triton kernel fuses history addition, the pointwise up projection, and the
    spatial skip. The PyTorch path remains available for training and FP32.

    The source ``Conv3d`` must have kernel time 3, stride/dilation 1 and groups 1.
    Its own padding is retained; outer VAE padding remains the outer VAE's job.
    Set ``chunk_frames`` for large videos; checkpointing recomputes each chunk
    during backward without changing the operator or temporal boundaries.
    """

    def __init__(self, shape, padding, padding_mode, rank, current_index,
                 *, bias=True, chunk_frames=None, checkpoint_chunks=False, folded=False):
        super().__init__()
        co, ci, kt, kh, kw = shape
        if kt != 3 or current_index not in (1, 2):
            raise ValueError('TLR requires a three-frame kernel and current_index 1 or 2')
        if not 0 <= rank <= min(co, 2 * ci * kh * kw):
            raise ValueError('Invalid temporal rank')
        if len(padding) != 3 or any(p < 0 for p in padding):
            raise ValueError('Expected nonnegative Conv3d padding (time, height, width)')
        if padding_mode not in ('zeros', 'replicate', 'reflect', 'circular'):
            raise ValueError(f'Unsupported Conv3d padding mode: {padding_mode}')
        if chunk_frames is not None and chunk_frames < 1:
            raise ValueError('chunk_frames must be positive')

        self.co = co
        self.rank = rank
        self.current_index = current_index
        self.other_indices = tuple(i for i in range(3) if i != current_index)
        self.source_shape = tuple(shape)
        self.padding = tuple(padding)
        self.padding_mode = padding_mode
        self.chunk_frames = chunk_frames
        self.checkpoint_chunks = checkpoint_chunks
        self.folded = folded
        self.down = nn.Conv2d(ci, co + 2 * rank, (kh, kw), bias=False)
        self.up = nn.Conv2d(rank, co, 1, bias=False) if rank else None
        self.bias = nn.Parameter(torch.zeros(co)) if bias and not folded else None
        packed_bias = torch.empty(co + 2 * rank) if bias and folded else None
        self.register_buffer('packed_bias', packed_bias)
        self.backend = 'auto'

    @classmethod
    @torch.no_grad()
    def from_conv3d(cls, source: nn.Conv3d, eta: float, current_index: int,
                    *, pure2d=False, svd_device=None, chunk_frames=None,
                    checkpoint_chunks=False):
        if (source.kernel_size[0] != 3 or source.stride != (1, 1, 1)
                or source.dilation != (1, 1, 1) or source.groups != 1):
            raise ValueError('Only ungrouped stride-one temporal-kernel-three Conv3d is supported')
        if not 0 < eta <= 1:
            raise ValueError('eta must be in (0,1]')

        weight = source.weight.detach().to(device=svd_device or source.weight.device,
                                           dtype=torch.float32)
        co, ci, _, kh, kw = weight.shape
        other = tuple(i for i in range(3) if i != current_index)
        matrix = torch.cat([weight[:, :, i].reshape(co, -1) for i in other], dim=1)
        singular_values = None
        if pure2d or not matrix.any():
            rank = 0
            a = matrix.new_empty(co, 0)
            b = matrix.new_empty(0, 2 * ci * kh * kw)
        elif eta == 1:
            rank = min(matrix.shape)
            if co <= matrix.shape[1]:
                a, b = torch.eye(co, device=matrix.device), matrix
            else:
                a, b = matrix, torch.eye(matrix.shape[1], device=matrix.device)
        else:
            u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
            rank = mass_rank(singular_values, eta)
            root = singular_values[:rank].sqrt()
            a = u[:, :rank] * root
            b = root[:, None] * vh[:rank]

        op = cls(weight.shape, source.padding, source.padding_mode, rank,
                 current_index, bias=source.bias is not None,
                 chunk_frames=chunk_frames, checkpoint_chunks=checkpoint_chunks)
        op.to(device=source.weight.device, dtype=torch.float32)
        op.down.weight[:co].copy_(weight.sum(dim=2).to(op.down.weight.device))
        if rank:
            factors = b.reshape(rank, 2, ci, kh, kw).transpose(0, 1)
            op.down.weight[co:].copy_(factors.reshape(2 * rank, ci, kh, kw).to(op.down.weight.device))
            op.up.weight.copy_(a[:, :, None, None].to(op.up.weight.device))
        if source.bias is not None:
            op.bias.copy_(source.bias.detach().float().to(op.bias.device))
        op.eta = eta
        op.pure2d = pure2d
        op.retained_sigma = (None if singular_values is None else
                             float(singular_values[:rank].sum().double()
                                   / singular_values.sum().double()))
        return op

    def _pad(self, x):
        pt, ph, pw = self.padding
        if self.padding_mode == 'zeros':
            if pt:
                x = F.pad(x, (0, 0, 0, 0, pt, pt))
            return x, (ph, pw)
        if pt or ph or pw:
            x = F.pad(x, (pw, pw, ph, ph, pt, pt), mode=self.padding_mode)
        return x, (0, 0)

    def _chunk(self, x, spatial_padding):
        batch, channels, time, height, width = x.shape
        output_time = time - 2
        if not self.rank:
            # Spatial-only layers never read the two temporal halo frames.
            # Trim before Conv2d so high-resolution tails do no discarded work.
            x = x[:, :, self.current_index:self.current_index + output_time]
            time = output_time
        flat = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        if self.folded:
            bias = self.packed_bias
        else:
            bias = None if self.bias is None else F.pad(self.bias, (0, 2 * self.rank))
        projected = F.conv2d(flat, self.down.weight, bias, padding=spatial_padding)
        projected = projected.reshape(batch, time, self.co + 2 * self.rank,
                                      *projected.shape[-2:])
        if not self.rank:
            return projected.permute(0, 2, 1, 3, 4)
        current = projected[:, self.current_index:self.current_index + output_time]
        spatial = current[:, :, :self.co]
        if self.folded:
            output = fused_tlr_history(projected, self.up.weight, self.current_index,
                                       self.other_indices, self.backend)
            return output.permute(0, 2, 1, 3, 4)
        r = self.rank
        i, j = self.other_indices
        past_a = projected[:, i:i + output_time, self.co:self.co + r]
        past_b = projected[:, j:j + output_time, self.co + r:]
        if not self.folded:
            past_a = past_a - current[:, :, self.co:self.co + r]
            past_b = past_b - current[:, :, self.co + r:]
        correction = past_a + past_b
        correction = correction.reshape(batch * output_time, r, *spatial.shape[-2:])
        correction = self.up(correction).reshape(batch, output_time, self.co,
                                                 *spatial.shape[-2:])
        return (spatial + correction).permute(0, 2, 1, 3, 4)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != self.source_shape[1]:
            raise ValueError('Expected BCTHW input with the source Conv3d input channels')
        x, spatial_padding = self._pad(x)
        output_time = x.shape[2] - 2
        if output_time < 1:
            raise ValueError('Input shorter than the temporal kernel')
        chunk_size = self.chunk_frames or output_time
        outputs = []
        for start in range(0, output_time, chunk_size):
            stop = min(start + chunk_size, output_time)
            window = x[:, :, start:stop + 2]
            if self.training and self.checkpoint_chunks and torch.is_grad_enabled():
                result = checkpoint(self._chunk, window, spatial_padding,
                                    use_reentrant=False)
            else:
                result = self._chunk(window, spatial_padding)
            outputs.append(result)
        return torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]

    @torch.no_grad()
    def fold(self):
        """Fold current-frame correction into ``down`` for evaluation/export.

        The folded module is an inference artifact; recover the unfused module
        from a training checkpoint before further optimization.
        """
        if self.folded:
            raise RuntimeError('TLR is already folded')
        if self.rank:
            r = self.rank
            factors = self.down.weight[self.co:self.co + r] + self.down.weight[self.co + r:]
            correction = torch.einsum('or,rihw->oihw', self.up.weight[:, :, 0, 0], factors)
            self.down.weight[:self.co].sub_(correction)
        if self.bias is not None:
            self.packed_bias = F.pad(self.bias.detach(), (0, 2 * self.rank)).clone()
            self.bias = None
        self.folded = True
        self.eval().requires_grad_(False)
        return self


def convert_decoder(decoder: nn.Module, paths_or_predicate, eta: float,
                    *, pure2d_paths=(), center_slice=2, svd_device=None,
                    chunk_frames=None, checkpoint_chunks=False, records=None):
    """Replace selected *inner* Conv3d modules in a decoder in place.

    ``paths_or_predicate`` is a list of module paths or ``(path, module) -> bool``.
    ``pure2d_paths`` is a list of paths or a predicate with the same arguments.
    ``center_slice`` is 1 or 2, a path-to-index map, or a two-argument callable.
    With ``records``, construct the recorded architecture without running SVD;
    the caller then loads the saved student state dict. Only paths in ``records``
    are replaced in that mode.
    """
    if records is None:
        if callable(paths_or_predicate):
            selected = [(path, module) for path, module in decoder.named_modules()
                        if paths_or_predicate(path, module)]
        else:
            selected = [(path, decoder.get_submodule(path)) for path in paths_or_predicate]
        if callable(pure2d_paths):
            is_pure2d = pure2d_paths
        else:
            pure2d_set = set(pure2d_paths)
            is_pure2d = lambda path, module: path in pure2d_set
        converted = []
        for path, source in selected:
            if not isinstance(source, nn.Conv3d):
                raise TypeError(f'{path} must be an inner Conv3d, got {type(source).__name__}')
            current = (center_slice(path, source) if callable(center_slice) else
                       center_slice[path] if isinstance(center_slice, dict) else center_slice)
            op = TLRConv3d.from_conv3d(source, eta, current,
                                       pure2d=bool(is_pure2d(path, source)),
                                       svd_device=svd_device, chunk_frames=chunk_frames,
                                       checkpoint_chunks=checkpoint_chunks)
            decoder.set_submodule(path, op)
            converted.append(dict(path=path, rank=op.rank, eta=eta,
                                  pure2d=op.pure2d, current_index=op.current_index,
                                  shape=list(op.source_shape), padding=list(op.padding),
                                  padding_mode=op.padding_mode, bias=op.bias is not None,
                                  retained_sigma=op.retained_sigma, folded=False))
        return converted

    for record in records:
        path = record['path']
        source = decoder.get_submodule(path)
        if not isinstance(source, nn.Conv3d):
            raise TypeError(f'{path} must be an inner Conv3d for checkpoint reconstruction')
        if (list(source.weight.shape) != record['shape']
                or list(source.padding) != record['padding']
                or source.padding_mode != record['padding_mode']
                or (source.bias is not None) != record['bias']):
            raise ValueError(f'{path} differs from the recorded source convolution')
        op = TLRConv3d(record['shape'], record['padding'], record['padding_mode'],
                       record['rank'], record['current_index'], bias=record['bias'],
                       chunk_frames=chunk_frames, checkpoint_chunks=checkpoint_chunks,
                       folded=record.get('folded', False))
        op.to(device=source.weight.device, dtype=torch.float32)
        op.eta = record['eta']
        op.pure2d = record['pure2d']
        op.retained_sigma = record['retained_sigma']
        decoder.set_submodule(path, op)
    return records


def fold_decoder(decoder: nn.Module, records):
    """Fold every recorded TLR after recovery and mark records for exact reload."""
    for record in records:
        decoder.get_submodule(record['path']).fold()
        record['folded'] = True
    return records

