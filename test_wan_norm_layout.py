"""Wan normalization must preserve values and expose zero-copy NCHW frames."""

import pytest
import torch

from lightvae.kernels import rms_silu, triton_kernels
from lightvae.utils import frames


@pytest.mark.skipif(triton_kernels is None or not torch.cuda.is_available(),
                    reason='Requires CUDA and Triton')
@pytest.mark.parametrize('shape,view,silu', [
    ((2, 13, 3, 7, 11), 'contiguous', True),
    ((2, 33, 4, 5, 9), 'slice', False),
    ((2, 7, 2, 13, 5), 'transpose', True),
    ((1, 256, 1, 9, 17), 'contiguous', False),
    ((2, 13, 7, 11), 'contiguous', True),
    ((2, 33, 9, 5), 'slice', False),
])
def test_frame_layout_and_per_frame_values(shape, view, silu):
    if torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip('Requires the NVIDIA BF16 fused kernel')
    generator = torch.Generator(device='cuda').manual_seed(2026)
    allocation = (*shape[:-1], shape[-1] * 2) if view == 'slice' else shape
    x = torch.randn(allocation, device='cuda', dtype=torch.bfloat16, generator=generator)
    if view == 'slice':
        x = x[..., ::2]
    elif view == 'transpose':
        x = x.transpose(-1, -2)
    channels = x.shape[1]
    # Strided affine parameters also exercise their contiguous conversion.
    gamma = torch.randn(channels * 2, device='cuda', dtype=x.dtype, generator=generator)[::2]
    bias = torch.randn(channels * 2, device='cuda', dtype=x.dtype, generator=generator)[::2]
    with torch.inference_mode():
        actual = rms_silu(x, gamma, bias, channels ** .5, silu)
        if x.ndim == 5:
            expected = torch.stack([
                torch.cat([
                    rms_silu(x[b:b + 1, :, t], gamma, bias, channels ** .5, silu)
                    for b in range(x.shape[0])
                ], dim=0)
                for t in range(x.shape[2])
            ], dim=2)
            flattened = frames(actual)
            assert flattened.is_contiguous()
            assert flattened.untyped_storage().data_ptr() == actual.untyped_storage().data_ptr()
        else:
            expected = torch.cat([
                rms_silu(x[b:b + 1], gamma, bias, channels ** .5, silu)
                for b in range(x.shape[0])
            ], dim=0)
            assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
