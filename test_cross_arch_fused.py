"""Regression coverage for large strides without allocating a large tensor."""

import pytest
import torch

from lightvae.cross_arch_fused import fused_norm_act, triton


@pytest.mark.skipif(triton is None or not torch.cuda.is_available(),
                    reason='Requires CUDA and Triton')
@pytest.mark.parametrize('kind', ['hunyuan', 'ltx'])
@pytest.mark.parametrize('shape', [(1, 8, 4, 5), (1, 8, 3, 4, 5)])
def test_singleton_batch_stride_beyond_int32(kind, shape):
    if torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip('Requires the NVIDIA BF16 fused kernel')
    generator = torch.Generator(device='cuda').manual_seed(2026)
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16, generator=generator)
    # This stride occurs in 81-frame HY decoding. Batch size one makes it
    # irrelevant to storage size, so a tiny view reproduces the compiler bug.
    large_stride = x.as_strided(shape, (4_140_564_480, *x.stride()[1:]))
    assert large_stride.untyped_storage().nbytes() == x.untyped_storage().nbytes()
    with torch.inference_mode():
        expected = fused_norm_act(x, kind, eps=1e-6, scale=8 ** .5)
        actual = fused_norm_act(large_stride, kind, eps=1e-6, scale=8 ** .5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
