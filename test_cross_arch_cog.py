import copy

import pytest
import torch
from torch.nn import functional as F

from diffusers.models.autoencoders.autoencoder_kl_cogvideox import (
    CogVideoXCausalConv3d, CogVideoXSpatialNorm3D,
)
from lightvae.cross_arch_cog import (
    CogSpatialNormAct, CogTLRPadding, conditioned_silu, packed_causal_history,
)
from lightvae.cross_arch_tlr import TLRConv3d


@pytest.mark.parametrize('rank_zero', [False, True])
@pytest.mark.parametrize('frames', [1, 2, 3])
def test_causal_cache_matches_native_wrapper(rank_zero, frames):
    torch.manual_seed(12)
    source = CogVideoXCausalConv3d(4, 6, 3, pad_mode='first')
    source.conv = TLRConv3d.from_conv3d(source.conv, 1, 2, pure2d=rank_zero).fold()
    adapter = CogTLRPadding(copy.deepcopy(source))
    expected_cache = actual_cache = None
    for _ in range(3):
        x = torch.randn(2, 4, frames, 7, 9)
        expected, expected_cache = source(x, expected_cache)
        actual, actual_cache = adapter(x, actual_cache)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_cache, expected_cache)


@pytest.mark.parametrize('frames', [1, 4, 5])
def test_spatial_norm_preserves_odd_first_frame_and_cache(frames):
    torch.manual_seed(13)
    source = CogVideoXSpatialNorm3D(32, 4)
    adapter = CogSpatialNormAct(copy.deepcopy(source))
    x = torch.randn(1, 32, frames, 7, 9)
    z = torch.randn(1, 4, 1 if frames == 1 else 3, 3, 4)
    expected, cache = source(x, z)
    actual, actual_cache = adapter(x, z)
    torch.testing.assert_close(actual, F.silu(expected), rtol=0, atol=0)
    for key in cache:
        torch.testing.assert_close(actual_cache[key], cache[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA parity test')
def test_bf16_fused_history_and_conditioned_silu():
    torch.manual_seed(14)
    with torch.inference_mode():
        for frames in (1, 2, 5):
            x = torch.randn(2, 32, frames, 7, 9, device='cuda', dtype=torch.bfloat16)
            for history in (None, torch.randn(2, 32, 2, 7, 9, device='cuda', dtype=torch.bfloat16)):
                past = [history] if history is not None else [x[:, :, :1]] * 2
                expected = torch.cat(past + [x], dim=2)
                torch.testing.assert_close(packed_causal_history(x, history), expected, rtol=0, atol=0)
            scale, bias = torch.randn_like(x), torch.randn_like(x)
            expected = F.silu(x * scale + bias)
            actual = conditioned_silu(x, scale, bias)
            torch.testing.assert_close(actual, expected, rtol=.008, atol=.002)
