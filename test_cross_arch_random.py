import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lightvae import cross_arch_random as random_models
from lightvae.cross_arch_tlr import TLRConv3d


class ToyNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(1, 4, 1, 1, 1))
        self.bias = nn.Parameter(torch.empty(1, 4, 1, 1, 1))
        self.register_buffer('std-of-means', torch.empty(4))
        self.register_buffer('mean-of-means', torch.empty(4))

    def forward(self, x):
        return x * self.gamma * self.get_buffer('std-of-means').view(1, -1, 1, 1, 1) + self.bias


class ToyVAE:
    def __init__(self):
        self.family = 'hunyuanvideo15'
        self.model = nn.Module()
        self.model.encoder = nn.Linear(2, 2)
        self.model.decoder = nn.Sequential()
        self.model.decoder.add_module('conv', nn.Conv3d(2, 4, (3, 1, 1)))
        self.model.decoder.add_module('norm', ToyNorm())

    @property
    def decoder(self):
        return self.model.decoder

    @property
    def eligible_decoder_convs(self):
        return [('decoder.conv', self.decoder.conv, False)]


@pytest.fixture
def structure(tmp_path):
    data = dict(schema_version=1, family='hunyuanvideo15', config={}, variants={})
    for eta, rank in ((.4, 2), (.6, 3)):
        record = dict(path='conv', rank=rank, eta=eta, pure2d=False, current_index=2,
                      shape=[4, 2, 3, 1, 1], padding=[0, 0, 0], padding_mode='zeros',
                      bias=True, retained_sigma=eta, folded=False)
        data['variants'][str(eta)] = dict(eta=eta, step=100, tlr_records=[record])
    path = tmp_path / 'hunyuanvideo15.json'
    path.write_text(json.dumps(data))
    return tmp_path, path, data


@pytest.mark.parametrize('variant,rank', [('base', None), ('0.4', 2), ('0.6', 3)])
def test_random_construction_uses_exact_records_without_weights_or_svd(structure, monkeypatch,
                                                                     variant, rank):
    directory, _, _ = structure
    optimized = []

    def native(*args):
        with torch.device('meta'):
            vae = ToyVAE()
        assert all(p.is_meta for p in vae.model.parameters())
        return vae

    def forbidden(*args, **kwargs):
        raise AssertionError('Random benchmarks must not read checkpoints or run SVD')

    monkeypatch.setattr(random_models, '_native_vae', native)
    monkeypatch.setattr(random_models, 'optimize_tlr', lambda vae: optimized.append(vae))
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.linalg, 'svd', forbidden)
    vae, info = random_models.load_random_variant('hunyuanvideo15', directory / 'no-weights',
                                                  directory, variant, dtype=torch.float32, seed=12)
    assert vae.model.encoder.weight.is_meta
    assert all(not p.is_meta and torch.isfinite(p).all() for p in vae.decoder.parameters())
    assert torch.equal(vae.decoder.norm.gamma, torch.ones_like(vae.decoder.norm.gamma))
    assert torch.equal(vae.decoder.norm.bias, torch.zeros_like(vae.decoder.norm.bias))
    assert torch.equal(vae.decoder.norm.get_buffer('std-of-means'), torch.ones(4))
    assert torch.equal(vae.decoder.norm.get_buffer('mean-of-means'), torch.zeros(4))
    output = vae.decoder(torch.randn(1, 2, 5, 2, 2))
    assert output.shape == (1, 4, 3, 2, 2) and torch.isfinite(output).all()
    assert output.abs().sum() > 0
    assert info['random_weights'] and len(info['structure_sha256']) == 64
    assert bool(optimized) is (variant != 'base')
    if rank is None:
        assert isinstance(vae.decoder.conv, nn.Conv3d)
    else:
        assert isinstance(vae.decoder.conv, TLRConv3d)
        assert vae.decoder.conv.rank == rank
        assert vae.decoder.conv.folded
        assert vae.decoder.conv.bias is None
        assert not vae.decoder.conv.packed_bias.any()
        assert info['tlr_ranks'] == {'conv': rank}
        assert info['eta'] == float(variant)

    duplicate, _ = random_models.load_random_variant('hunyuanvideo15', directory, directory,
                                                     variant, dtype=torch.float32, seed=12)
    for key, value in vae.decoder.state_dict().items():
        torch.testing.assert_close(value, duplicate.decoder.state_dict()[key], rtol=0, atol=0)
    bf16, _ = random_models.load_random_variant('hunyuanvideo15', directory, directory,
                                                variant, dtype=torch.bfloat16, seed=12)
    assert all(p.dtype == torch.bfloat16 for p in bf16.decoder.parameters())


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'eta', 'rank', 'shape'])
def test_random_construction_rejects_inexact_architecture(structure, monkeypatch, problem):
    directory, path, data = structure
    rows = data['variants']['0.4']['tlr_records']
    if problem == 'duplicate':
        rows.append(dict(rows[0]))
    elif problem == 'missing':
        rows.clear()
    elif problem == 'eta':
        rows[0]['eta'] = .6
    elif problem == 'rank':
        rows[0]['rank'] = 2.5
    else:
        rows[0]['shape'][0] = 5
    path.write_text(json.dumps(data))
    monkeypatch.setattr(random_models, '_native_vae', lambda *args: ToyVAE())
    with pytest.raises(ValueError):
        random_models.load_random_variant('hunyuanvideo15', directory, directory, '0.4')


def test_native_factories_construct_meta_architecture_without_reading_weights(monkeypatch, tmp_path):
    calls = []

    def from_config(config):
        assert torch.empty(0).is_meta
        calls.append(('hy', config))
        return ToyVAE().model

    def model_class(family):
        assert not torch.empty(0).is_meta, 'Source imports must run outside the meta context'
        return SimpleNamespace(from_config=from_config)

    monkeypatch.setattr(random_models, '_model_class', model_class)

    def decoder_from_metadata(config):
        assert torch.empty(0).is_meta
        calls.append(('ltx', config))
        return nn.Linear(2, 2)

    def official_ltx(root):
        assert not torch.empty(0).is_meta, 'Source imports must run outside the meta context'
        return None, nn.Conv3d, None, SimpleNamespace(from_metadata=decoder_from_metadata), None, None

    monkeypatch.setattr(random_models, '_official_ltx', official_ltx)
    hy = random_models._native_vae('hunyuanvideo15', {'native': True}, tmp_path, tmp_path / 'hy.json')
    ltx = random_models._native_vae('ltx25', {'metadata': True}, tmp_path, tmp_path / 'ltx.json')
    assert calls == [('hy', {'native': True}), ('ltx', {'metadata': True})]
    assert hy.model.encoder.weight.is_meta and hy.model.decoder.conv.weight.is_meta
    assert not hy.model.cpu_offload
    assert list(ltx.model._modules) == ['decoder']
    assert ltx.decoder.weight.is_meta


@pytest.mark.parametrize('random_weights', [False, True])
@pytest.mark.parametrize('mismatched_input', [False, True])
def test_benchmark_orchestrates_matching_inputs_and_preserves_weight_mode(tmp_path, monkeypatch,
                                                                       random_weights,
                                                                       mismatched_input):
    import sys
    import cross_arch_benchmark as benchmark

    variants = ['base', 'eta0.4', 'eta0.6'] if random_weights else ['base', 'tlr']
    output = tmp_path / 'benchmark'
    argv = ['cross_arch_benchmark.py', '--model', 'ltx25', '--output', str(output),
            '--frames', '17', '--height', '96', '--width', '128', '--warmup', '1', '--repeats', '1']
    argv += ['--random-weights'] if random_weights else ['--tlr-checkpoint', 'unused.pt']
    monkeypatch.setattr(sys, 'argv', argv)
    calls = []

    def subprocess_run(command, **kwargs):
        variant = command[-1]
        calls.append(variant)
        row = dict(gpu='test', torch='test', cuda='test', cudnn='test', dtype='bf16',
                   shape=[1, 3, 17, 96, 128], latent_shape=[1, 128, 3, 3, 4],
                   latent_sha256='same-input', warmup=1, repeats=1, compile_mode='none',
                   ms=10 if variant == 'base' else 5, fps=170 if variant == 'base' else 340,
                   parameters=1, peak_allocated_gib=.1)
        if mismatched_input and variant == variants[-1]:
            row['latent_sha256'] = 'different-input'
        (output / f'{variant}.json').write_text(json.dumps(row))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, 'run', subprocess_run)
    if mismatched_input:
        with pytest.raises(ValueError, match='Unmatched benchmark setting: latent_sha256'):
            benchmark.main()
        return
    benchmark.main()
    result = json.loads((output / 'summary.json').read_text())
    assert calls == variants
    assert list(result['reports']) == variants
    assert len((output / 'summary.csv').read_text().splitlines()) == len(variants) + 1
    if random_weights:
        assert result['random_weights']
        assert result['tlr_speedups'] == {'eta0.4': 2, 'eta0.6': 2}
    else:
        assert result['tlr_speedup'] == 2
        assert 'random_weights' not in result
