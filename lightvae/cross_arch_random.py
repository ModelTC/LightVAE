"""Weight-free decoder construction from published configs and exact TLR records."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .cross_arch import VideoVAE, _model_class
from .cross_arch_ltx import LTXVideoVAE, _official_ltx
from .cross_arch_runtime import optimize_tlr
from .cross_arch_tlr import convert_decoder


STRUCTURE_ROOT = Path(__file__).parent / 'config/cross_arch_benchmark'
RANDOM_VARIANTS = ('0.4', '0.6')


def _native_vae(family, config, model_root, structure_path):
    """Resolve source imports normally, then instantiate architecture on meta."""
    if family == 'hunyuanvideo15':
        model_class = _model_class(family)
        with torch.device('meta'):
            model = model_class.from_config(config)
        model.cpu_offload = False
        return VideoVAE(family, model, structure_path)
    if family == 'ltx25':
        _, causal_conv, _, decoder_configurator, _, _ = _official_ltx(model_root)
        with torch.device('meta'):
            model = nn.Module()
            model.add_module('decoder', decoder_configurator.from_metadata(config))
        return LTXVideoVAE(model, structure_path, causal_conv)
    raise ValueError('Random-weight benchmarks support hunyuanvideo15 and ltx25')


@torch.no_grad()
def initialize_decoder(decoder, seed, dtype):
    """Materialize only the decoder, with deterministic finite synthetic weights."""
    decoder.to_empty(device='cpu')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    for module in decoder.modules():
        norm = 'norm' in type(module).__name__.lower()
        for name, parameter in module.named_parameters(recurse=False):
            if name == 'bias':
                parameter.zero_()
            elif name == 'gamma' or (name == 'weight' and (norm or parameter.ndim == 1)):
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0, std=0.01, generator=generator)
        for name, buffer in module.named_buffers(recurse=False):
            buffer.fill_(1 if name in ('running_var', 'std-of-means') else 0)
    decoder.to(dtype=dtype).eval().requires_grad_(False)


def load_random_variant(family, model_root, structure_dir=STRUCTURE_ROOT,
                        variant='base', *, dtype=torch.bfloat16, seed=2026):
    """Build base or exact recovered TLR geometry without weights, SVD or an encoder load."""
    if family not in ('hunyuanvideo15', 'ltx25'):
        raise ValueError('Random-weight benchmarks support hunyuanvideo15 and ltx25')
    if variant not in ('base', *RANDOM_VARIANTS):
        raise ValueError(f'Unknown random-weight variant: {variant}')
    structure_path = Path(structure_dir) / f'{family}.json'
    raw = structure_path.read_bytes()
    structure = json.loads(raw)
    if structure.get('schema_version') != 1 or structure.get('family') != family:
        raise ValueError('Benchmark structure schema or family mismatch')
    if not isinstance(structure.get('config'), dict) or set(structure.get('variants', {})) != set(RANDOM_VARIANTS):
        raise ValueError('Benchmark structure needs native config and variants 0.4/0.6')
    vae = _native_vae(family, structure['config'], model_root, structure_path)
    expected = {path.removeprefix('decoder.') for path, _, _ in vae.eligible_decoder_convs}
    source = dict(variant='base' if variant == 'base' else f'eta{variant}',
                  random_weights=True, structure=str(structure_path.resolve()),
                  structure_sha256=hashlib.sha256(raw).hexdigest(), weight_seed=seed,
                  initialization='normal std=0.01; normalization scales/std buffers=1; biases/other buffers=0',
                  implementation='original source; no LightVAE kernels')
    if variant != 'base':
        saved = structure['variants'][variant]
        records = deepcopy(saved['tlr_records'])
        paths = [row['path'] for row in records]
        eta = float(variant)
        if (saved.get('eta') != eta or len(paths) != len(set(paths)) or set(paths) != expected
                or any(row.get('eta') != eta or type(row.get('rank')) is not int for row in records)):
            raise ValueError('Benchmark TLR records do not match the family and exact eta architecture')
        # Both trained checkpoint formats are folded for inference. Construct that
        # final shape directly before initializing random values; no SVD/folding is needed.
        for record in records:
            record['folded'] = True
        with torch.device('meta'):
            convert_decoder(vae.decoder, None, eta, records=records)
        source.update(eta=eta, step=saved['step'], converted=len(records),
                      spatial_only=sum(row['rank'] == 0 for row in records), folded=True,
                      tlr_ranks={row['path']: row['rank'] for row in records},
                      implementation='folded TLR; LightVAE inference kernels')
    initialize_decoder(vae.decoder, seed, dtype)
    if variant != 'base':
        optimize_tlr(vae)
    return vae, source
