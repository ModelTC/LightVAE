"""Load original and recovered TLR video VAEs for inference."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .cross_arch import load_vae
from .cross_arch_attention import HunyuanFrameCausalAttention
from .cross_arch_tlr import convert_decoder, fold_decoder


def optimize_tlr(vae, *, attention=True, padding=True, norms=True):
    """Install inference-only kernels after converting or loading a TLR decoder."""
    if attention and vae.family == 'hunyuanvideo15':
        vae.decoder.mid.attn_1 = HunyuanFrameCausalAttention(
            vae.decoder.mid.attn_1)
    if vae.family in ('hunyuanvideo15', 'ltx25'):
        from .cross_arch_ops import optimize_norms, optimize_padding
        if padding:
            optimize_padding(vae)
        if norms:
            optimize_norms(vae)
    elif vae.family == 'cogvideox15':
        from .cross_arch_cog import optimize_cog
        optimize_cog(vae, padding=padding, norms=norms)


def load_variant(family: str, model_root: Path, checkpoint: Path | None = None,
                 dtype: torch.dtype = torch.float32):
    """Load the base in its inference dtype; keep recovery folding in FP32."""
    load_dtype = torch.float32 if checkpoint is not None else dtype
    vae = load_vae(family, root=model_root, device='cpu', dtype=load_dtype)
    if checkpoint is None:
        return vae, {'variant': 'base', 'weights': str(vae.weights),
                     'implementation': 'original source; no LightVAE kernels'}

    if Path(checkpoint).suffix == '.safetensors':
        with safe_open(checkpoint, framework='pt', device='cpu') as handle:
            metadata = handle.metadata() or {}
            if (metadata.get('model_type') != 'lightvae_decoder'
                    or 'checkpoint_metadata' not in metadata):
                raise ValueError('Recovered decoder safetensors metadata is missing or invalid')
            saved = json.loads(metadata['checkpoint_metadata'])
            saved['decoder_state'] = {key: handle.get_tensor(key) for key in handle.keys()}
    else:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if saved['schema_version'] != 1 or saved['family'] != family:
        raise ValueError('Recovered decoder checkpoint schema or family mismatch')
    records = saved['tlr_records']
    expected = {path.removeprefix('decoder.') for path, _, _ in vae.eligible_decoder_convs}
    if {record['path'] for record in records} != expected:
        raise ValueError('Recovered decoder does not cover the original residual Conv3ds')
    convert_decoder(vae.decoder, None, saved['eta'], records=records)
    vae.decoder.load_state_dict(saved['decoder_state'], strict=True, assign=True)
    if not all(record.get('folded', False) for record in records):
        fold_decoder(vae.decoder, records)
    optimize_tlr(vae)
    provenance = {
        'variant': 'tlr', 'checkpoint': str(Path(checkpoint).resolve()),
        'step': saved['step'], 'eta': saved['eta'],
        'converted': len(records), 'spatial_only': sum(r['rank'] == 0 for r in records),
        'folded': True, 'source_weights': str(vae.weights),
    }
    del saved
    return vae, provenance
