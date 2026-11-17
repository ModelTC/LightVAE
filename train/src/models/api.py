"""Shared decoder API; Wan2.1 normalized latents, Wan2.2 raw means."""

import torch
from torch import nn
from . import temporal, wan22
from .wan21 import WanPrunedEncoder, WanPrunedDecoder, normalized_posterior_stats


def sample_posterior(mean, std, mode="mean"):
    if mode not in ("mean", "sample"):
        raise ValueError("posterior mode must be mean or sample")
    return mean if mode == "mean" else mean + std * torch.randn_like(mean)


def collect_features(features, anchor):
    # Merge streaming chunks separately for each stage, never average stages together.
    if isinstance(anchor, tuple):
        if any(len(chunk) != len(anchor) for chunk in features):
            raise ValueError("Requested feature stages were not captured")
        return {index: torch.cat([chunk[i] for chunk in features], dim=2) for i, index in enumerate(anchor)}
    return features


class Decoder(nn.Module):
    def __init__(self, component, family):
        super().__init__()
        self.component, self.family = component, family

    def forward(self, z, anchor=None):
        ops = temporal if self.family == "wan21" else wan22
        chunks, features = ops.decode(self.component, z, anchor, checkpoint_chunks=self.training)
        return torch.cat(chunks, dim=2), collect_features(features, anchor)


class Teacher(nn.Module):
    def __init__(self, family, path):
        super().__init__()
        self.family = family
        if family == "wan21":
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            self.encoder = WanPrunedEncoder()
            self.component = WanPrunedDecoder()
            self.encoder.initialize_from_teacher_state(state)
            self.component.initialize_from_teacher_state(state)
        elif family == "wan22":
            self.component = wan22.load_teacher(path)
        else:
            raise ValueError(family)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, pixels, mode="mean"):
        """pixels: B,C,T,H,W in [-1,1], T=4n+1, spatial aligned."""
        if self.family == "wan22":
            model = self.component
            model.clear_cache()
            try:
                x = wan22.native.patchify(pixels, patch_size=2)
                outputs = []
                for end in range(0, x.shape[2], 4):
                    start, stop = (0, 1) if end == 0 else (end - 3, end + 1)
                    outputs.append(model.encoder(x[:, :, start:stop], feat_cache=model._enc_feat_map, feat_idx=[0]))
                mean, logvar = model.conv1(torch.cat(outputs, 2)).float().chunk(2, dim=1)
                return sample_posterior(mean, (0.5 * logvar.clamp(-30, 20)).exp(), mode)
            finally:
                model.clear_cache()
        cache, outputs = [None] * 256, []
        for end in range(0, pixels.shape[2], 4):
            start, stop = (0, 1) if end == 0 else (end - 3, end + 1)
            outputs.append(self.encoder.encoder(pixels[:, :, start:stop], feat_cache=cache, feat_idx=[0]))
        mean, std = normalized_posterior_stats(self.encoder.conv1(torch.cat(outputs, 2)))
        return sample_posterior(mean, std, mode)

    @torch.no_grad()
    def forward(self, z, anchor=None):
        ops = temporal if self.family == "wan21" else wan22
        chunks, features = ops.decode(self.component, z, anchor)
        return torch.cat(chunks, 2), collect_features(features, anchor)
