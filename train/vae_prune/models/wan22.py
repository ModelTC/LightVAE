"""Wan2.2 TI2V-5B: 5-group residual search and streaming temporal factors.

Keep Up_ResidualBlock's DupUp3D path and channel-changing shortcuts, even
when its residual branch is removed. Latents here are RAW posterior means;
both decoders receive the identical latent, so no Wan2.1 normalization is used.
"""

import copy

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .wan21 import SearchLoRAConv3d
from .search_masks import build_residual_search_masks
from .temporal import PackedTemporalConv, frames, unframes, mass_rank
from . import wan22_native as native

GROUPS = ((0, 1), (2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13))


def load_teacher(path):
    with torch.device("meta"):
        model = native.WanVAE_(dim=160, dec_dim=256, z_dim=48, temperal_downsample=[False, True, True])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), assign=True)
    return model.eval().requires_grad_(False)


def block_paths():
    return ["middle.0", "middle.2"] + [f"upsamples.{s}.upsamples.{b}" for s in range(4) for b in range(3)]


class ShortcutOnly(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.shortcut = block.shortcut


class Component(nn.Module):
    def __init__(self, teacher=None, search=None, kept=None):
        super().__init__()
        self.conv2 = copy.deepcopy(teacher.conv2) if teacher is not None else native.CausalConv3d(48, 48, 1)
        self.decoder = (
            copy.deepcopy(teacher.decoder)
            if teacher is not None
            else native.Decoder3d(dim=256, z_dim=48, temperal_upsample=[True, True, False])
        )
        self.search_config = dict(search) if search is not None else None
        self.search = search is not None
        self.kept = list(range(14)) if kept is None else list(kept)
        self.temperature = 4.0
        self.noise = None
        if self.search:
            if kept is not None:
                raise ValueError("Search starts from the full teacher decoder.")
            masks = build_residual_search_masks(14, search, GROUPS, default_keep=5)
            self.register_buffer("candidate_masks", masks, persistent=False)
            self.gate_logits = nn.Parameter(torch.zeros(14))
            self.gate_scale = float(search.get("gate_scale", 100))
            self.temperature = float(search.get("temperature", 4))
            rank = int(search["lora_rank"])
            alpha = float(search["lora_alpha"])
            if rank <= 0 or alpha <= 0 or self.gate_scale <= 0 or self.temperature <= 0:
                raise ValueError("LoRA rank/alpha, gate scale and temperature must be positive.")
            for path in block_paths():
                block = self.decoder.get_submodule(path)
                for index in (2, 6):
                    block.residual[index] = SearchLoRAConv3d(block.residual[index], rank, alpha)
            for name, parameter in self.named_parameters():
                parameter.requires_grad_("lora_" in name or name == "gate_logits")
        else:
            for index, path in enumerate(block_paths()):
                if index not in self.kept:
                    self.decoder.set_submodule(path, ShortcutOnly(self.decoder.get_submodule(path)))

    def selected_layers(self, logits=None):
        if not self.search:
            return self.kept
        scores = self.candidate_masks @ (self.gate_logits if logits is None else logits).detach()
        return self.candidate_masks[scores.argmax()].nonzero().flatten().tolist()

    def masks(self):
        if not self.search:
            return None
        score = (self.candidate_masks @ self.gate_logits) * self.gate_scale
        if not self.training:
            return self.candidate_masks[score.argmax()]
        if self.noise is None:
            raise RuntimeError("Prepare/broadcast search noise once per video, before checkpointed forward")
        soft = ((score + self.noise) / self.temperature).softmax(0)
        hard = torch.nn.functional.one_hot(soft.argmax(), len(soft)).to(soft.dtype)
        return (hard - soft.detach() + soft) @ self.candidate_masks

    def prepare_search_step(self, batch_size):
        if self.search:
            self.noise = -torch.empty(len(self.candidate_masks), device=self.gate_logits.device).exponential_().log()

    def clear_search_step(self):
        self.noise = None

    def forward(self, z, anchor=None):
        return decode(self, z, anchor, checkpoint_chunks=True)


def cached(layer, x, history, slot):
    if isinstance(layer, PackedTemporalConv):
        index = slot[0]
        slot[0] += 1
        x, history[index] = layer(x, history[index])
        return x
    if isinstance(layer, (native.CausalConv3d, SearchLoRAConv3d)) and layer.kernel_size[0] > 1:
        index = slot[0]
        slot[0] += 1
        old = history[index]
        keep = x[:, :, -2:]
        if keep.shape[2] < 2 and old is not None:
            keep = torch.cat((old[:, :, -1:], keep), 2)
        out = layer(x, old)
        history[index] = keep.clone()
        return out
    return layer(x)


def residual(block, x, history, slot, gate=None):
    skip = block.shortcut(x)
    if isinstance(block, ShortcutOnly):
        return skip
    for layer in block.residual:
        x = cached(layer, x, history, slot)
    return skip + x * (1 if gate is None else gate.to(x.dtype))


def resample(layer, x, history, slot):
    b, c, t, _, _ = x.shape
    if layer.mode == "upsample3d":
        i = slot[0]
        old = history[i]
        if old is None:
            history[i] = "startup"
            slot[0] += 1
        else:
            if isinstance(old, str):
                history[i] = None
            x = cached(layer.time_conv, x, history, slot)
            x = x.reshape(b, 2, c, t, *x.shape[-2:]).permute(0, 2, 3, 1, 4, 5).flatten(2, 3)
    return unframes(layer.resample(frames(x)), b, x.shape[2])


def group(component, index, x, history, slot, gates, first):
    decoder = component.decoder
    if index == 0:
        x = residual(decoder.middle[0], x, history, slot, None if gates is None else gates[0])
        x = decoder.middle[1](x)
        return residual(decoder.middle[2], x, history, slot, None if gates is None else gates[1])
    stage = decoder.upsamples[index - 1]
    skip = x
    for b, layer in enumerate(stage.upsamples):
        if isinstance(layer, native.Resample):
            x = resample(layer, x, history, slot)
        else:
            i = 2 + (index - 1) * 3 + b
            x = residual(layer, x, history, slot, None if gates is None else gates[i])
    if stage.avg_shortcut is not None:
        x = x + stage.avg_shortcut(skip, first_chunk=first)
    return x


def finish(component, x, history, slot):
    for layer in component.decoder.head:
        x = cached(layer, x, history, slot)
    return native.unpatchify(x, 2)


def decode(component, z, anchor=None, checkpoint_chunks=False):
    projected = component.conv2(z)
    gates = component.masks() if isinstance(component, Component) else None
    history = [None] * 128
    outputs = []
    features = []
    for t in range(projected.shape[2]):
        # Bind first_chunk in the closure: backward recomputation must not see
        # the loop's final t value.
        def chunk(x, mask, *old, first=(t == 0)):
            old = list(old)
            slot = [0]
            feature = None
            selected = []
            x = cached(component.decoder.conv1, x, old, slot)
            for g in range(5):
                x = group(component, g, x, old, slot, mask, first)
                if g == anchor:
                    feature = x
                if isinstance(anchor, tuple) and g in anchor:
                    selected.append(x)
            if isinstance(anchor, tuple):
                feature = tuple(selected)
            return finish(component, x, old, slot), feature, *old

        args = (projected[:, :, t : t + 1], gates, *history)
        if checkpoint_chunks and torch.is_grad_enabled():
            y, h, *history = checkpoint(chunk, *args, use_reentrant=False)
        else:
            y, h, *history = chunk(*args)
        outputs.append(y)
        if h is not None:
            features.append(h)
    return outputs, features


def suffix(component, features, anchor):
    history = [None] * 128
    outputs = []
    for t, feature in enumerate(features):

        def chunk(x, *old, first=(t == 0)):
            old = list(old)
            slot = [0]
            for g in range(anchor + 1, 5):
                x = group(component, g, x, old, slot, None, first)
            return finish(component, x, old, slot), *old

        y, *history = checkpoint(chunk, feature, *history, use_reentrant=False)
        outputs.append(y)
    return outputs


@torch.no_grad()
def merge(search, teacher, ema):
    search.gate_logits.copy_(ema)
    search.eval()
    fixed = Component(teacher, kept=search.selected_layers())
    original = search.state_dict()
    state = {k: original[k].clone() for k in fixed.state_dict()}
    for name, layer in search.named_modules():
        if isinstance(layer, SearchLoRAConv3d) and name + ".weight" in state:
            delta = torch.einsum("or,rithw->oithw", layer.lora_B.float()[:, :, 0, 0, 0], layer.lora_A.float())
            state[name + ".weight"] = layer.weight.float() + layer.scale * delta
    fixed.load_state_dict(state, strict=True)
    return fixed


@torch.no_grad()
def convert(component, fraction=0.6, records=None, tail_2d=True):
    result = []
    candidates = [
        (n, m)
        for n, m in component.decoder.named_modules()
        if isinstance(m, native.CausalConv3d) and m.kernel_size[0] == 3
    ]
    if records is not None and {r["path"] for r in records} != {n for n, _ in candidates}:
        raise ValueError("Checkpoint temporal paths mismatch")
    for name, layer in candidates:
        if layer.stride != (1, 1, 1) or layer.dilation != (1, 1, 1) or layer.groups != 1:
            raise ValueError(f"Unsupported temporal operator {name}")
        co, ci, _, kh, kw = layer.weight.shape
        pure2d = tail_2d and name.startswith(("upsamples.2.", "upsamples.3."))
        if records is None:
            w = layer.weight.float()
            if pure2d:
                rank = 0
            else:
                matrix = torch.cat((w[:, :, 1].flatten(1), w[:, :, 0].flatten(1)), 1)
                u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
                rank = mass_rank(s, fraction)
            rec = dict(
                path=name,
                shape=list(w.shape),
                padding=layer._padding[0],
                rank=rank,
                bias=layer.bias is not None,
                pure2d=pure2d,
                fraction=fraction,
            )
        else:
            rec = next(r for r in records if r["path"] == name)
            rank = rec["rank"]
        op = PackedTemporalConv(rec["shape"], rec["padding"], rank, rec["bias"]).to(layer.weight.device)
        if records is None:
            weights = [w.sum(2)]
            if rank:
                root = s[:rank].sqrt()
                factors = (root[:, None] * vh[:rank]).reshape(rank, 2, ci, kh, kw)
                weights.extend((factors[:, 0], factors[:, 1]))
                op.up.weight.copy_((u[:, :rank] * root)[:, :, None, None])
            op.down.weight.copy_(torch.cat(weights))
            if op.bias is not None:
                op.bias.copy_(layer.bias)
        component.decoder.set_submodule(name, op)
        result.append(rec)
    return result
