"""Search configuration, LoRA merge equivalence, and export provenance."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vae_prune import export, search
from vae_prune.models import temporal, wan22
from vae_prune.models.wan21 import WAN_VAE_CONFIG, SearchLoRAConv3d, WanPrunedDecoder
from vae_prune.models.wan21_native import CausalConv3d


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    yield
    torch.set_num_threads(previous)


def search_options(**kwargs):
    return {"grouping": "stage", "keep_per_group": [1] * 5, "lora_rank": 2, "lora_alpha": 3, "gate_scale": 20, **kwargs}


def tiny_wan21(**kwargs):
    config = {**WAN_VAE_CONFIG, "dim": 4}
    return WanPrunedDecoder(config, **kwargs)


def tiny_wan22_teacher():
    teacher = nn.Module()
    teacher.conv2 = wan22.native.CausalConv3d(2, 2, 1)
    teacher.decoder = wan22.native.Decoder3d(dim=4, z_dim=2, temperal_upsample=[True, True, False])
    return teacher


def randomize_lora(model):
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, SearchLoRAConv3d):
                layer.lora_B.normal_(0, 0.03)


def test_lora_merge_preserves_causal_convolution_and_input_gradient():
    original = CausalConv3d(3, 4, 3, padding=1)
    adapter = SearchLoRAConv3d(original, 2, 3)
    randomize_lora(adapter)
    merged = CausalConv3d(3, 4, 3, padding=1)
    with torch.no_grad():
        delta = torch.einsum("or,rithw->oithw", adapter.lora_B[:, :, 0, 0, 0], adapter.lora_A)
        merged.weight.copy_(adapter.weight + adapter.scale * delta)
        merged.bias.copy_(adapter.bias)
    pixels = torch.randn(1, 3, 3, 5, 5, requires_grad=True)
    history = torch.randn(1, 3, 2, 5, 5)
    actual, expected = merged(pixels, history), adapter(pixels, history)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), pixels)[0], torch.autograd.grad(expected.sum(), pixels)[0]
    )


def test_wan21_selected_decoder_matches_merged_decoder():
    searched = tiny_wan21(search=search_options()).eval()
    randomize_lora(searched)
    gate_ema = torch.arange(14).float()
    saved = {"model": searched.state_dict(), "gate_ema": gate_ema, "kept": searched.selected_layers(gate_ema)}
    with patch.object(temporal, "WanPrunedDecoder", tiny_wan21):
        source, merged, records = temporal.merge_search(saved, {"search": search_options()})
    assert len(records) == 10
    assert not any("lora_" in name for name in merged.state_dict())
    z = torch.randn(1, 16, 2, 2, 2)
    expected = torch.cat(temporal.decode(source, z)[0], dim=2)
    actual = torch.cat(temporal.decode(merged, z)[0], dim=2)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)


def test_wan22_search_config_and_merge_are_consistent():
    teacher = tiny_wan22_teacher()
    options = search_options(keep_per_group=[2, 1, 1, 1, 1])
    searched = wan22.Component(teacher, search=options).eval()
    assert searched.candidate_masks.shape == (81, 14)
    assert searched.gate_scale == 20
    assert all(
        layer.lora_A.shape[0] == 2 and layer.scale == 1.5
        for layer in searched.modules()
        if isinstance(layer, SearchLoRAConv3d)
    )
    assert all(
        "lora_" in name or name == "gate_logits"
        for name, parameter in searched.named_parameters()
        if parameter.requires_grad
    )
    randomize_lora(searched)
    gate_ema = torch.arange(14).float()
    merged = wan22.merge(searched, teacher, gate_ema).eval()
    assert len(merged.kept) == 6
    z = torch.randn(1, 2, 2, 2, 2)
    expected = torch.cat(wan22.decode(searched, z)[0], dim=2)
    actual = torch.cat(wan22.decode(merged, z)[0], dim=2)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)


def test_wan22_export_rejects_inconsistent_selection():
    teacher = tiny_wan22_teacher()
    options = search_options()
    searched = wan22.Component(teacher, search=options)
    wrapper = nn.Module()
    wrapper.component = teacher
    saved = {
        "family": "wan22",
        "config": {"search": options},
        "model": searched.state_dict(),
        "gate_ema": torch.arange(14).float(),
        "kept": [0],
    }
    with pytest.raises(ValueError, match="Saved EMA selection"):
        export.merge_searched(wrapper, saved, "wan22")


def test_export_retains_search_step_and_merged_weights(tmp_path):
    teacher_path = tmp_path / "teacher.pth"
    teacher_path.write_bytes(b"teacher identity")
    checkpoint = tmp_path / "search.pt"
    torch.save({"step": 37, "kept": [1, 4, 7, 10, 13]}, checkpoint)
    component = nn.Linear(2, 3)
    config = {"family": "wan21", "variant": "lite", "fraction": 0.6, "tail_2d": True}
    output = tmp_path / "lite.pt"
    with (
        patch.object(export, "Teacher"),
        patch.object(export, "merge_searched", return_value=component),
        patch.object(export, "compress", return_value=[]),
    ):
        export.run(config, teacher_path, checkpoint, output)
    saved = torch.load(output, weights_only=True)
    assert saved["provenance"]["source_step"] == 37
    assert saved["provenance"]["source_sha256"] == export.sha256(checkpoint)
    assert saved["kept"] == [1, 4, 7, 10, 13]
    torch.testing.assert_close(saved["model"]["weight"], component.weight)


def test_example_configs_use_shared_search_contract():
    root = Path(__file__).resolve().parents[1] / "configs"
    for family in ("wan21", "wan22"):
        config = json.loads((root / "search" / f"{family}.json").read_text())
        assert config["family"] == family
        assert search.temperature_at(config["steps"], config["steps"], **config["temperature"]) == pytest.approx(0.4)
        for variant in ("pro", "lite"):
            exported = json.loads((root / "export" / f"{family}_{variant}.json").read_text())
            assert exported == {"family": family, "variant": variant, "fraction": 0.6, "tail_2d": True}
    a = json.loads((root / "search/wan21.json").read_text())
    b = json.loads((root / "search/wan22.json").read_text())
    a["family"] = b["family"]
    assert a == b
