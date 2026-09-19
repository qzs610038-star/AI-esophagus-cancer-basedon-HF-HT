from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from errors import ConfigError, NonFiniteDataError, OfflineModelError
from model_adapters import (
    EncoderAdapter,
    adapt_encoder_output,
    fake_spec,
    last_hidden_state_output,
    load_model_specs,
)


def test_timm_final_embedding_keeps_2d_output():
    spec = fake_spec(output_mode="timm_final_embedding", feature_dim=4)
    features = adapt_encoder_output(torch.arange(8, dtype=torch.float32).reshape(2, 4), spec, batch_size=2)
    assert features.shape == (2, 4)
    assert features.dtype == torch.float32
    assert features.device.type == "cpu"


def test_hoptimus_rejects_token_rank():
    spec = fake_spec(output_mode="timm_final_embedding", feature_dim=4)
    with pytest.raises(ConfigError, match="二维"):
        adapt_encoder_output(torch.ones(2, 3, 4), spec, batch_size=2)


def test_phikon_uses_cls_and_rejects_missing_hidden_state():
    spec = fake_spec(name="phikonv2", output_mode="transformers_last_hidden_cls", feature_dim=4, cls_index=0)
    hidden = torch.zeros(2, 5, 4)
    hidden[:, 0, :] = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    hidden[:, 1:, :] = 99
    features = adapt_encoder_output(last_hidden_state_output(hidden), spec, batch_size=2)
    torch.testing.assert_close(features, hidden[:, 0, :])
    with pytest.raises(ConfigError, match="last_hidden_state"):
        adapt_encoder_output(torch.ones(2, 5, 4), spec, batch_size=2)


def test_timm_tokens_cls_mode():
    spec = fake_spec(output_mode="timm_tokens_cls", feature_dim=3, cls_index=0)
    tokens = torch.zeros(1, 4, 3)
    tokens[0, 0] = torch.tensor([9.0, 8.0, 7.0])
    features = adapt_encoder_output(tokens, spec, batch_size=1)
    torch.testing.assert_close(features, tokens[:, 0, :])


def test_nonfinite_and_dim_mismatch_fail():
    spec = fake_spec(output_mode="timm_final_embedding", feature_dim=2)
    bad = torch.tensor([[1.0, float("nan")]])
    with pytest.raises(NonFiniteDataError):
        adapt_encoder_output(bad, spec, batch_size=1)
    with pytest.raises(ConfigError, match="维数"):
        adapt_encoder_output(torch.ones(1, 3), spec, batch_size=1)


class _Fake2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(2, 3)) + self.bias


def test_adapter_freezes_encoder_and_returns_cpu_float32():
    spec = fake_spec(output_mode="timm_final_embedding", feature_dim=3)
    adapter = EncoderAdapter(spec, _Fake2D(), torch.device("cpu"))
    out = adapter.encode(torch.ones(2, 3, 224, 224))
    assert out.shape == (2, 3)
    assert all(not p.requires_grad for p in adapter.model.parameters())
    assert adapter.model.training is False


def test_manifest_rejects_wrong_models(tmp_path: Path):
    path = tmp_path / "model_manifest.json"
    path.write_text(json.dumps({"schema_version": "1.0", "models": {"uni": {}}}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Exception, match="hoptimus0"):
        load_model_specs(path, require_local_files=False)


def test_packaged_manifest_pins_three_new_encoders():
    specs = load_model_specs(
        Path(__file__).resolve().parents[1] / "inputs" / "model_manifest.json",
        require_local_files=False,
    )
    assert specs["hoptimus0"].output_mode == "timm_final_embedding"
    assert specs["hoptimus1"].feature_dim == 1536
    assert specs["phikonv2"].output_mode == "transformers_last_hidden_cls"
    assert specs["phikonv2"].feature_dim == 1024
    assert specs["phikonv2"].normalization_profile == "imagenet_rgb_v1"
    assert specs["hoptimus0"].normalization_profile == "hoptimus_rgb_v1"
    assert specs["hoptimus0"].checkpoint_filename == "pytorch_model.bin"
    assert "pytorch_model.bin" in specs["hoptimus0"].allow_patterns
    assert "model.safetensors" not in specs["hoptimus0"].allow_patterns
