"""Load granite safetensors directly (bf16 on disk -> requested dtype).

Supports loading only a subset of experts per layer, so an EP rank materializes
the expert weights it owns (plus any replicas), never the full expert set.
"""
from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from . import MODEL_ID
from .modeling import ModelConfig

_EXPERT_SUFFIXES = (
    "block_sparse_moe.input_linear.weight",
    "block_sparse_moe.output_linear.weight",
)


def expert_weight_keys(layer: int) -> tuple[str, str]:
    """(input_linear, output_linear) fused-expert weight keys for one layer."""
    in_suffix, out_suffix = _EXPERT_SUFFIXES
    return f"model.layers.{layer}.{in_suffix}", f"model.layers.{layer}.{out_suffix}"


def model_path() -> Path:
    return Path(snapshot_download(MODEL_ID))


def load_config() -> ModelConfig:
    return ModelConfig.from_json(model_path() / "config.json")


def load_weights(experts_per_layer: dict[int, list[int]] | None = None,
                 dtype=torch.float32, only_experts: bool = False) -> dict[str, torch.Tensor]:
    """experts_per_layer: layer -> sorted expert ids to materialize (None = all).

    Expert tensors come back with shape (len(ids), ...) in the given id order;
    the caller owns the global-id -> local-row mapping.
    only_experts: skip all non-expert tensors (live re-placement hot-swap, where
    the non-expert weights are already resident).
    """
    path = model_path() / "model.safetensors"
    weights = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith(_EXPERT_SUFFIXES):
                if experts_per_layer is None:
                    weights[key] = f.get_tensor(key).to(dtype)
                    continue
                layer = int(key.split(".layers.")[1].split(".")[0])
                ids = experts_per_layer[layer]
                sl = f.get_slice(key)
                weights[key] = torch.cat(
                    [sl[e:e + 1] for e in ids], dim=0).to(dtype)
            elif not only_experts:
                weights[key] = f.get_tensor(key).to(dtype)
    return weights


def build_local_model(dtype=torch.float32):
    """Single-process model with every expert resident (reference path)."""
    from .modeling import GraniteMoeModel, LocalMoEBackend

    cfg = load_config()
    w = load_weights(dtype=dtype)
    backends = [
        LocalMoEBackend(w[in_key], w[out_key])
        for in_key, out_key in map(expert_weight_keys, range(cfg.num_hidden_layers))
    ]
    return GraniteMoeModel(cfg, w, backends), cfg
