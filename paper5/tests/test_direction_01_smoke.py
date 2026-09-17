"""Tests for the Direction 1 RTX 3050 smoke-test pipeline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from paper5.experiments.canary.direction_01.smoke.analysis import (
    summarize_pairs,
    write_csv,
    write_json,
)
from paper5.experiments.canary.direction_01.smoke.core import (
    evaluate_next_token_metrics,
    sample_layer_pairs,
    select_evenly_spaced_layers,
    whole_block_skip,
)
from paper5.experiments.canary.direction_01.smoke.run_smoke import load_config
from paper5.experiments.canary.direction_01.smoke.transport import toy_jvp_transport


class ToyBlock(nn.Module):
    """Small residual block used to validate static edit semantics."""

    def __init__(self, width: int, scale: float) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.linear.weight)
        self.scale = scale

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply a deterministic residual transform."""
        return hidden_states + self.scale * self.linear(hidden_states)


class ToyBackbone(nn.Module):
    """Backbone exposing the Hugging Face `model.layers` contract."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ToyBlock(width, 0.05), ToyBlock(width, 0.10), ToyBlock(width, 0.15)]
        )


class ToyCausalLM(nn.Module):
    """Tiny causal LM for metric and edit restoration tests."""

    def __init__(self, vocabulary: int = 11, width: int = 8) -> None:
        super().__init__()
        self.model = ToyBackbone(width)
        self.embedding = nn.Embedding(vocabulary, width)
        self.head = nn.Linear(width, vocabulary, bias=False)

    def forward(self, input_ids: Tensor, use_cache: bool = False) -> object:
        """Return an object with causal-LM logits."""
        del use_cache
        hidden_states = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        logits = self.head(hidden_states)
        return type("ToyOutput", (), {"logits": logits})()


def test_block_skip_restores_model_and_pair_order_is_irrelevant() -> None:
    """Static skip edits are scoped, reversible, and order independent."""
    torch.manual_seed(7)
    model = ToyCausalLM().eval()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    parent = model(tokens).logits.detach().clone()
    with whole_block_skip(model, [0]):
        single = model(tokens).logits.detach().clone()
    assert not torch.equal(parent, single)
    assert torch.equal(model(tokens).logits, parent)
    with whole_block_skip(model, [0, 2]):
        forward_order = model(tokens).logits.detach().clone()
    with whole_block_skip(model, [2, 0]):
        reverse_order = model(tokens).logits.detach().clone()
    assert torch.equal(forward_order, reverse_order)
    assert torch.equal(model(tokens).logits, parent)


def test_no_edit_metrics_repeat_and_all_deltas_are_finite() -> None:
    """No-edit metrics repeat exactly and edited metrics remain finite."""
    torch.manual_seed(11)
    model = ToyCausalLM().eval()
    sequences = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.long)
    first = evaluate_next_token_metrics(model, sequences, 1, torch.device("cpu"))
    with whole_block_skip(model, []):
        second = evaluate_next_token_metrics(model, sequences, 1, torch.device("cpu"))
    with whole_block_skip(model, [1]):
        edited = evaluate_next_token_metrics(model, sequences, 1, torch.device("cpu"))
    assert first == second
    assert math.isfinite(edited.nll)
    assert math.isfinite(edited.perplexity)
    assert math.isfinite(edited.nll - first.nll)


def test_layer_and_pair_selection_are_deterministic() -> None:
    """Layer and pair selection reproduce under a fixed seed."""
    layers = select_evenly_spaced_layers(24, 6, 1)
    assert len(layers) == 6
    assert len(set(layers)) == 6
    first = sample_layer_pairs(layers, 10, 123)
    second = sample_layer_pairs(layers, 10, 123)
    assert first == second
    assert len(first) == 10
    assert all(left < right for left, right in first)


def test_analysis_and_artifact_schema(tmp_path: Path) -> None:
    """CSV/JSON writers preserve the required additive-analysis fields."""
    rows = [
        {
            "run_label": "SMOKE TEST / PRELIMINARY",
            "split": "validation",
            "edit_family": "whole_transformer_block_skip",
            "first_layer": index,
            "second_layer": index + 1,
            "layer_distance": 1,
            "first_delta_nll": 0.1,
            "second_delta_nll": 0.1,
            "additive_delta_nll": 0.2 + index * 0.01,
            "actual_delta_nll": 0.21 + index * 0.02,
            "interaction_nll": 0.01 + index * 0.01,
            "relative_abs_interaction": 0.05 + index * 0.05,
            "parent_perplexity": 10.0,
            "edited_perplexity": 10.5,
            "delta_perplexity": 0.5,
            "token_count": 32,
        }
        for index in range(3)
    ]
    analysis = summarize_pairs(rows)
    assert analysis["pair_count"] == 3
    assert "kendall_tau_b_additive_vs_actual" in analysis
    csv_path = tmp_path / "pairs.csv"
    json_path = tmp_path / "metrics.json"
    write_csv(csv_path, rows)
    write_json(json_path, {"rows": rows, "analysis": analysis})
    with csv_path.open(encoding="utf-8", newline="") as handle:
        loaded_rows = list(csv.DictReader(handle))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(loaded_rows) == 3
    assert payload["analysis"]["pair_count"] == 3


def test_toy_jvp_transport_shape_and_value() -> None:
    """The deferred H2 helper has the intended projected-JVP contract."""
    matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    state = torch.tensor([1.0, -1.0])
    residual = torch.tensor([0.5, 2.0])
    projection = torch.tensor([[1.0, 1.0], [1.0, -1.0], [0.5, 0.0]])
    transported = toy_jvp_transport(
        lambda value: matrix @ value,
        state,
        residual,
        projection,
    )
    expected = projection @ (matrix @ residual)
    assert transported.shape == (3,)
    assert torch.allclose(transported, expected)


def test_smoke_config_is_explicitly_non_scientific() -> None:
    """The committed run config cannot be mistaken for a formal canary."""
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "canary"
        / "direction_01"
        / "smoke"
        / "rtx3050_qwen_0_5b.json"
    )
    config = load_config(config_path)
    assert config["run_label"] == "SMOKE TEST / PRELIMINARY"
    assert config["formal_a800_h1_h2_deferred"] is True
    assert config["memory_release_tolerance_mib"] == 16.0
    assert config["edit_count"] == 6
    assert 10 <= config["pair_count"] <= 15
