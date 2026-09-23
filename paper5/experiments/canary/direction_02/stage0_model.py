"""Physical decoder-block removal and minimal, explicit q/v LoRA recovery."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen original linear map plus zero-initialized low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(
            torch.empty(
                rank, base.in_features, device=base.weight.device, dtype=torch.float32
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features, rank, device=base.weight.device, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Retain the original map and add the scaled low-rank update."""
        output = self.base(inputs)
        update = F.linear(
            F.linear(inputs.to(self.lora_a.dtype), self.lora_a), self.lora_b
        )
        return output + (self.scale * update).to(output.dtype)


def prune(model: nn.Module, removed: list[int]) -> dict[str, Any]:
    """Physically remove whole blocks and expose the exact parameter mapping."""
    layers = list(model.model.layers)
    if len(set(removed)) != 2 or any(i < 0 or i >= len(layers) for i in removed):
        raise ValueError("Exactly two distinct valid block indices are required")
    retained = [i for i in range(len(layers)) if i not in removed]
    before = sum(p.numel() for p in model.parameters())
    deleted = sum(p.numel() for i in removed for p in layers[i].parameters())
    model.model.layers = nn.ModuleList([layers[i] for i in retained])
    model.config.num_hidden_layers = len(retained)
    model.model.config.num_hidden_layers = len(retained)
    model.config.use_cache = False
    for new, original in enumerate(retained):
        block = model.model.layers[new]
        block.self_attn.layer_idx = new
        if block is not layers[original]:
            raise RuntimeError("Inherited block mapping differs")
    after = sum(p.numel() for p in model.parameters())
    if before - after != deleted:
        raise RuntimeError("Physical parameter removal count differs")
    return {
        "original_layers": len(layers),
        "retained_original_indices": retained,
        "removed_original_indices": removed,
        "new_to_original": dict(enumerate(retained)),
        "base_parameters_before": before,
        "base_parameters_after": after,
        "removed_parameters": deleted,
        "use_cache": False,
    }


def attach_lora(model: nn.Module, cfg: dict[str, Any], seed: int) -> int:
    """Freeze all base weights and attach fresh adapters only to retained q/v."""
    torch.manual_seed(seed)
    model.requires_grad_(False)
    for layer in model.model.layers:
        for name in cfg["lora_targets"]:
            base = getattr(layer.self_attn, name)
            if not isinstance(base, nn.Linear):
                raise TypeError("Expected a pristine q_proj/v_proj Linear")
            setattr(
                layer.self_attn,
                name,
                LoRALinear(base, cfg["lora_rank"], cfg["lora_alpha"]),
            )
    for name, value in model.named_parameters():
        if value.requires_grad != (name.endswith("lora_a") or name.endswith("lora_b")):
            raise RuntimeError("Base freezing or adapter training assignment failed")
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def base_hash(model: nn.Module) -> str:
    """Hash every frozen base byte to detect unintended optimizer mutation."""
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if name.endswith(("lora_a", "lora_b")):
            continue
        if value.requires_grad:
            raise RuntimeError("Trainable base parameter")
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy adapter weights only; no parent weights in checkpoints."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }


def load_adapter(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load an adapter into the same verified structure with strict key matching."""
    params = dict(model.named_parameters())
    expected = {name for name in params if name.endswith(("lora_a", "lora_b"))}
    if set(state) != expected:
        raise ValueError("Adapter/architecture parameter mapping mismatch")
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != params[name].shape:
                raise ValueError("Adapter shape mismatch")
            params[name].copy_(value)


def nll(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Mean next-token CE for all 511 labels of a full 512-token window."""
    logits = model(input_ids=tokens, use_cache=False).logits[:, :-1, :].float()
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
    )


def verify_forward_order(model: nn.Module, tokens: torch.Tensor) -> list[int]:
    """Count execution of real retained blocks; observation hooks do not edit output."""
    observed = []
    handles = [
        layer.register_forward_pre_hook(lambda _m, _x, i=i: observed.append(i))
        for i, layer in enumerate(model.model.layers)
    ]
    try:
        model.eval()
        with torch.no_grad():
            nll(model, tokens)
    finally:
        for handle in handles:
            handle.remove()
    if observed != list(range(model.config.num_hidden_layers)):
        raise RuntimeError("Unexpected retained-block execution")
    return observed
