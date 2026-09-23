"""CPU tests of checkpoint continuation and the exploratory access boundary."""

import copy
from pathlib import Path

import pytest
import torch

from paper5.experiments.canary.direction_02 import run_recovery_capacity as cap
from paper5.experiments.canary.direction_02 import stage0_data as data
from paper5.experiments.canary.direction_02 import stage0_model as impl
from paper5.tests.direction_02.test_stage0 import tiny_model


def test_data_resume_never_repeats_consumed_windows() -> None:
    order = data.training_order(400, 17)
    continued = [
        x for step in range(10, 100) for x in cap.training_indices(order, step)
    ]
    assert continued == order[40:400]
    assert len(set(continued)) == 360
    assert not set(continued) & set(order[:40])
    for step in [0, 9, 100]:
        with pytest.raises(ValueError):
            cap.training_indices(order, step)


def test_no_e_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Underlying loader must not be called")

    monkeypatch.setattr(cap.original, "load_split", forbidden)
    with pytest.raises(ValueError, match="Only T and S"):
        cap.load_ts("E", {})


def test_resume_adamw_matches_continuous_training(tmp_path: Path) -> None:
    def construct() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        model = tiny_model()
        impl.prune(model, [1, 3])
        model.requires_grad_(False)
        impl.attach_lora(model, data.config(), 17)
        params = [p for p in model.parameters() if p.requires_grad]
        return model, torch.optim.AdamW(params, lr=1e-4, weight_decay=0)

    def advance(
        model: torch.nn.Module, opt: torch.optim.Optimizer, start: int, end: int
    ) -> None:
        for step in range(start, end):
            opt.zero_grad(set_to_none=True)
            for micro in range(4):
                tokens = ((torch.arange(8) + step * 4 + micro) % 48).reshape(1, 8)
                (impl.nll(model, tokens) / 4).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()

    continuous, optimizer = construct()
    base = impl.base_hash(continuous)
    advance(continuous, optimizer, 0, 10)
    saved = {
        "adapter": impl.adapter_state(continuous),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": [torch.get_rng_state()],
        "step": 10,
        "seed": 17,
        "candidate": "a00",
        "executed_code_sha": cap.SPEC["source_code_sha"],
        "config_sha256": cap.SPEC["source_config_sha256"],
        "mapping": {"removed_original_indices": [12, 15]},
        "training_order": data.training_order(400, 17),
    }
    torch.save(saved, tmp_path / "step_10.pt")
    loaded = torch.load(tmp_path / "step_10.pt", weights_only=True)
    cap.validate_checkpoint(loaded)
    advance(continuous, optimizer, 10, 20)
    resumed, resumed_opt = construct()
    cap.restore(resumed, resumed_opt, loaded, cuda=False)
    assert all(float(s["step"]) == 10 for s in resumed_opt.state.values())
    advance(resumed, resumed_opt, 10, 20)
    assert impl.base_hash(resumed) == impl.base_hash(continuous) == base
    for name, value in impl.adapter_state(continuous).items():
        torch.testing.assert_close(
            value, impl.adapter_state(resumed)[name], rtol=0, atol=0
        )
    for a, b in zip(optimizer.state.values(), resumed_opt.state.values(), strict=True):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    for key, wrong in [
        ("step", 0),
        ("seed", 29),
        ("candidate", "a01"),
        ("executed_code_sha", "wrong"),
    ]:
        changed = dict(loaded, **{key: wrong})
        with pytest.raises(ValueError):
            cap.validate_checkpoint(changed)
    changed = copy.deepcopy(loaded)
    changed["optimizer"]["state"][0]["step"].zero_()
    with pytest.raises(ValueError, match="step 10"):
        cap.validate_checkpoint(changed)


def test_original_stage0_and_fixed_diagnostic_contract() -> None:
    assert data.file_hash(data.CONFIG) == cap.SPEC["source_config_sha256"]
    assert (
        data.file_hash(cap.original.PROTOCOL)
        == "f1dfa962b1500ad66a9618a59ea2af97f3220316339b755aaeecd81c2a21467c"
    )
    assert cap.SPEC["s_checkpoints"] == [20, 50, 100]
    assert cap.SPEC["quality_max_ppl_ratio"] == 1.15
    assert cap.SPEC["physical_gpu"] == 2
    assert not cap.SPEC["e_access_allowed"] and not cap.SPEC["formal_launch_allowed"]
    assert cap.OUT != cap.SOURCE_OUT and cap.CACHE != cap.SOURCE_CACHE
