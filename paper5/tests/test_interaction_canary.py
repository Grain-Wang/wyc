from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_canary_module() -> ModuleType:
    """Load the canary script as a module from its repository path."""
    path = Path(__file__).parents[1] / "experiments" / "interaction_canary.py"
    spec = importlib.util.spec_from_file_location("interaction_canary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spearman_perfect_and_reversed() -> None:
    """The local Spearman implementation handles monotonic rankings."""
    module = load_canary_module()
    assert module.spearman_correlation([1, 2, 3], [2, 4, 8]) == pytest.approx(1.0)
    assert module.spearman_correlation([1, 2, 3], [8, 4, 2]) == pytest.approx(-1.0)


def test_preregistered_decisions() -> None:
    """Decision labels follow the preregistered compound thresholds."""
    module = load_canary_module()
    assert module.classify_signal(0.11, 0.89, 0.70).startswith("A")
    assert module.classify_signal(0.11, 0.89, 0.90).startswith("B")
    assert module.classify_signal(0.02, 0.96, 0.95).startswith("C")
    assert module.classify_signal(0.05, 0.93, 0.80).startswith("D")
