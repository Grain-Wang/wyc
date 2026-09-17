"""Toy-only transport validation; real-model H2 execution is deferred to A800."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def toy_jvp_transport(
    function: Callable[[Tensor], Tensor],
    parent_state: Tensor,
    local_residual: Tensor,
    projection: Tensor,
) -> Tensor:
    """Project a toy JVP and validate the intended transport tensor contract."""
    if parent_state.shape != local_residual.shape:
        raise ValueError("parent_state and local_residual must have the same shape.")
    _, tangent = torch.func.jvp(
        function,
        (parent_state,),
        (local_residual,),
    )
    flattened = tangent.reshape(-1)
    if projection.ndim != 2 or projection.shape[1] != flattened.numel():
        raise ValueError("projection must have shape [sketch_dim, output_elements].")
    return projection @ flattened
