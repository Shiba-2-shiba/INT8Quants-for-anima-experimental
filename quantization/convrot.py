# Copyright Comfy Quants Project contributors
# SPDX-License-Identifier: GPL-3.0-only
# Modified for this Anima-only custom node on 2026-07-14.
"""Offline regular-Hadamard ConvRot for Anima INT8 weight export.

Derived from ``Comfy-Org/comfy-quants`` at commit
``1e0d481f24847c4914578f5468917902ad53ea46`` and its
``formats/convrot.py`` module.  The online activation rotation belongs to the
ComfyUI runtime and is deliberately outside this custom node's export surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    import torch


class ConvRotError(ValueError):
    """ConvRot input or environment does not satisfy the export contract."""


CONVROT_GROUP_SIZE: Final = 256

# Cache by size, normalized device text, and dtype to avoid rebuilding H256 for
# every selected weight. This preserves the upstream cache behavior.
_HADAMARD_CACHE: dict[tuple[int, str, Any], Any] = {}


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - ComfyUI always provides torch
        raise ConvRotError("torch is required for ConvRot weight rotation.") from exc
    return torch


def is_power_of_four(size: int) -> bool:
    """Return whether ``size`` is one of 4, 16, 64, 256, ... ."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 4:
        return False
    remaining = size
    while remaining > 1:
        if remaining % 4:
            return False
        remaining //= 4
    return True


def build_hadamard(
    size: int,
    *,
    device: str | Any = "cpu",
    dtype: Any = None,
) -> torch.Tensor:
    """Build the normalized regular-Hadamard matrix used by ConvRot.

    ``size`` must be a power of four.  The regular H4 has no all-ones column;
    larger matrices are Kronecker powers of H4 and are normalized by
    ``1 / sqrt(size)``.  The result is symmetric and orthogonal.
    """

    torch = _require_torch()
    if not is_power_of_four(size):
        raise ConvRotError(
            f"ConvRot Hadamard size must be a power of four, got {size!r}."
        )
    if dtype is None:
        dtype = torch.float32

    cache_key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    h4 = torch.tensor(
        (
            (1, 1, 1, -1),
            (1, 1, -1, 1),
            (1, -1, 1, 1),
            (-1, 1, 1, 1),
        ),
        dtype=dtype,
        device=device,
    )
    hadamard = h4
    current_size = 4
    while current_size < size:
        hadamard = torch.kron(hadamard, h4)
        current_size *= 4

    normalized = hadamard / (size**0.5)
    _HADAMARD_CACHE[cache_key] = normalized
    return normalized


def rotate_weight(
    weight: torch.Tensor,
    hadamard: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate a rank-2 Linear weight blockwise along its input dimension.

    For a weight shaped ``(out_features, in_features)``, the operation is
    ``weight.reshape(out, groups, group_size) @ hadamard.T`` and the result has
    the original shape.  It does not mutate either input tensor.
    """

    torch = _require_torch()
    if getattr(weight, "ndim", None) != 2:
        raise ConvRotError(
            f"ConvRot requires a rank-2 weight, got shape {getattr(weight, 'shape', None)}."
        )
    if not is_power_of_four(group_size):
        raise ConvRotError(
            f"ConvRot group_size must be a power of four, got {group_size!r}."
        )
    if getattr(hadamard, "ndim", None) != 2 or tuple(hadamard.shape) != (
        group_size,
        group_size,
    ):
        raise ConvRotError(
            "ConvRot Hadamard shape mismatch: "
            f"expected {(group_size, group_size)}, got {getattr(hadamard, 'shape', None)}."
        )

    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ConvRotError(
            f"in_features {in_features} is not divisible by ConvRot group_size {group_size}."
        )

    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    h_transposed = hadamard.T.to(dtype=weight.dtype, device=weight.device)
    rotated = torch.matmul(grouped, h_transposed)
    return rotated.reshape(out_features, in_features)


__all__ = [
    "CONVROT_GROUP_SIZE",
    "ConvRotError",
    "build_hadamard",
    "is_power_of_four",
    "rotate_weight",
]
