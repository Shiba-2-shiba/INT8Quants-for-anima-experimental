from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Checkpoint key and exact matrix shape selected for quantization."""

    name: str
    shape: tuple[int, int]


__all__ = ["TensorSpec"]
