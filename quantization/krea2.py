# SPDX-License-Identifier: GPL-3.0-only
"""Krea2 open-weight tensor contract for INT8 ConvRot export."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from .contracts import TensorSpec


class Krea2ContractError(ValueError):
    """A Krea2 model or quantization preset violates the supported contract."""


Krea2TensorSpec = TensorSpec

KREA2_FEATURES: Final = 6144
KREA2_BLOCK_COUNT: Final = 28
KREA2_HEADS: Final = 48
KREA2_KV_HEADS: Final = 12
KREA2_HEAD_DIM: Final = 128
KREA2_MLP_DIM: Final = 16384
KREA2_TEXT_DIM: Final = 2560
KREA2_TEXT_LAYERS: Final = 12

KREA2_FIRST_SIGNATURE: Final = "first.weight"
KREA2_TEXT_FUSION_SIGNATURE: Final = "txtfusion.projector.weight"
KREA2_TEXT_DIM_SIGNATURE: Final = "txtfusion.layerwise_blocks.0.prenorm.scale"

DEFAULT_KREA2_QUANTIZATION_PRESET: Final = "quality_keep"
KREA2_QUANTIZATION_PRESETS: Final = (DEFAULT_KREA2_QUANTIZATION_PRESET,)
EXPECTED_QUANTIZED_TENSORS = MappingProxyType(
    {DEFAULT_KREA2_QUANTIZATION_PRESET: KREA2_BLOCK_COUNT * 8}
)

_BLOCK_PATTERN = re.compile(r"^blocks\.(\d+)\.")
_BLOCK_WEIGHT_SHAPES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("attn.wq", (KREA2_FEATURES, KREA2_FEATURES)),
    ("attn.wk", (KREA2_KV_HEADS * KREA2_HEAD_DIM, KREA2_FEATURES)),
    ("attn.wv", (KREA2_KV_HEADS * KREA2_HEAD_DIM, KREA2_FEATURES)),
    ("attn.gate", (KREA2_FEATURES, KREA2_FEATURES)),
    ("attn.wo", (KREA2_FEATURES, KREA2_FEATURES)),
    ("mlp.gate", (KREA2_MLP_DIM, KREA2_FEATURES)),
    ("mlp.up", (KREA2_MLP_DIM, KREA2_FEATURES)),
    ("mlp.down", (KREA2_FEATURES, KREA2_MLP_DIM)),
)


def _validate_preset(quantization_preset: str) -> str:
    if quantization_preset not in KREA2_QUANTIZATION_PRESETS:
        supported = ", ".join(KREA2_QUANTIZATION_PRESETS)
        raise Krea2ContractError(
            f"unsupported Krea2 INT8 quantization preset {quantization_preset!r}; "
            f"expected one of: {supported}."
        )
    return quantization_preset


def _build_tensor_specs() -> tuple[Krea2TensorSpec, ...]:
    return tuple(
        Krea2TensorSpec(
            name=f"blocks.{block}.{suffix}.weight",
            shape=shape,
        )
        for block in range(KREA2_BLOCK_COUNT)
        for suffix, shape in _BLOCK_WEIGHT_SHAPES
    )


_TENSOR_SPECS = _build_tensor_specs()


def expected_quantized_tensors(
    quantization_preset: str = DEFAULT_KREA2_QUANTIZATION_PRESET,
) -> int:
    preset = _validate_preset(quantization_preset)
    return EXPECTED_QUANTIZED_TENSORS[preset]


def get_krea2_tensor_specs(
    quantization_preset: str = DEFAULT_KREA2_QUANTIZATION_PRESET,
) -> tuple[Krea2TensorSpec, ...]:
    preset = _validate_preset(quantization_preset)
    expected = EXPECTED_QUANTIZED_TENSORS[preset]
    if len(_TENSOR_SPECS) != expected:
        raise Krea2ContractError(
            f"Krea2 INT8 {preset} selection contract changed: "
            f"expected {expected}, got {len(_TENSOR_SPECS)}."
        )
    return _TENSOR_SPECS


def get_krea2_tensor_names(
    quantization_preset: str = DEFAULT_KREA2_QUANTIZATION_PRESET,
) -> tuple[str, ...]:
    return tuple(spec.name for spec in get_krea2_tensor_specs(quantization_preset))


def _tensor_shape(tensor: Any, *, name: str) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise Krea2ContractError(f"Krea2 tensor {name!r} has no shape.")
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as exc:
        raise Krea2ContractError(
            f"Krea2 tensor {name!r} has an invalid shape: {shape!r}."
        ) from exc


def _require_shape(
    state_dict: Mapping[str, Any],
    name: str,
    expected_shape: tuple[int, ...],
) -> None:
    tensor = state_dict.get(name)
    if tensor is None:
        raise Krea2ContractError(f"Not a supported Krea2 model: missing {name}.")
    actual_shape = _tensor_shape(tensor, name=name)
    if actual_shape != expected_shape:
        raise Krea2ContractError(
            f"Krea2 tensor shape mismatch for {name}: "
            f"expected {expected_shape}, got {actual_shape}."
        )


def validate_krea2_state_dict(
    state_dict: Mapping[str, Any],
    *,
    preset: str = DEFAULT_KREA2_QUANTIZATION_PRESET,
    require_selected_tensors: bool = True,
) -> tuple[Krea2TensorSpec, ...]:
    """Validate the native ComfyUI Krea2 namespace and open-weight shape contract."""

    specs = get_krea2_tensor_specs(preset)
    _require_shape(state_dict, KREA2_FIRST_SIGNATURE, (KREA2_FEATURES, 64))
    _require_shape(
        state_dict,
        KREA2_TEXT_FUSION_SIGNATURE,
        (1, KREA2_TEXT_LAYERS),
    )
    _require_shape(
        state_dict,
        KREA2_TEXT_DIM_SIGNATURE,
        (KREA2_TEXT_DIM,),
    )

    actual_blocks = {
        int(match.group(1))
        for key in state_dict
        if (match := _BLOCK_PATTERN.match(key)) is not None
    }
    expected_blocks = set(range(KREA2_BLOCK_COUNT))
    if actual_blocks != expected_blocks:
        missing = sorted(expected_blocks - actual_blocks)
        unexpected = sorted(actual_blocks - expected_blocks)
        raise Krea2ContractError(
            "Krea2 block contract mismatch: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)."
        )

    if not require_selected_tensors:
        return specs

    for spec in specs:
        _require_shape(state_dict, spec.name, spec.shape)
    return specs


__all__ = [
    "DEFAULT_KREA2_QUANTIZATION_PRESET",
    "EXPECTED_QUANTIZED_TENSORS",
    "KREA2_BLOCK_COUNT",
    "KREA2_FEATURES",
    "KREA2_FIRST_SIGNATURE",
    "KREA2_HEADS",
    "KREA2_HEAD_DIM",
    "KREA2_KV_HEADS",
    "KREA2_MLP_DIM",
    "KREA2_QUANTIZATION_PRESETS",
    "KREA2_TEXT_DIM",
    "KREA2_TEXT_DIM_SIGNATURE",
    "KREA2_TEXT_FUSION_SIGNATURE",
    "KREA2_TEXT_LAYERS",
    "Krea2ContractError",
    "Krea2TensorSpec",
    "expected_quantized_tensors",
    "get_krea2_tensor_names",
    "get_krea2_tensor_specs",
    "validate_krea2_state_dict",
]
