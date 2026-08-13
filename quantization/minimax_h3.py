# SPDX-License-Identifier: GPL-3.0-only
"""MiniMax H3 reference tensor contract for INT8 ConvRot export."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from .contracts import TensorSpec


class MiniMaxH3ContractError(ValueError):
    """A MiniMax H3 model or preset violates the supported reference contract."""


MiniMaxH3TensorSpec = TensorSpec

MINIMAX_H3_HIDDEN_SIZE: Final = 5376
MINIMAX_H3_BLOCK_COUNT: Final = 50
MINIMAX_H3_TOKEN_REFINER_BLOCK_COUNT: Final = 2
MINIMAX_H3_ATTENTION_INNER_SIZE: Final = 7168
MINIMAX_H3_FFN_HIDDEN_SIZE: Final = 14336
MINIMAX_H3_TEXT_DIM: Final = 5120
MINIMAX_H3_ADALN_CURVE_GRID: Final = 1025
MINIMAX_H3_TIME_EMBED_DIM: Final = 8

DEFAULT_MINIMAX_H3_QUANTIZATION_PRESET: Final = "strict_reference"
MINIMAX_H3_QUANTIZATION_PRESETS: Final = (
    DEFAULT_MINIMAX_H3_QUANTIZATION_PRESET,
)
EXPECTED_QUANTIZED_TENSORS = MappingProxyType(
    {DEFAULT_MINIMAX_H3_QUANTIZATION_PRESET: MINIMAX_H3_BLOCK_COUNT * 4}
)

_BLOCK_PATTERN = re.compile(r"^blocks\.(\d+)\.")
_TOKEN_REFINER_BLOCK_PATTERN = re.compile(r"^token_refiner\.blocks\.(\d+)\.")
_BLOCK_WEIGHT_SHAPES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("attn.qkv_proj", (21504, MINIMAX_H3_HIDDEN_SIZE)),
    ("attn.out_proj", (MINIMAX_H3_HIDDEN_SIZE, MINIMAX_H3_ATTENTION_INNER_SIZE)),
    ("mlp.fc1", (2 * MINIMAX_H3_FFN_HIDDEN_SIZE, MINIMAX_H3_HIDDEN_SIZE)),
    ("mlp.fc2", (MINIMAX_H3_HIDDEN_SIZE, MINIMAX_H3_FFN_HIDDEN_SIZE)),
)

_REFERENCE_SIGNATURES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("adaln_t_table", (MINIMAX_H3_ADALN_CURVE_GRID, MINIMAX_H3_TIME_EMBED_DIM)),
    ("rope.inv_freq", (16,)),
    ("video_patch_proj.weight", (MINIMAX_H3_HIDDEN_SIZE, 96)),
    ("audio_patch_proj.weight", (MINIMAX_H3_HIDDEN_SIZE, 32)),
    ("condition_proj.weight", (MINIMAX_H3_HIDDEN_SIZE, MINIMAX_H3_TEXT_DIM)),
    ("final_layer.video_out.weight", (96, MINIMAX_H3_HIDDEN_SIZE)),
    ("final_layer.audio_out.weight", (32, MINIMAX_H3_HIDDEN_SIZE)),
)


def _build_reference_source_tensor_shapes() -> dict[str, tuple[int, ...]]:
    shapes = dict(_REFERENCE_SIGNATURES)
    shapes.update(
        {
            "audio_patch_proj.bias": (MINIMAX_H3_HIDDEN_SIZE,),
            "condition_proj.bias": (MINIMAX_H3_HIDDEN_SIZE,),
            "final_layer.adaln_proj.linear.bias": (10752,),
            "final_layer.adaln_proj.linear.weight": (
                10752,
                MINIMAX_H3_TIME_EMBED_DIM,
            ),
            "final_layer.audio_out.bias": (32,),
            "final_layer.norm.weight": (MINIMAX_H3_HIDDEN_SIZE,),
            "final_layer.video_out.bias": (96,),
            "token_refiner.final_norm.weight": (MINIMAX_H3_HIDDEN_SIZE,),
            "video_patch_proj.bias": (MINIMAX_H3_HIDDEN_SIZE,),
        }
    )
    for block in range(MINIMAX_H3_BLOCK_COUNT):
        prefix = f"blocks.{block}"
        shapes.update(
            {
                f"{prefix}.adaln_proj.linear.bias": (96768,),
                f"{prefix}.adaln_proj.linear.weight": (
                    96768,
                    MINIMAX_H3_TIME_EMBED_DIM,
                ),
                f"{prefix}.attn.k_norm.weight": (128,),
                f"{prefix}.attn.q_norm.weight": (128,),
                f"{prefix}.norm1.weight": (MINIMAX_H3_HIDDEN_SIZE,),
                f"{prefix}.norm2.weight": (MINIMAX_H3_HIDDEN_SIZE,),
            }
        )
        for suffix, shape in _BLOCK_WEIGHT_SHAPES:
            shapes[f"{prefix}.{suffix}.weight"] = shape
    for block in range(MINIMAX_H3_TOKEN_REFINER_BLOCK_COUNT):
        prefix = f"token_refiner.blocks.{block}"
        shapes.update(
            {
                f"{prefix}.attn.k_norm.weight": (128,),
                f"{prefix}.attn.q_norm.weight": (128,),
                f"{prefix}.norm1.weight": (MINIMAX_H3_HIDDEN_SIZE,),
                f"{prefix}.norm2.weight": (MINIMAX_H3_HIDDEN_SIZE,),
            }
        )
        for suffix, shape in _BLOCK_WEIGHT_SHAPES:
            shapes[f"{prefix}.{suffix}.weight"] = shape
    return shapes


_REFERENCE_SOURCE_TENSOR_SHAPES = MappingProxyType(
    _build_reference_source_tensor_shapes()
)
EXPECTED_SOURCE_TENSOR_COUNT: Final = 532
EXPECTED_KEEP_TENSOR_COUNT: Final = 332

if len(_REFERENCE_SOURCE_TENSOR_SHAPES) != EXPECTED_SOURCE_TENSOR_COUNT:
    raise RuntimeError(
        "MiniMax H3 source manifest changed: "
        f"expected {EXPECTED_SOURCE_TENSOR_COUNT}, "
        f"got {len(_REFERENCE_SOURCE_TENSOR_SHAPES)}"
    )


def _validate_preset(quantization_preset: str) -> str:
    if quantization_preset not in MINIMAX_H3_QUANTIZATION_PRESETS:
        supported = ", ".join(MINIMAX_H3_QUANTIZATION_PRESETS)
        raise MiniMaxH3ContractError(
            "unsupported MiniMax H3 INT8 quantization preset "
            f"{quantization_preset!r}; expected one of: {supported}."
        )
    return quantization_preset


def _build_tensor_specs() -> tuple[MiniMaxH3TensorSpec, ...]:
    return tuple(
        MiniMaxH3TensorSpec(
            name=f"blocks.{block}.{suffix}.weight",
            shape=shape,
        )
        for block in range(MINIMAX_H3_BLOCK_COUNT)
        for suffix, shape in _BLOCK_WEIGHT_SHAPES
    )


_TENSOR_SPECS = _build_tensor_specs()


def expected_quantized_tensors(
    quantization_preset: str = DEFAULT_MINIMAX_H3_QUANTIZATION_PRESET,
) -> int:
    """Return the fixed selected tensor count for the reference preset."""

    preset = _validate_preset(quantization_preset)
    return EXPECTED_QUANTIZED_TENSORS[preset]


def get_minimax_h3_tensor_specs(
    quantization_preset: str = DEFAULT_MINIMAX_H3_QUANTIZATION_PRESET,
) -> tuple[MiniMaxH3TensorSpec, ...]:
    """Return the 200 main-DiT matrices selected by the official reference."""

    preset = _validate_preset(quantization_preset)
    expected = EXPECTED_QUANTIZED_TENSORS[preset]
    if len(_TENSOR_SPECS) != expected:
        raise MiniMaxH3ContractError(
            f"MiniMax H3 INT8 {preset} selection contract changed: "
            f"expected {expected}, got {len(_TENSOR_SPECS)}."
        )
    return _TENSOR_SPECS


def get_minimax_h3_tensor_names(
    quantization_preset: str = DEFAULT_MINIMAX_H3_QUANTIZATION_PRESET,
) -> tuple[str, ...]:
    """Return selected checkpoint keys in deterministic block order."""

    return tuple(
        spec.name for spec in get_minimax_h3_tensor_specs(quantization_preset)
    )


def get_minimax_h3_source_tensor_shapes() -> Mapping[str, tuple[int, ...]]:
    """Return the complete 532-tensor floating-point source manifest."""

    return _REFERENCE_SOURCE_TENSOR_SHAPES


def _tensor_shape(tensor: Any, *, name: str) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise MiniMaxH3ContractError(f"MiniMax H3 tensor {name!r} has no shape.")
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as exc:
        raise MiniMaxH3ContractError(
            f"MiniMax H3 tensor {name!r} has an invalid shape: {shape!r}."
        ) from exc


def _require_shape(
    state_dict: Mapping[str, Any],
    name: str,
    expected_shape: tuple[int, ...],
) -> None:
    tensor = state_dict.get(name)
    if tensor is None:
        raise MiniMaxH3ContractError(
            f"Not the supported MiniMax H3 reference model: missing {name}."
        )
    actual_shape = _tensor_shape(tensor, name=name)
    if actual_shape != expected_shape:
        raise MiniMaxH3ContractError(
            f"MiniMax H3 tensor shape mismatch for {name}: "
            f"expected {expected_shape}, got {actual_shape}."
        )


def _validate_block_coverage(
    state_dict: Mapping[str, Any],
    *,
    pattern: re.Pattern[str],
    count: int,
    label: str,
) -> None:
    actual = {
        int(match.group(1))
        for key in state_dict
        if (match := pattern.match(key)) is not None
    }
    expected = set(range(count))
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise MiniMaxH3ContractError(
            f"MiniMax H3 {label} contract mismatch: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)."
        )


def validate_minimax_h3_state_dict(
    state_dict: Mapping[str, Any],
    *,
    preset: str = DEFAULT_MINIMAX_H3_QUANTIZATION_PRESET,
    require_selected_tensors: bool = True,
) -> tuple[MiniMaxH3TensorSpec, ...]:
    """Validate the metadata-free curve-form official MiniMax H3 contract.

    Only the four main-DiT matrices per block are selected.  Token-refiner,
    conditioning, patch projection, AdaLN, norm, bias, RoPE, and final-layer
    tensors remain in their source precision, matching the official INT8 file.
    """

    specs = get_minimax_h3_tensor_specs(preset)
    if any(key.startswith("time_embedder.") for key in state_dict):
        raise MiniMaxH3ContractError(
            "Only the adaln_t_table curve-form MiniMax H3 reference is supported; "
            "time_embedder variants are rejected."
        )

    for name, shape in _REFERENCE_SIGNATURES:
        _require_shape(state_dict, name, shape)

    _validate_block_coverage(
        state_dict,
        pattern=_BLOCK_PATTERN,
        count=MINIMAX_H3_BLOCK_COUNT,
        label="block",
    )
    _validate_block_coverage(
        state_dict,
        pattern=_TOKEN_REFINER_BLOCK_PATTERN,
        count=MINIMAX_H3_TOKEN_REFINER_BLOCK_COUNT,
        label="token-refiner block",
    )

    expected_names = set(_REFERENCE_SOURCE_TENSOR_SHAPES)
    actual_names = set(state_dict)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise MiniMaxH3ContractError(
            "MiniMax H3 complete source tensor contract mismatch: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)."
        )

    for name, shape in _REFERENCE_SOURCE_TENSOR_SHAPES.items():
        _require_shape(state_dict, name, shape)

    if require_selected_tensors:
        for spec in specs:
            _require_shape(state_dict, spec.name, spec.shape)
    return specs


__all__ = [
    "DEFAULT_MINIMAX_H3_QUANTIZATION_PRESET",
    "EXPECTED_KEEP_TENSOR_COUNT",
    "EXPECTED_QUANTIZED_TENSORS",
    "EXPECTED_SOURCE_TENSOR_COUNT",
    "MINIMAX_H3_ADALN_CURVE_GRID",
    "MINIMAX_H3_ATTENTION_INNER_SIZE",
    "MINIMAX_H3_BLOCK_COUNT",
    "MINIMAX_H3_FFN_HIDDEN_SIZE",
    "MINIMAX_H3_HIDDEN_SIZE",
    "MINIMAX_H3_QUANTIZATION_PRESETS",
    "MINIMAX_H3_TEXT_DIM",
    "MINIMAX_H3_TIME_EMBED_DIM",
    "MINIMAX_H3_TOKEN_REFINER_BLOCK_COUNT",
    "MiniMaxH3ContractError",
    "MiniMaxH3TensorSpec",
    "expected_quantized_tensors",
    "get_minimax_h3_tensor_names",
    "get_minimax_h3_tensor_specs",
    "get_minimax_h3_source_tensor_shapes",
    "validate_minimax_h3_state_dict",
]
