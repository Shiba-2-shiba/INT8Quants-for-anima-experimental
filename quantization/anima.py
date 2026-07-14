# Copyright Comfy Quants Project contributors
# SPDX-License-Identifier: GPL-3.0-only
# Modified for this Anima-only custom node on 2026-07-14.
"""Self-contained Anima 2B tensor contract for INT8 ConvRot export.

Derived from ``Comfy-Org/comfy-quants`` at commit
``1e0d481f24847c4914578f5468917902ad53ea46``.  The source declarations lived in
``model_adapters/anima_contracts/anima.py`` and the preset selection lived in
``model_adapters/anima.py``.  This module intentionally does not carry over the
generic graph, policy, registry, or algorithm layers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final


class AnimaContractError(ValueError):
    """An Anima model or quantization preset violates the supported contract."""


@dataclass(frozen=True, slots=True)
class AnimaTensorSpec:
    """Name and exact matrix shape of one quantized Anima weight."""

    name: str
    shape: tuple[int, int]


ANIMA_MODEL_CHANNELS: Final = 2048
ANIMA_BLOCK_COUNT: Final = 28
ANIMA_NUM_HEADS: Final = 16
ANIMA_HEAD_DIM: Final = 128
ANIMA_CONTEXT_DIM: Final = 1024
ANIMA_ADALN_LORA_DIM: Final = 256
ANIMA_MLP_DIM: Final = 4 * ANIMA_MODEL_CHANNELS
ANIMA_ADALN_OUTPUT_DIM: Final = 3 * ANIMA_MODEL_CHANNELS

ANIMA_X_EMBEDDER_SIGNATURE: Final = "net.x_embedder.proj.1.weight"
ANIMA_LLM_ADAPTER_SIGNATURE: Final = (
    "net.llm_adapter.blocks.0.cross_attn.q_proj.weight"
)

DEFAULT_ANIMA_QUANTIZATION_PRESET: Final = "quality_keep"
ANIMA_QUANTIZATION_PRESETS: Final = (
    DEFAULT_ANIMA_QUANTIZATION_PRESET,
    "public_examples",
)
EXPECTED_QUANTIZED_TENSORS = MappingProxyType(
    {
        DEFAULT_ANIMA_QUANTIZATION_PRESET: 426,
        "public_examples": 448,
    }
)

_BLOCK_PATTERN = re.compile(r"^net\.blocks\.(\d+)\.")

# Order matches the static upstream graph and is part of deterministic export.
_BLOCK_WEIGHT_SHAPES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("self_attn.q_proj", (ANIMA_MODEL_CHANNELS, ANIMA_MODEL_CHANNELS)),
    ("self_attn.k_proj", (ANIMA_MODEL_CHANNELS, ANIMA_MODEL_CHANNELS)),
    ("self_attn.v_proj", (ANIMA_MODEL_CHANNELS, ANIMA_MODEL_CHANNELS)),
    ("self_attn.output_proj", (ANIMA_MODEL_CHANNELS, ANIMA_MODEL_CHANNELS)),
    ("cross_attn.q_proj", (ANIMA_MODEL_CHANNELS, ANIMA_MODEL_CHANNELS)),
    ("cross_attn.k_proj", (ANIMA_MODEL_CHANNELS, ANIMA_CONTEXT_DIM)),
    ("cross_attn.v_proj", (ANIMA_MODEL_CHANNELS, ANIMA_CONTEXT_DIM)),
    ("cross_attn.output_proj", (ANIMA_MODEL_CHANNELS, ANIMA_MODEL_CHANNELS)),
    ("mlp.layer1", (ANIMA_MLP_DIM, ANIMA_MODEL_CHANNELS)),
    ("mlp.layer2", (ANIMA_MODEL_CHANNELS, ANIMA_MLP_DIM)),
    (
        "adaln_modulation_self_attn.1",
        (ANIMA_ADALN_LORA_DIM, ANIMA_MODEL_CHANNELS),
    ),
    (
        "adaln_modulation_self_attn.2",
        (ANIMA_ADALN_OUTPUT_DIM, ANIMA_ADALN_LORA_DIM),
    ),
    (
        "adaln_modulation_cross_attn.1",
        (ANIMA_ADALN_LORA_DIM, ANIMA_MODEL_CHANNELS),
    ),
    (
        "adaln_modulation_cross_attn.2",
        (ANIMA_ADALN_OUTPUT_DIM, ANIMA_ADALN_LORA_DIM),
    ),
    ("adaln_modulation_mlp.1", (ANIMA_ADALN_LORA_DIM, ANIMA_MODEL_CHANNELS)),
    ("adaln_modulation_mlp.2", (ANIMA_ADALN_OUTPUT_DIM, ANIMA_ADALN_LORA_DIM)),
)

_ADALN_PREFIX = "adaln_modulation_"


def _validate_preset(quantization_preset: str) -> str:
    if quantization_preset not in ANIMA_QUANTIZATION_PRESETS:
        supported = ", ".join(ANIMA_QUANTIZATION_PRESETS)
        raise AnimaContractError(
            f"unsupported Anima INT8 quantization preset {quantization_preset!r}; "
            f"expected one of: {supported}."
        )
    return quantization_preset


def _build_tensor_specs(quantization_preset: str) -> tuple[AnimaTensorSpec, ...]:
    specs: list[AnimaTensorSpec] = []
    for block in range(ANIMA_BLOCK_COUNT):
        for suffix, shape in _BLOCK_WEIGHT_SHAPES:
            if quantization_preset == DEFAULT_ANIMA_QUANTIZATION_PRESET:
                if block == 0 or (block == 1 and suffix.startswith(_ADALN_PREFIX)):
                    continue
            specs.append(
                AnimaTensorSpec(
                    name=f"net.blocks.{block}.{suffix}.weight",
                    shape=shape,
                )
            )
    return tuple(specs)


_TENSOR_SPECS = MappingProxyType(
    {
        preset: _build_tensor_specs(preset)
        for preset in ANIMA_QUANTIZATION_PRESETS
    }
)


def expected_quantized_tensors(
    quantization_preset: str = DEFAULT_ANIMA_QUANTIZATION_PRESET,
) -> int:
    """Return the fixed selected tensor count for an Anima preset."""

    preset = _validate_preset(quantization_preset)
    return EXPECTED_QUANTIZED_TENSORS[preset]


def get_anima_tensor_specs(
    quantization_preset: str = DEFAULT_ANIMA_QUANTIZATION_PRESET,
) -> tuple[AnimaTensorSpec, ...]:
    """Return selected Anima weights in deterministic checkpoint order."""

    preset = _validate_preset(quantization_preset)
    specs = _TENSOR_SPECS[preset]
    expected = EXPECTED_QUANTIZED_TENSORS[preset]
    if len(specs) != expected:  # defensive guard against contract edits
        raise AnimaContractError(
            f"Anima 2B INT8 {preset} selection contract changed: "
            f"expected {expected}, got {len(specs)}."
        )
    return specs


def get_anima_tensor_names(
    quantization_preset: str = DEFAULT_ANIMA_QUANTIZATION_PRESET,
) -> tuple[str, ...]:
    """Return only the selected checkpoint keys for an Anima preset."""

    return tuple(spec.name for spec in get_anima_tensor_specs(quantization_preset))


def _tensor_shape(tensor: Any, *, name: str) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise AnimaContractError(f"Anima tensor {name!r} has no shape.")
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as exc:
        raise AnimaContractError(
            f"Anima tensor {name!r} has an invalid shape: {shape!r}."
        ) from exc


def validate_anima_state_dict(
    state_dict: Mapping[str, Any],
    *,
    preset: str = DEFAULT_ANIMA_QUANTIZATION_PRESET,
    require_selected_tensors: bool = True,
) -> tuple[AnimaTensorSpec, ...]:
    """Validate the stock ``net.*`` Anima 2B signature and selected weights.

    The two signature checks mirror ComfyUI model detection: Anima must contain
    its LLM adapter and ``x_embedder`` must report 2048 model channels.  Block
    coverage is fixed to 0..27.  Export callers should keep
    ``require_selected_tensors=True`` so every selected matrix is also checked
    against the static architecture shape before expensive rotation begins.
    """

    specs = get_anima_tensor_specs(preset)

    x_embedder = state_dict.get(ANIMA_X_EMBEDDER_SIGNATURE)
    if x_embedder is None:
        raise AnimaContractError(
            f"Not a supported Anima model: missing {ANIMA_X_EMBEDDER_SIGNATURE}."
        )
    x_shape = _tensor_shape(x_embedder, name=ANIMA_X_EMBEDDER_SIGNATURE)
    if not x_shape:
        raise AnimaContractError(
            f"Invalid shape for {ANIMA_X_EMBEDDER_SIGNATURE}: {x_shape}."
        )
    if x_shape[0] != ANIMA_MODEL_CHANNELS:
        variant = "Anima 14B" if x_shape[0] == 5120 else "unsupported model"
        raise AnimaContractError(
            f"{variant} detected with model_channels={x_shape[0]}; "
            f"only Anima 2B ({ANIMA_MODEL_CHANNELS}) is supported."
        )

    llm_signature = state_dict.get(ANIMA_LLM_ADAPTER_SIGNATURE)
    if llm_signature is None:
        raise AnimaContractError(
            f"Not a supported Anima model: missing {ANIMA_LLM_ADAPTER_SIGNATURE}."
        )
    _tensor_shape(llm_signature, name=ANIMA_LLM_ADAPTER_SIGNATURE)

    actual_blocks = {
        int(match.group(1))
        for key in state_dict
        if (match := _BLOCK_PATTERN.match(key)) is not None
    }
    expected_blocks = set(range(ANIMA_BLOCK_COUNT))
    if actual_blocks != expected_blocks:
        missing = sorted(expected_blocks - actual_blocks)
        unexpected = sorted(actual_blocks - expected_blocks)
        raise AnimaContractError(
            "Anima 2B block contract mismatch: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)."
        )

    if not require_selected_tensors:
        return specs

    for spec in specs:
        tensor = state_dict.get(spec.name)
        if tensor is None:
            raise AnimaContractError(f"missing selected Anima tensor: {spec.name}.")
        actual_shape = _tensor_shape(tensor, name=spec.name)
        if actual_shape != spec.shape:
            raise AnimaContractError(
                f"Anima tensor shape mismatch for {spec.name}: "
                f"expected {spec.shape}, got {actual_shape}."
            )
    return specs


__all__ = [
    "ANIMA_ADALN_LORA_DIM",
    "ANIMA_ADALN_OUTPUT_DIM",
    "ANIMA_BLOCK_COUNT",
    "ANIMA_CONTEXT_DIM",
    "ANIMA_HEAD_DIM",
    "ANIMA_LLM_ADAPTER_SIGNATURE",
    "ANIMA_MLP_DIM",
    "ANIMA_MODEL_CHANNELS",
    "ANIMA_NUM_HEADS",
    "ANIMA_QUANTIZATION_PRESETS",
    "ANIMA_X_EMBEDDER_SIGNATURE",
    "DEFAULT_ANIMA_QUANTIZATION_PRESET",
    "EXPECTED_QUANTIZED_TENSORS",
    "AnimaContractError",
    "AnimaTensorSpec",
    "expected_quantized_tensors",
    "get_anima_tensor_names",
    "get_anima_tensor_specs",
    "validate_anima_state_dict",
]
