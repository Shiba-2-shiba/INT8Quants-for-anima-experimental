# SPDX-License-Identifier: GPL-3.0-only
"""MOSS-Image 42-layer tensor contract in stock ComfyUI's native namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .contracts import TensorSpec


class MossImageContractError(ValueError):
    """A MOSS-Image input does not match the supported floating-point layout."""


MOSS_IMAGE_BLOCK_COUNT: Final = 42
DEFAULT_MOSS_IMAGE_QUANTIZATION_PRESET: Final = "quality_keep"
MOSS_IMAGE_QUANTIZATION_PRESETS: Final = (DEFAULT_MOSS_IMAGE_QUANTIZATION_PRESET,)
_SOURCE_DTYPES = frozenset({"torch.bfloat16", "torch.float16", "torch.float32"})


def get_moss_image_tensor_specs(
    quantization_preset: str = DEFAULT_MOSS_IMAGE_QUANTIZATION_PRESET,
    *,
    fused_qkv: bool = True,
) -> tuple[TensorSpec, ...]:
    if quantization_preset not in MOSS_IMAGE_QUANTIZATION_PRESETS:
        raise MossImageContractError(
            f"unsupported MOSS-Image INT8 quantization preset {quantization_preset!r}; "
            f"expected {DEFAULT_MOSS_IMAGE_QUANTIZATION_PRESET!r}."
        )
    projections = (
        (("qkv", (9216, 3072)),) if fused_qkv else
        tuple((part, (3072, 3072)) for part in ("q", "k", "v"))
    ) + (("o", (3072, 3072)), ("ffn_up_gate", (16384, 3072)), ("ffn_down", (3072, 8192)))
    return tuple(
        TensorSpec(f"layers.{layer}.proj.{name}.weight", shape)
        for layer in range(MOSS_IMAGE_BLOCK_COUNT)
        for name, shape in projections
    )


def get_moss_image_source_tensor_shapes(*, fused_qkv: bool = True) -> dict[str, tuple[int, ...]]:
    """Required learned tensors; RoPE and attention settings are optional buffers."""
    shapes = {spec.name: spec.shape for spec in get_moss_image_tensor_specs(fused_qkv=fused_qkv)}
    for layer in range(MOSS_IMAGE_BLOCK_COUNT):
        for norm, width in (("norm1", 3072), ("norm2", 3072), ("q_norm", 128), ("k_norm", 128)):
            shapes[f"layers.{layer}.{norm}.weight"] = (width,)
    linear_shapes = {
        "video_embedder": (3072, 128),
        "text_embedder.proj": (3072, 4096),
        "time_embedding.0": (3072, 256),
        "time_embedding.2": (3072, 3072),
        "shared_block_modulation": (18432, 3072),
        "video_head_modulation": (6144, 3072),
    }
    for block in range(2):
        prefix = f"text_embedder.blocks.{block}"
        for norm, width in (("norm1", 3072), ("norm2", 3072), ("q_norm", 128), ("k_norm", 128)):
            shapes[f"{prefix}.{norm}.weight"] = (width,)
        for name, shape in (
            ("qkv", (9216, 3072)), ("proj", (3072, 3072)),
            ("ffn_gate", (8192, 3072)), ("ffn_up", (8192, 3072)),
            ("ffn_down", (3072, 8192)),
        ):
            linear_shapes[f"{prefix}.{name}"] = shape
    for name, shape in linear_shapes.items():
        shapes[f"{name}.weight"] = shape
        shapes[f"{name}.bias"] = (shape[0],)
    shapes["final_norm_video.weight"] = (3072,)
    shapes["final_linear_video.weight"] = (128, 3072)
    return shapes


def validate_moss_image_state_dict(
    state_dict: Mapping[str, Any],
    *,
    preset: str = DEFAULT_MOSS_IMAGE_QUANTIZATION_PRESET,
    require_selected_tensors: bool = True,
) -> tuple[TensorSpec, ...]:
    """Validate all learned tensors, allowing the two native QKV layouts.

    Shape validation is intentionally complete even during model extraction:
    silently keeping a missing/wrong high-precision layer produces a broken model.
    """
    fused = "layers.0.proj.qkv.weight" in state_dict
    specs = get_moss_image_tensor_specs(preset, fused_qkv=fused)
    shapes = get_moss_image_source_tensor_shapes(fused_qkv=fused)
    attention_names = {
        f"layers.{layer}.comfy_attention.config" for layer in range(MOSS_IMAGE_BLOCK_COUNT)
    } | {f"text_embedder.blocks.{block}.comfy_attention.config" for block in range(2)}
    optional = attention_names | {"rope.bands"}
    missing = sorted(set(shapes) - state_dict.keys())
    unexpected = sorted(state_dict.keys() - set(shapes) - optional)
    if missing or unexpected:
        raise MossImageContractError(
            "Not a supported MOSS-Image 42-layer model (uniform fused or separate QKV required): "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}. "
            "Load the original floating-point model; already-quantized inputs are unsupported."
        )
    for name, tensor in state_dict.items():
        try:
            shape = tuple(int(dim) for dim in tensor.shape)
        except (AttributeError, TypeError, ValueError) as exc:
            raise MossImageContractError(f"MOSS-Image tensor {name!r} has no valid shape.") from exc
        dtype = str(getattr(tensor, "dtype", ""))
        if name in attention_names:
            if len(shape) != 1 or shape[0] <= 0 or dtype != "torch.uint8":
                raise MossImageContractError(f"Invalid MOSS-Image attention config tensor: {name}.")
            continue
        expected = (16,) if name == "rope.bands" else shapes[name]
        if shape != expected:
            raise MossImageContractError(
                f"MOSS-Image tensor shape mismatch for {name}: expected {expected}, got {shape}."
            )
        if dtype not in _SOURCE_DTYPES:
            raise MossImageContractError(
                f"MOSS-Image tensor dtype mismatch for {name}: expected BF16, FP16 or FP32, got {dtype}."
            )
    return specs
