# Copyright Comfy Quants Project contributors
# SPDX-License-Identifier: GPL-3.0-only
# Modified for this custom node on 2026-07-15.
#
# Derived from Comfy-Org/comfy-quants at
# 1e0d481f24847c4914578f5468917902ad53ea46, principally
# comfy_quants/backends/int8_tensorwise_model_export.py and comfy_quants/api.py.
# This version is limited to the in-memory Anima and Krea2 INT8 ConvRot paths.

"""Write self-contained INT8 ConvRot checkpoints for stock ComfyUI."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .anima import (
    DEFAULT_ANIMA_QUANTIZATION_PRESET,
    validate_anima_state_dict,
)
from .contracts import TensorSpec
from .convrot import CONVROT_GROUP_SIZE, build_hadamard, is_power_of_four, rotate_weight
from .krea2 import (
    DEFAULT_KREA2_QUANTIZATION_PRESET,
    validate_krea2_state_dict,
)

INT8_TENSORWISE_FORMAT_NAME = "int8_tensorwise"
_ARTIFACT_CONTRACT = "int8_tensorwise_inference_checkpoint.v1"


class QuantizationExportError(RuntimeError):
    """Raised when the self-contained checkpoint writer cannot export safely."""


@dataclass
class AnimaInt8ExportReport:
    """Serializable summary of an in-memory diffusion checkpoint export."""

    source_checkpoint: str
    output_checkpoint: str
    quantized_tensor_count: int
    copied_tensor_count: int
    output_tensor_count: int
    schema_version: str = "int8_tensorwise_checkpoint_export_report.v1"
    status: str = "model_written"
    source_format: str = "state_dict"
    target_format: str = "safetensors"
    requested_device: str = "cpu"
    execution_device: str = "cpu"
    output_tensor_device: str = "cpu"
    artifact_target: str = "comfyui_diffusion_model"
    target_dtype: str = INT8_TENSORWISE_FORMAT_NAME
    quant_storage_dtype: str = "int8"
    scale_dtype: str = "fp32"
    scale_granularity: str = "per_channel"
    scale_axis: str | int | None = "out_features"
    convrot: bool = True
    convrot_groupsize: int = CONVROT_GROUP_SIZE
    rotated_tensor_count: int = 0
    nonrotated_tensor_count: int = 0
    quantization_preset: str = DEFAULT_ANIMA_QUANTIZATION_PRESET
    source_layout: str = "in_memory"
    source_tensor_count: int = 0
    source_file_count: int = 0
    selected_source_files: dict[str, int] = field(default_factory=dict)
    missing_tensor_count: int = 0
    missing_tensors: list[str] = field(default_factory=list)
    quant_metadata_tensor_count: int = 0
    scale_tensor_count: int = 0
    output_bytes: int = 0
    output_hash: str = ""
    output_hash_state: str = "not_requested"
    config_path: str | None = None
    cuda_max_memory_allocated_bytes: int | None = None
    cuda_max_memory_reserved_bytes: int | None = None
    dtype_counts: dict[str, int] = field(default_factory=dict)
    written_files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return plain Python values suitable for JSON or YAML serialization."""

        return asdict(self)


Int8ExportReport = AnimaInt8ExportReport


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - only reached without ComfyUI deps
        raise QuantizationExportError("torch is required for INT8 checkpoint export") from exc
    return torch


def _require_safetensors_save_file():
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - only reached without declared deps
        raise QuantizationExportError("safetensors is required for INT8 checkpoint export") from exc
    return save_file


def _emit_progress(
    progress: Callable[[dict[str, Any]], None] | None,
    **event: Any,
) -> None:
    if progress is not None:
        progress(event)


def _metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_torch_device(device: str):
    torch = _require_torch()
    requested = str(device or "cpu")
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        device_obj = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise QuantizationExportError(f"invalid torch device {device!r}") from exc
    if device_obj.type == "cuda":
        if not torch.cuda.is_available():
            raise QuantizationExportError(
                f"CUDA device requested but torch.cuda is not available: {device}"
            )
        index = torch.cuda.current_device() if device_obj.index is None else int(device_obj.index)
        try:
            torch.cuda.set_device(index)
        except (RuntimeError, ValueError) as exc:
            raise QuantizationExportError(f"unable to select CUDA device cuda:{index}") from exc
        return torch.device(f"cuda:{index}")
    return device_obj


def _layer_name_from_weight(weight_name: str) -> str:
    suffix = ".weight"
    if not weight_name.endswith(suffix):
        raise QuantizationExportError(f"selected tensor is not a module weight: {weight_name}")
    return weight_name[: -len(suffix)]


def _marker_tensor(*, convrot: bool, group_size: int):
    """Encode the marker with stock ComfyUI's key order and JSON separators."""

    torch = _require_torch()
    marker: dict[str, str | bool | int] = {"format": INT8_TENSORWISE_FORMAT_NAME}
    if convrot:
        marker["convrot"] = True
        marker["convrot_groupsize"] = int(group_size)
    data = json.dumps(marker).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8, device="cpu")


def _quantize_int8_tensorwise_per_row(
    tensor: Any,
    *,
    convrot: bool,
    group_size: int,
) -> tuple[Any, Any, bool]:
    """Return ``(int8 weight, fp32 row scale, was_rotated)`` without mutation."""

    torch = _require_torch()
    weight = tensor.detach()
    if weight.dim() != 2:
        raise QuantizationExportError("int8_tensorwise export requires a rank-2 weight tensor")
    if not weight.is_floating_point():
        raise QuantizationExportError(
            "int8_tensorwise export requires a floating-point weight tensor"
        )

    rotated = False
    if convrot and int(weight.shape[1]) % group_size == 0:
        hadamard = build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
        weight = rotate_weight(weight, hadamard, group_size)
        rotated = True

    abs_max = weight.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    scale_math = scale.to(device=weight.device, dtype=weight.dtype)
    scale_math = torch.where(
        scale_math == 0,
        torch.full_like(scale_math, torch.finfo(weight.dtype).tiny),
        scale_math,
    )
    quantized = (weight / scale_math).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return quantized.contiguous(), scale.to(torch.float32).contiguous(), rotated


def _validate_state_dict_values(state_dict: Mapping[str, Any]) -> None:
    torch = _require_torch()
    if not isinstance(state_dict, Mapping):
        raise QuantizationExportError("state_dict must be a Mapping[str, torch.Tensor]")
    invalid_names = [name for name in state_dict if not isinstance(name, str)]
    if invalid_names:
        raise QuantizationExportError("state_dict keys must be strings")
    invalid_values = [
        name for name, tensor in state_dict.items() if not isinstance(tensor, torch.Tensor)
    ]
    if invalid_values:
        preview = ", ".join(str(name) for name in invalid_values[:8])
        raise QuantizationExportError(
            f"state_dict values must be torch.Tensor instances: {preview}"
        )


def _validated_specs(
    tensor_specs: Sequence[TensorSpec],
) -> dict[str, TensorSpec]:
    selected: dict[str, TensorSpec] = {}
    for spec in tensor_specs:
        name = spec.name
        if name in selected:
            raise QuantizationExportError(f"duplicate selected tensor in export contract: {name}")
        _layer_name_from_weight(name)
        if len(spec.shape) != 2 or any(int(dimension) <= 0 for dimension in spec.shape):
            raise QuantizationExportError(
                f"invalid selected tensor shape for {name}: {spec.shape}"
            )
        selected[name] = spec
    return selected


def _validate_selected_tensor(
    name: str,
    tensor: Any,
    spec: TensorSpec,
    *,
    validate_source_dtype: bool,
) -> None:
    torch = _require_torch()
    actual_shape = tuple(int(dimension) for dimension in tensor.shape)
    expected_shape = tuple(int(dimension) for dimension in spec.shape)
    if tensor.dim() != 2:
        raise QuantizationExportError(
            f"source tensor rank mismatch for {name}: expected rank 2, got rank {tensor.dim()}"
        )
    if actual_shape != expected_shape:
        raise QuantizationExportError(
            f"source tensor shape mismatch for {name}: expected {expected_shape}, got {actual_shape}"
        )
    if not tensor.is_floating_point():
        raise QuantizationExportError(
            f"source tensor dtype mismatch for {name}: expected a floating-point tensor, "
            f"got {tensor.dtype}"
        )
    if validate_source_dtype and tensor.dtype not in (torch.bfloat16, torch.float16):
        raise QuantizationExportError(
            f"source tensor dtype mismatch for {name}: expected bfloat16 or float16, "
            f"got {tensor.dtype}"
        )


def _check_generated_tensor_collisions(
    source_names: Sequence[str],
    selected_names: Sequence[str],
) -> None:
    generated: set[str] = set()
    for selected_name in selected_names:
        layer = _layer_name_from_weight(selected_name)
        generated.add(f"{layer}.weight_scale")
        generated.add(f"{layer}.comfy_quant")
    collisions = sorted(set(source_names) & generated)
    if collisions:
        preview = ", ".join(collisions[:8])
        raise QuantizationExportError(
            f"source already contains generated quantization tensors: {preview}"
        )


def _write_int8_convrot_checkpoint_from_specs(
    *,
    state_dict: Mapping[str, Any],
    output_checkpoint: str | Path,
    tensor_specs: Sequence[TensorSpec],
    convrot: bool = True,
    convrot_groupsize: int = CONVROT_GROUP_SIZE,
    device: str = "cpu",
    strict: bool = True,
    validate_source_dtype: bool = True,
    require_all_rotated: bool = True,
    hash_output: bool = False,
    metadata: Mapping[str, Any] | None = None,
    quantization_preset: str = DEFAULT_ANIMA_QUANTIZATION_PRESET,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Int8ExportReport:
    """Write one model contract without coupling the quantization math to its family."""

    save_file = _require_safetensors_save_file()
    torch = _require_torch()
    _validate_state_dict_values(state_dict)
    if convrot_groupsize <= 0:
        raise QuantizationExportError("convrot_groupsize must be positive")
    if convrot and not is_power_of_four(convrot_groupsize):
        raise QuantizationExportError(
            f"ConvRot groupsize must be a power of four, got {convrot_groupsize}"
        )
    if require_all_rotated and not convrot:
        raise QuantizationExportError("require_all_rotated=True requires convrot=True")

    selected = _validated_specs(tensor_specs)
    selected_names = list(selected)
    source_names = list(state_dict)
    missing = sorted(set(selected_names) - set(source_names))
    if missing and strict:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise QuantizationExportError(
            f"source state_dict is missing selected tensors: {preview}{suffix}"
        )
    _check_generated_tensor_collisions(source_names, selected_names)

    for name in selected_names:
        tensor = state_dict.get(name)
        if tensor is None:
            continue
        _validate_selected_tensor(
            name,
            tensor,
            selected[name],
            validate_source_dtype=bool(strict and validate_source_dtype),
        )
        if require_all_rotated and int(tensor.shape[1]) % convrot_groupsize != 0:
            raise QuantizationExportError(
                f"ConvRot is required for every selected tensor, but {name} "
                f"has in_features={tensor.shape[1]} which is not divisible by groupsize "
                f"{convrot_groupsize}"
            )

    requested_device = str(device or "cpu")
    execution_device_obj = _resolve_torch_device(requested_device)
    execution_device = str(execution_device_obj)
    if execution_device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(execution_device_obj)

    output_path = Path(output_checkpoint).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _emit_progress(
        progress,
        stage="prepare",
        target_dtype=INT8_TENSORWISE_FORMAT_NAME,
        requested_device=requested_device,
        execution_device=execution_device,
        source_file_count=0,
        selected_tensor_count=len(selected_names),
        convrot=bool(convrot),
    )

    output_tensors: dict[str, Any] = {}
    dtype_counts: dict[str, int] = {}
    copied = 0
    quantized = 0
    rotated_count = 0
    copied_storage_ids: set[tuple[int, int]] = set()
    for name, source_tensor in state_dict.items():
        spec = selected.get(name)
        if spec is None:
            # Reuse ordinary contiguous CPU storage. Clone only aliases, which
            # safetensors rejects when two output names share one storage block.
            stored = source_tensor.detach().to(device="cpu").contiguous()
            storage = stored.untyped_storage()
            storage_id = (int(storage.data_ptr()), int(storage.nbytes()))
            if storage_id in copied_storage_ids:
                stored = stored.clone()
                storage = stored.untyped_storage()
                storage_id = (int(storage.data_ptr()), int(storage.nbytes()))
            copied_storage_ids.add(storage_id)
            output_tensors[name] = stored
            dtype_name = str(stored.dtype).removeprefix("torch.")
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
            copied += 1
            continue

        tensor_for_quant = source_tensor
        if tensor_for_quant.device != execution_device_obj:
            tensor_for_quant = tensor_for_quant.to(
                device=execution_device_obj,
                non_blocking=execution_device_obj.type == "cuda",
            )
        if not bool(torch.isfinite(tensor_for_quant).all().item()):
            raise QuantizationExportError(
                f"source tensor contains non-finite values: {name}"
            )
        quantized_weight, scale, rotated = _quantize_int8_tensorwise_per_row(
            tensor_for_quant,
            convrot=convrot,
            group_size=convrot_groupsize,
        )
        if require_all_rotated and not rotated:
            raise QuantizationExportError(
                f"ConvRot was not applied to required selected tensor: {name}"
            )
        layer = _layer_name_from_weight(name)
        output_tensors[name] = quantized_weight.detach().to(device="cpu").contiguous()
        output_tensors[f"{layer}.weight_scale"] = scale.detach().to(device="cpu").contiguous()
        output_tensors[f"{layer}.comfy_quant"] = _marker_tensor(
            convrot=rotated,
            group_size=convrot_groupsize,
        )
        dtype_counts["int8"] = dtype_counts.get("int8", 0) + 1
        dtype_counts["float32"] = dtype_counts.get("float32", 0) + 1
        dtype_counts["uint8"] = dtype_counts.get("uint8", 0) + 1
        quantized += 1
        if rotated:
            rotated_count += 1
        _emit_progress(
            progress,
            stage="quantize_tensor",
            target_dtype=INT8_TENSORWISE_FORMAT_NAME,
            tensor_name=name,
            quantized_tensor_count=quantized,
            selected_tensor_count=len(selected_names),
            convrot=rotated,
            execution_device=execution_device,
        )

    output_metadata: dict[str, Any] = dict(metadata or {})
    output_metadata.update(
        {
            "artifact_target": "comfyui_diffusion_model",
            "artifact_contract": _ARTIFACT_CONTRACT,
            "target_dtype": INT8_TENSORWISE_FORMAT_NAME,
            "quant_storage_dtype": "int8",
            "scale_granularity": "per_channel",
            "scale_axis": "out_features",
            "convrot": bool(convrot),
            "convrot_groupsize": int(convrot_groupsize),
            "quantized_tensor_count": quantized,
        }
    )

    cuda_peak_allocated: int | None = None
    cuda_peak_reserved: int | None = None
    if execution_device_obj.type == "cuda":
        torch.cuda.synchronize(execution_device_obj)
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated(execution_device_obj))
        cuda_peak_reserved = int(torch.cuda.max_memory_reserved(execution_device_obj))
        torch.cuda.empty_cache()

    _emit_progress(
        progress,
        stage="save_checkpoint",
        output_checkpoint=str(output_path),
        output_tensor_count=len(output_tensors),
        output_tensor_device="cpu",
    )
    save_file(
        output_tensors,
        str(output_path),
        metadata={str(key): _metadata_value(value) for key, value in output_metadata.items()},
    )

    output_hash = ""
    output_hash_state = "not_requested"
    if hash_output:
        _emit_progress(progress, stage="hash_checkpoint", output_checkpoint=str(output_path))
        output_hash = _hash_file(output_path)
        output_hash_state = "written"
    output_bytes = output_path.stat().st_size
    written_files = [
        {
            "path": str(output_path),
            "kind": "int8_tensorwise_inference_checkpoint",
            "state": "written",
            "tensor_count": len(output_tensors),
            "bytes": output_bytes,
            "hash": output_hash,
            "hash_state": output_hash_state,
        }
    ]
    return Int8ExportReport(
        source_checkpoint="<state_dict>",
        output_checkpoint=str(output_path),
        quantized_tensor_count=quantized,
        copied_tensor_count=copied,
        output_tensor_count=len(output_tensors),
        requested_device=requested_device,
        execution_device=execution_device,
        convrot=bool(convrot),
        convrot_groupsize=int(convrot_groupsize),
        rotated_tensor_count=rotated_count,
        nonrotated_tensor_count=quantized - rotated_count,
        quantization_preset=quantization_preset,
        source_tensor_count=len(state_dict),
        missing_tensor_count=len(missing),
        missing_tensors=missing,
        quant_metadata_tensor_count=quantized,
        scale_tensor_count=quantized,
        output_bytes=output_bytes,
        output_hash=output_hash,
        output_hash_state=output_hash_state,
        cuda_max_memory_allocated_bytes=cuda_peak_allocated,
        cuda_max_memory_reserved_bytes=cuda_peak_reserved,
        dtype_counts=dict(sorted(dtype_counts.items())),
        written_files=written_files,
    )


def _export_int8_convrot_from_state_dict(
    *,
    state_dict: Mapping[str, Any],
    output_checkpoint: str | Path,
    family: str,
    validator: Callable[..., Sequence[TensorSpec]],
    convrot: bool = True,
    convrot_groupsize: int = CONVROT_GROUP_SIZE,
    device: str = "cpu",
    strict: bool = True,
    validate_source_dtype: bool = True,
    quantization_preset: str,
    require_all_rotated: bool = True,
    hash_output: bool = False,
    metadata: Mapping[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Int8ExportReport:
    if convrot_groupsize <= 0:
        raise QuantizationExportError("convrot_groupsize must be positive")

    tensor_specs = validator(
        state_dict,
        preset=quantization_preset,
        require_selected_tensors=bool(strict),
    )
    output_metadata = dict(metadata or {})
    output_metadata.setdefault("model_family", family)
    output_metadata.setdefault("quantization_preset", quantization_preset)
    return _write_int8_convrot_checkpoint_from_specs(
        state_dict=state_dict,
        output_checkpoint=output_checkpoint,
        tensor_specs=tensor_specs,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
        device=device,
        strict=strict,
        validate_source_dtype=validate_source_dtype,
        require_all_rotated=require_all_rotated,
        hash_output=hash_output,
        metadata=output_metadata,
        quantization_preset=quantization_preset,
        progress=progress,
    )


def export_anima_int8_convrot_from_state_dict(
    *,
    state_dict: Mapping[str, Any],
    output_checkpoint: str | Path,
    family: str = "anima",
    convrot: bool = True,
    convrot_groupsize: int = CONVROT_GROUP_SIZE,
    device: str = "cpu",
    strict: bool = True,
    validate_source_dtype: bool = True,
    quantization_preset: str = DEFAULT_ANIMA_QUANTIZATION_PRESET,
    require_all_rotated: bool = True,
    hash_output: bool = False,
    metadata: Mapping[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Int8ExportReport:
    """Export an Anima 2B state dict as a stock-ComfyUI INT8 checkpoint."""

    if family != "anima":
        detail = (
            "Anima 14B is not supported"
            if family == "anima_14b"
            else "only family='anima' is supported"
        )
        raise QuantizationExportError(
            f"unsupported INT8 state_dict family {family!r}: {detail}"
        )
    return _export_int8_convrot_from_state_dict(
        state_dict=state_dict,
        output_checkpoint=output_checkpoint,
        family=family,
        validator=validate_anima_state_dict,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
        device=device,
        strict=strict,
        validate_source_dtype=validate_source_dtype,
        quantization_preset=quantization_preset,
        require_all_rotated=require_all_rotated,
        hash_output=hash_output,
        metadata=metadata,
        progress=progress,
    )


def export_krea2_int8_convrot_from_state_dict(
    *,
    state_dict: Mapping[str, Any],
    output_checkpoint: str | Path,
    family: str = "krea2",
    convrot: bool = True,
    convrot_groupsize: int = CONVROT_GROUP_SIZE,
    device: str = "cpu",
    strict: bool = True,
    validate_source_dtype: bool = True,
    quantization_preset: str = DEFAULT_KREA2_QUANTIZATION_PRESET,
    require_all_rotated: bool = True,
    hash_output: bool = False,
    metadata: Mapping[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Int8ExportReport:
    """Export a native-namespace Krea2 state dict as a stock-ComfyUI INT8 checkpoint."""

    if family != "krea2":
        raise QuantizationExportError(
            f"unsupported INT8 state_dict family {family!r}: only family='krea2' is supported"
        )
    return _export_int8_convrot_from_state_dict(
        state_dict=state_dict,
        output_checkpoint=output_checkpoint,
        family=family,
        validator=validate_krea2_state_dict,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
        device=device,
        strict=strict,
        validate_source_dtype=validate_source_dtype,
        require_all_rotated=require_all_rotated,
        hash_output=hash_output,
        metadata=metadata,
        quantization_preset=quantization_preset,
        progress=progress,
    )


__all__ = [
    "AnimaInt8ExportReport",
    "Int8ExportReport",
    "QuantizationExportError",
    "export_anima_int8_convrot_from_state_dict",
    "export_krea2_int8_convrot_from_state_dict",
]
