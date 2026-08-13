from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .quantization import (
    QuantizationExportError,
    export_anima_int8_convrot_from_state_dict,
    export_krea2_int8_convrot_from_state_dict,
    export_minimax_h3_int8_convrot_from_state_dict,
)
from .quantization.export import DEFAULT_STREAMING_CHUNK_BYTES
from .quantization.anima import (
    ANIMA_QUANTIZATION_PRESETS,
    DEFAULT_ANIMA_QUANTIZATION_PRESET,
    AnimaContractError,
    expected_quantized_tensors as _expected_quantized_tensors,
    validate_anima_state_dict,
)
from .quantization.krea2 import (
    KREA2_QUANTIZATION_PRESETS,
    DEFAULT_KREA2_QUANTIZATION_PRESET,
    Krea2ContractError,
    expected_quantized_tensors as _expected_krea2_quantized_tensors,
    validate_krea2_state_dict,
)
from .quantization.minimax_h3 import (
    EXPECTED_KEEP_TENSOR_COUNT,
    MINIMAX_H3_QUANTIZATION_PRESETS,
    DEFAULT_MINIMAX_H3_QUANTIZATION_PRESET,
    MiniMaxH3ContractError,
    expected_quantized_tensors as _expected_minimax_h3_quantized_tensors,
    get_minimax_h3_tensor_names,
    validate_minimax_h3_state_dict,
)

DIFFUSION_PREFIX = "diffusion_model."
DEFAULT_QUANTIZATION_PRESET = DEFAULT_ANIMA_QUANTIZATION_PRESET
QUANTIZATION_PRESETS = ANIMA_QUANTIZATION_PRESETS
DEFAULT_KREA2_PRESET = DEFAULT_KREA2_QUANTIZATION_PRESET
KREA2_PRESETS = KREA2_QUANTIZATION_PRESETS
DEFAULT_MINIMAX_H3_PRESET = DEFAULT_MINIMAX_H3_QUANTIZATION_PRESET
MINIMAX_H3_PRESETS = MINIMAX_H3_QUANTIZATION_PRESETS
DISK_HEADROOM_BYTES = 64 * 1024 * 1024
_INVALID_FILENAME_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_MINIMAX_H3_REFERENCE_CONFIG = {
    "image_model": "minimax_h3",
    "num_layers": 50,
    "token_refiner_num_layers": 2,
    "hidden_size": 5376,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "attention_head_dim": 128,
    "num_attention_heads": 56,
    "ffn_hidden_size": 14336,
    "text_dim": 5120,
    "adaln_curve_grid": 1025,
    "time_embed_dim": 8,
    "rope_inv_freq_len": 16,
}
_MINIMAX_H3_RUNTIME_CONFIG_KEYS = frozenset({"dtype"})
_MINIMAX_H3_SOURCE_DTYPES = frozenset({"torch.bfloat16", "torch.float32"})
MINIMAX_H3_MEMORY_HEADROOM_RATIO = 1.20
MINIMAX_H3_QUANT_MARKER_BYTES = 72
MINIMAX_H3_WORKSPACE_COPIES = 6


class QuantizationNodeError(RuntimeError):
    """Actionable input or environment error raised by the custom node."""


def expected_quantized_tensors(quantization_preset: str) -> int:
    try:
        return _expected_quantized_tensors(quantization_preset)
    except AnimaContractError as exc:
        supported = ", ".join(QUANTIZATION_PRESETS)
        raise QuantizationNodeError(
            f"Unsupported quantization_preset {quantization_preset!r}; expected one of: {supported}."
        ) from exc


def expected_krea2_quantized_tensors(quantization_preset: str) -> int:
    try:
        return _expected_krea2_quantized_tensors(quantization_preset)
    except Krea2ContractError as exc:
        supported = ", ".join(KREA2_PRESETS)
        raise QuantizationNodeError(
            f"Unsupported quantization_preset {quantization_preset!r}; expected one of: {supported}."
        ) from exc


def expected_minimax_h3_quantized_tensors(quantization_preset: str) -> int:
    try:
        return _expected_minimax_h3_quantized_tensors(quantization_preset)
    except MiniMaxH3ContractError as exc:
        supported = ", ".join(MINIMAX_H3_PRESETS)
        raise QuantizationNodeError(
            f"Unsupported quantization_preset {quantization_preset!r}; expected one of: {supported}."
        ) from exc


@dataclass(frozen=True)
class OutputPaths:
    checkpoint: Path
    report: Path


@dataclass(frozen=True)
class MiniMaxH3MemoryEstimate:
    source_bytes: int
    output_file_bytes: int
    workspace_bytes: int
    estimated_peak_bytes: int
    required_additional_bytes: int


def _is_int8_tensor(tensor: Any) -> bool:
    return str(getattr(tensor, "dtype", "")) == "torch.int8"


def _extract_diffusion_state_dict(
    model: Any,
    *,
    node_display_name: str,
    normalize_key: Callable[[str], str],
) -> dict[str, Any]:
    model_state_dict = getattr(model, "model_state_dict", None)
    if not callable(model_state_dict):
        raise QuantizationNodeError("The model input is not a ComfyUI ModelPatcher-compatible MODEL.")

    for patch_field in (
        "patches",
        "object_patches",
        "weight_wrapper_patches",
        "hook_patches",
        "hook_patches_backup",
        "cached_hook_patches",
        "current_hooks",
        "injections",
    ):
        if getattr(model, patch_field, None):
            raise QuantizationNodeError(
                f"The input MODEL contains {patch_field}. Connect {node_display_name} "
                "directly after Load Diffusion Model. Patched-model export is not supported."
            )

    model_options = getattr(model, "model_options", None)
    if model_options is not None:
        allowed = (
            isinstance(model_options, Mapping)
            and not (set(model_options) - {"transformer_options"})
            and isinstance(model_options.get("transformer_options", {}), Mapping)
            and not model_options.get("transformer_options", {})
        )
        if not allowed:
            raise QuantizationNodeError(
                f"The input MODEL contains model_options. Connect {node_display_name} "
                "directly after Load Diffusion Model. Patched-model export is not supported."
            )

    raw_state_dict = model_state_dict(filter_prefix=DIFFUSION_PREFIX)
    if not raw_state_dict:
        raise QuantizationNodeError(
            "The MODEL has no diffusion_model state dict entries. Use the MODEL output from Load Diffusion Model."
        )

    normalized: dict[str, Any] = {}
    for key, tensor in raw_state_dict.items():
        if not isinstance(key, str) or not key.startswith(DIFFUSION_PREFIX):
            continue
        internal_key = key[len(DIFFUSION_PREFIX) :]
        normalized_key = normalize_key(internal_key)
        if normalized_key in normalized:
            raise QuantizationNodeError(f"State dict key collision after prefix normalization: {normalized_key}")
        normalized[normalized_key] = tensor

    if not normalized:
        raise QuantizationNodeError("Prefix normalization produced an empty diffusion model state dict.")
    if any(key.endswith(".comfy_quant") for key in normalized) or any(
        key.endswith(".weight") and _is_int8_tensor(tensor) for key, tensor in normalized.items()
    ):
        raise QuantizationNodeError(
            "The input MODEL is already quantized. Load the original floating-point model instead."
        )
    return normalized


def extract_anima_state_dict(model: Any) -> dict[str, Any]:
    """Extract the unpatched diffusion model state dict in the Anima ``net.*`` namespace."""

    state_dict = _extract_diffusion_state_dict(
        model,
        node_display_name="Anima INT8 ConvRot Save",
        normalize_key=lambda key: key if key.startswith("net.") else f"net.{key}",
    )
    try:
        validate_anima_state_dict(state_dict, require_selected_tensors=False)
    except AnimaContractError as exc:
        raise QuantizationNodeError(str(exc)) from exc
    return state_dict


def extract_krea2_state_dict(model: Any) -> dict[str, Any]:
    """Extract an unpatched diffusion model state dict in Krea2's native namespace."""

    state_dict = _extract_diffusion_state_dict(
        model,
        node_display_name="Krea2 INT8 ConvRot Save",
        normalize_key=lambda key: key,
    )
    try:
        validate_krea2_state_dict(state_dict, require_selected_tensors=False)
    except Krea2ContractError as exc:
        raise QuantizationNodeError(str(exc)) from exc
    return state_dict


def _validate_minimax_h3_effective_config(model: Any) -> None:
    model_config = getattr(getattr(model, "model", None), "model_config", None)
    unet_config = getattr(model_config, "unet_config", None)
    if not isinstance(unet_config, Mapping):
        raise QuantizationNodeError(
            "MiniMax H3 effective config is unavailable. Use the MODEL output directly from "
            "the current stock ComfyUI Load Diffusion Model node."
        )

    mismatches = {
        key: (expected, unet_config.get(key, "<missing>"))
        for key, expected in _MINIMAX_H3_REFERENCE_CONFIG.items()
        if unet_config.get(key, "<missing>") != expected
    }
    unexpected = sorted(
        set(unet_config) - set(_MINIMAX_H3_REFERENCE_CONFIG) - _MINIMAX_H3_RUNTIME_CONFIG_KEYS
    )
    source_dtype = unet_config.get("dtype")
    invalid_dtype = str(source_dtype) not in _MINIMAX_H3_SOURCE_DTYPES
    if mismatches or unexpected or invalid_dtype:
        details = [
            f"{key}: expected {expected!r}, got {actual!r}"
            for key, (expected, actual) in sorted(mismatches.items())
        ]
        if unexpected:
            details.append(f"unexpected transformer config keys: {', '.join(unexpected)}")
        if invalid_dtype:
            details.append(
                "dtype: expected torch.bfloat16 or torch.float32, "
                f"got {source_dtype!r}"
            )
        raise QuantizationNodeError(
            "MiniMax H3 effective config does not match the official curve-form reference: "
            + "; ".join(details)
        )


def extract_minimax_h3_state_dict(model: Any) -> dict[str, Any]:
    """Extract the exact official MiniMax H3 curve-form diffusion namespace."""

    state_dict = _extract_diffusion_state_dict(
        model,
        node_display_name="MiniMax H3 INT8 ConvRot Save",
        normalize_key=lambda key: key,
    )
    try:
        validate_minimax_h3_state_dict(state_dict, require_selected_tensors=False)
    except MiniMaxH3ContractError as exc:
        raise QuantizationNodeError(str(exc)) from exc
    _validate_minimax_h3_effective_config(model)
    return state_dict


def estimate_state_dict_bytes(state_dict: Mapping[str, Any]) -> int:
    total = 0
    for tensor in state_dict.values():
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if callable(numel) and callable(element_size):
            total += int(numel()) * int(element_size())
    return total


def estimate_minimax_h3_memory(
    state_dict: Mapping[str, Any],
) -> MiniMaxH3MemoryEstimate:
    """Estimate H3 streaming output size and bounded quantization workspace."""

    selected_names = set(get_minimax_h3_tensor_names())
    source_bytes = 0
    output_file_bytes = 0
    workspace_bytes = 0
    for name, tensor in state_dict.items():
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if not callable(numel) or not callable(element_size):
            continue
        tensor_numel = int(numel())
        tensor_bytes = tensor_numel * int(element_size())
        source_bytes += tensor_bytes
        if name not in selected_names:
            output_file_bytes += tensor_bytes
            continue
        shape = getattr(tensor, "shape", ())
        rows = int(shape[0]) if shape else 0
        output_file_bytes += (
            tensor_numel + rows * 4 + MINIMAX_H3_QUANT_MARKER_BYTES
        )
        workspace_bytes = max(
            workspace_bytes,
            min(tensor_bytes, DEFAULT_STREAMING_CHUNK_BYTES)
            * MINIMAX_H3_WORKSPACE_COPIES,
        )

    required_additional_bytes = math.ceil(
        workspace_bytes * MINIMAX_H3_MEMORY_HEADROOM_RATIO
    )
    estimated_peak_bytes = source_bytes + required_additional_bytes
    return MiniMaxH3MemoryEstimate(
        source_bytes=source_bytes,
        output_file_bytes=output_file_bytes,
        workspace_bytes=workspace_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
        required_additional_bytes=required_additional_bytes,
    )


def _available_memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, ValueError):
            return None
        return None

    try:
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return available_pages * page_size


def _preflight_minimax_h3_memory(
    state_dict: Mapping[str, Any],
) -> tuple[MiniMaxH3MemoryEstimate, int | None]:
    estimate = estimate_minimax_h3_memory(state_dict)
    available_bytes = _available_memory_bytes()
    if (
        available_bytes is not None
        and available_bytes < estimate.required_additional_bytes
    ):
        raise QuantizationNodeError(
            "Insufficient available system memory for MiniMax H3 export: "
            f"required_additional={estimate.required_additional_bytes}, "
            f"available={available_bytes}, "
            f"estimated_peak={estimate.estimated_peak_bytes}."
        )
    return estimate, available_bytes


def _ensure_output_capacity(
    output_directory: Path,
    state_dict: Mapping[str, Any],
    *,
    required_output_bytes: int | None = None,
) -> None:
    required_bytes = (
        estimate_state_dict_bytes(state_dict)
        if required_output_bytes is None
        else int(required_output_bytes)
    ) + DISK_HEADROOM_BYTES
    free_bytes = int(shutil.disk_usage(output_directory).free)
    if free_bytes < required_bytes:
        raise QuantizationNodeError(
            "Insufficient free disk space for the temporary checkpoint: "
            f"required_at_least={required_bytes}, available={free_bytes}, "
            f"directory={output_directory}."
        )


def _validated_relative_prefix(filename_prefix: str) -> Path:
    value = str(filename_prefix).strip()
    if not value:
        raise QuantizationNodeError("filename_prefix must not be empty.")
    value = value.replace("\\", "/")
    if value.casefold().endswith(".safetensors"):
        value = value[: -len(".safetensors")]
    relative = Path(value)
    if not value or relative.is_absolute() or relative.anchor or ".." in relative.parts:
        raise QuantizationNodeError(
            "filename_prefix must be a relative path inside output/diffusion_models."
        )
    if any(char in _INVALID_FILENAME_CHARS or ord(char) < 32 for char in value):
        raise QuantizationNodeError("filename_prefix contains characters that are invalid on Windows.")
    if any(part in ("", ".") for part in relative.parts):
        raise QuantizationNodeError("filename_prefix contains an invalid empty path component.")
    if any(
        part.endswith((" ", ".")) or part.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
        for part in relative.parts
    ):
        raise QuantizationNodeError("filename_prefix contains a Windows-reserved path component.")
    return relative


def resolve_output_paths(
    output_models_root: str | os.PathLike[str], filename_prefix: str
) -> OutputPaths:
    output_root = Path(output_models_root).expanduser().resolve(strict=False)
    if output_root.name.casefold() != "diffusion_models":
        raise QuantizationNodeError(
            "The output model root must be ComfyUI output/diffusion_models."
        )
    relative = _validated_relative_prefix(filename_prefix)
    checkpoint_base = output_root / relative
    checkpoint = Path(f"{checkpoint_base}.safetensors").resolve(strict=False)
    try:
        checkpoint.relative_to(output_root)
    except ValueError as exc:
        raise QuantizationNodeError(
            f"Output path escapes the allowed directory: {output_root}"
        ) from exc
    report = checkpoint.with_suffix(".export_report.json")
    return OutputPaths(checkpoint=checkpoint, report=report)


def _write_json_temporary(path: Path, payload: Mapping[str, Any]) -> Path:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _report_to_dict(report: Any) -> dict[str, Any]:
    if isinstance(report, Mapping):
        return dict(report)
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise QuantizationNodeError("The internal quantizer returned an unsupported export report object.")


def _publish_without_overwrite(source: Path, destination: Path) -> None:
    """Publish without a check-then-replace race on Windows and POSIX."""

    try:
        if os.name == "nt":
            os.rename(source, destination)
        else:
            os.link(source, destination)
    except FileExistsError as exc:
        raise QuantizationNodeError(
            f"Output appeared while overwrite was disabled: {destination}"
        ) from exc
    except OSError as exc:
        if os.name != "nt":
            raise QuantizationNodeError(
                "Atomic no-overwrite publication is not supported by the output filesystem: "
                f"{destination}"
            ) from exc
        raise


def _export_int8_convrot(
    *,
    state_dict: Mapping[str, Any],
    paths: OutputPaths,
    family: str,
    project: str,
    model_display_name: str,
    expected_count: int,
    expected_copied_count: int | None = None,
    validate_source_dtype: bool = False,
    required_output_bytes: int | None = None,
    additional_report_fields: Mapping[str, Any] | None = None,
    default_exporter: Callable[..., Any],
    device: str,
    overwrite: bool,
    write_report: bool,
    hash_output: bool,
    quantization_preset: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
    exporter: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Export with a model-specific quantizer and publish both artifacts atomically."""
    if paths.checkpoint.exists() and not overwrite:
        raise QuantizationNodeError(
            f"Output checkpoint already exists and overwrite is disabled: {paths.checkpoint}"
        )
    if paths.report.exists() and not overwrite:
        raise QuantizationNodeError(
            f"Output report already exists and overwrite is disabled: {paths.report}"
        )
    try:
        paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        _ensure_output_capacity(
            paths.checkpoint.parent,
            state_dict,
            required_output_bytes=required_output_bytes,
        )
    except QuantizationNodeError:
        raise
    except OSError as exc:
        raise QuantizationNodeError(
            f"Unable to prepare output directory {paths.checkpoint.parent}: {exc}"
        ) from exc
    temporary = paths.checkpoint.with_name(
        f".{paths.checkpoint.stem}.{uuid.uuid4().hex}.tmp.safetensors"
    )
    temporary_report: Path | None = None
    checkpoint_backup: Path | None = None
    report_backup: Path | None = None
    export = exporter or default_exporter
    try:
        try:
            report_object = export(
                state_dict=state_dict,
                output_checkpoint=temporary,
                family=family,
                convrot=True,
                convrot_groupsize=256,
                device=device,
                strict=True,
                validate_source_dtype=validate_source_dtype,
                quantization_preset=quantization_preset,
                require_all_rotated=True,
                hash_output=hash_output,
                metadata={
                    "model_family": family,
                    "project": project,
                    "source": "comfyui_model_state_dict",
                },
                progress=progress,
            )
        except (
            AnimaContractError,
            Krea2ContractError,
            MiniMaxH3ContractError,
            QuantizationExportError,
        ) as exc:
            raise QuantizationNodeError(str(exc)) from exc
        report = _report_to_dict(report_object)
        if additional_report_fields:
            report.update(additional_report_fields)
        quantized_count = int(report.get("quantized_tensor_count", -1))
        rotated_count = int(report.get("rotated_tensor_count", -1))
        copied_count = int(report.get("copied_tensor_count", -1))
        copied_mismatch = (
            expected_copied_count is not None
            and copied_count != expected_copied_count
        )
        if (
            quantized_count != expected_count
            or rotated_count != quantized_count
            or copied_mismatch
        ):
            raise QuantizationNodeError(
                "Unexpected quantization result: "
                f"quantized={quantized_count}, rotated={rotated_count}, "
                f"copied={copied_count}, expected_quantized={expected_count}, "
                f"expected_copied={expected_copied_count}, "
                f"preset={quantization_preset}."
            )
        if not temporary.is_file():
            raise QuantizationNodeError("The internal quantizer completed without writing the checkpoint.")
        report["output_checkpoint"] = str(paths.checkpoint)
        report["quantization_preset"] = quantization_preset
        for written in report.get("written_files", []):
            if isinstance(written, dict) and written.get("kind") == "int8_tensorwise_inference_checkpoint":
                written["path"] = str(paths.checkpoint)
        report_path = ""
        if write_report:
            temporary_report = _write_json_temporary(paths.report, report)
            report_path = str(paths.report)

        published_checkpoint = False
        published_report = False
        try:
            if overwrite and paths.checkpoint.exists():
                checkpoint_backup = paths.checkpoint.with_name(
                    f".{paths.checkpoint.name}.{uuid.uuid4().hex}.bak"
                )
                os.replace(paths.checkpoint, checkpoint_backup)
            if overwrite and paths.report.exists():
                report_backup = paths.report.with_name(f".{paths.report.name}.{uuid.uuid4().hex}.bak")
                os.replace(paths.report, report_backup)

            if overwrite:
                os.replace(temporary, paths.checkpoint)
            else:
                _publish_without_overwrite(temporary, paths.checkpoint)
            published_checkpoint = True
            if temporary_report is not None:
                if overwrite:
                    os.replace(temporary_report, paths.report)
                else:
                    _publish_without_overwrite(temporary_report, paths.report)
                published_report = True
        except BaseException:
            if published_report:
                paths.report.unlink(missing_ok=True)
            if published_checkpoint:
                paths.checkpoint.unlink(missing_ok=True)
            if checkpoint_backup is not None and checkpoint_backup.exists():
                os.replace(checkpoint_backup, paths.checkpoint)
            if report_backup is not None and report_backup.exists():
                os.replace(report_backup, paths.report)
            raise
        else:
            if checkpoint_backup is not None:
                checkpoint_backup.unlink(missing_ok=True)
            if report_backup is not None:
                report_backup.unlink(missing_ok=True)
        return report, report_path
    except QuantizationNodeError:
        raise
    except Exception as exc:
        raise QuantizationNodeError(
            f"{model_display_name} INT8 export failed while writing or publishing "
            f"{paths.checkpoint}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)


def export_anima_int8_convrot(
    *,
    state_dict: Mapping[str, Any],
    paths: OutputPaths,
    device: str,
    overwrite: bool,
    write_report: bool,
    hash_output: bool,
    quantization_preset: str = DEFAULT_QUANTIZATION_PRESET,
    progress: Callable[[dict[str, Any]], None] | None = None,
    exporter: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], str]:
    return _export_int8_convrot(
        state_dict=state_dict,
        paths=paths,
        family="anima",
        project="anima-int8-convrot",
        model_display_name="Anima",
        expected_count=expected_quantized_tensors(quantization_preset),
        default_exporter=export_anima_int8_convrot_from_state_dict,
        device=device,
        overwrite=overwrite,
        write_report=write_report,
        hash_output=hash_output,
        quantization_preset=quantization_preset,
        progress=progress,
        exporter=exporter,
    )


def export_krea2_int8_convrot(
    *,
    state_dict: Mapping[str, Any],
    paths: OutputPaths,
    device: str,
    overwrite: bool,
    write_report: bool,
    hash_output: bool,
    quantization_preset: str = DEFAULT_KREA2_PRESET,
    progress: Callable[[dict[str, Any]], None] | None = None,
    exporter: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], str]:
    return _export_int8_convrot(
        state_dict=state_dict,
        paths=paths,
        family="krea2",
        project="krea2-int8-convrot",
        model_display_name="Krea2",
        expected_count=expected_krea2_quantized_tensors(quantization_preset),
        default_exporter=export_krea2_int8_convrot_from_state_dict,
        device=device,
        overwrite=overwrite,
        write_report=write_report,
        hash_output=hash_output,
        quantization_preset=quantization_preset,
        progress=progress,
        exporter=exporter,
    )


def export_minimax_h3_int8_convrot(
    *,
    state_dict: Mapping[str, Any],
    paths: OutputPaths,
    device: str,
    overwrite: bool,
    write_report: bool,
    hash_output: bool,
    quantization_preset: str = DEFAULT_MINIMAX_H3_PRESET,
    progress: Callable[[dict[str, Any]], None] | None = None,
    exporter: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], str]:
    memory_estimate, available_memory_bytes = _preflight_minimax_h3_memory(
        state_dict
    )
    memory_report = {
        "estimated_peak_memory_bytes": memory_estimate.estimated_peak_bytes,
        "required_additional_memory_bytes": memory_estimate.required_additional_bytes,
        "available_memory_bytes_at_preflight": available_memory_bytes,
        "estimated_output_file_bytes": memory_estimate.output_file_bytes,
        "streaming_chunk_bytes": DEFAULT_STREAMING_CHUNK_BYTES,
    }
    return _export_int8_convrot(
        state_dict=state_dict,
        paths=paths,
        family="minimax_h3",
        project="minimax-h3-int8-convrot",
        model_display_name="MiniMax H3",
        expected_count=expected_minimax_h3_quantized_tensors(quantization_preset),
        expected_copied_count=EXPECTED_KEEP_TENSOR_COUNT,
        validate_source_dtype=True,
        required_output_bytes=memory_estimate.output_file_bytes,
        additional_report_fields=memory_report,
        default_exporter=export_minimax_h3_int8_convrot_from_state_dict,
        device=device,
        overwrite=overwrite,
        write_report=write_report,
        hash_output=hash_output,
        quantization_preset=quantization_preset,
        progress=progress,
        exporter=exporter,
    )
