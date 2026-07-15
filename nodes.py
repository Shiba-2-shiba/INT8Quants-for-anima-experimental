from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folder_paths
from comfy.utils import ProgressBar
from comfy_api.v0_0_2 import ComfyExtension, io

from .service import (
    DEFAULT_KREA2_PRESET,
    DEFAULT_QUANTIZATION_PRESET,
    KREA2_PRESETS,
    QUANTIZATION_PRESETS,
    OutputPaths,
    estimate_state_dict_bytes,
    expected_krea2_quantized_tensors,
    expected_quantized_tensors,
    export_anima_int8_convrot,
    export_krea2_int8_convrot,
    extract_anima_state_dict,
    extract_krea2_state_dict,
    resolve_output_paths,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _NodeProfile:
    node_id: str
    display_name: str
    model_name: str
    model_tooltip: str
    description: str
    filename_prefix: str
    presets: tuple[str, ...]
    default_preset: str
    preset_tooltip: str
    search_aliases: tuple[str, ...]


_ANIMA_PROFILE = _NodeProfile(
    node_id="ComfyQuantsAnimaInt8ConvRotSave",
    display_name="Anima INT8 ConvRot Save",
    model_name="Anima",
    model_tooltip="Unpatched Anima 2B MODEL from Load Diffusion Model.",
    description=(
        "Quantize an unpatched Anima 2B MODEL to stock ComfyUI INT8 ConvRot "
        "and save it as safetensors."
    ),
    filename_prefix="anima_int8_convrot",
    presets=tuple(QUANTIZATION_PRESETS),
    default_preset=DEFAULT_QUANTIZATION_PRESET,
    preset_tooltip=(
        "quality_keep: keep block 0 and block 1 AdaLN in source precision (426 INT8). "
        "public_examples: match the inspected public checkpoints (448 INT8)."
    ),
    search_aliases=("anima", "int8", "convrot", "comfy quants", "quantize"),
)

_KREA2_PROFILE = _NodeProfile(
    node_id="ComfyQuantsKrea2Int8ConvRotSave",
    display_name="Krea2 INT8 ConvRot Save",
    model_name="Krea2",
    model_tooltip="Unpatched Krea2 Raw or Turbo MODEL from Load Diffusion Model.",
    description=(
        "Quantize an unpatched Krea2 Raw or Turbo MODEL to stock ComfyUI INT8 ConvRot "
        "and save it as safetensors."
    ),
    filename_prefix="krea2_int8_convrot",
    presets=tuple(KREA2_PRESETS),
    default_preset=DEFAULT_KREA2_PRESET,
    preset_tooltip=(
        "quality_keep: quantize the 224 transformer-block linear weights while keeping "
        "embedders, text fusion, modulation, normalization, and final layers in source precision."
    ),
    search_aliases=("krea2", "krea 2", "int8", "convrot", "comfy quants", "quantize"),
)


def _register_output_models_root() -> Path:
    output_root = (
        Path(folder_paths.get_output_directory()) / "diffusion_models"
    ).resolve(strict=False)
    folder_paths.add_model_folder_path("diffusion_models", str(output_root))
    return output_root


def _define_schema(profile: _NodeProfile) -> io.Schema:
    return io.Schema(
        node_id=profile.node_id,
        display_name=profile.display_name,
        category="Comfy Quants/quantization",
        description=profile.description,
        search_aliases=list(profile.search_aliases),
        is_output_node=True,
        not_idempotent=True,
        inputs=[
            io.Model.Input("model", tooltip=profile.model_tooltip),
            io.String.Input(
                "filename_prefix",
                default=profile.filename_prefix,
                tooltip="Path below output/diffusion_models, without an extension.",
            ),
            io.Combo.Input(
                "device",
                options=["cpu", "auto", "cuda"],
                default="cpu",
                advanced=True,
                tooltip="Torch device used one tensor at a time for ConvRot and quantization.",
            ),
            io.Boolean.Input(
                "overwrite",
                default=False,
                advanced=True,
                tooltip="Replace an existing checkpoint with the same name.",
            ),
            io.Boolean.Input(
                "write_report",
                default=True,
                advanced=True,
                tooltip="Write a JSON export report next to the checkpoint.",
            ),
            io.Boolean.Input(
                "hash_output",
                default=False,
                advanced=True,
                tooltip="Compute SHA-256 after writing the checkpoint.",
            ),
            io.Combo.Input(
                "quantization_preset",
                options=list(profile.presets),
                default=profile.default_preset,
                tooltip=profile.preset_tooltip,
            ),
        ],
        outputs=[
            io.String.Output(
                "checkpoint_path",
                display_name="checkpoint path",
                tooltip="Absolute path of the exported INT8 .safetensors checkpoint.",
            ),
            io.String.Output(
                "report_path",
                display_name="report path",
                tooltip=(
                    "Absolute path of the JSON export report, or an empty string "
                    "when write_report is disabled."
                ),
            ),
            io.String.Output(
                "summary",
                display_name="summary",
                tooltip="Compact JSON summary of the completed export.",
            ),
        ],
        hidden=[io.Hidden.unique_id],
    )


def _execute_export(
    node_class: type[io.ComfyNode],
    profile: _NodeProfile,
    *,
    model: Any,
    filename_prefix: str,
    quantization_preset: str,
    device: str,
    overwrite: bool,
    write_report: bool,
    hash_output: bool,
    extract_state_dict: Callable[[Any], dict[str, Any]],
    expected_count_for_preset: Callable[[str], int],
    export_checkpoint: Callable[..., tuple[dict[str, Any], str]],
) -> io.NodeOutput:
    state_dict = extract_state_dict(model)
    output_root = _register_output_models_root()
    paths: OutputPaths = resolve_output_paths(output_root, filename_prefix)
    source_bytes = estimate_state_dict_bytes(state_dict)
    LOGGER.info(
        "Starting %s INT8 ConvRot export: tensors=%d, source_bytes=%d, device=%s, output=%s",
        profile.model_name,
        len(state_dict),
        source_bytes,
        device,
        paths.checkpoint,
    )

    expected_count = expected_count_for_preset(quantization_preset)
    progress_bar = ProgressBar(expected_count, node_id=node_class.hidden.unique_id)

    def progress(event: Mapping[str, Any]) -> None:
        if event.get("stage") == "quantize_tensor":
            current = int(event.get("quantized_tensor_count", 0))
            total = int(event.get("selected_tensor_count", expected_count))
            progress_bar.update_absolute(current, total)

    report, report_path = export_checkpoint(
        state_dict=state_dict,
        paths=paths,
        device=device,
        overwrite=overwrite,
        write_report=write_report,
        hash_output=hash_output,
        quantization_preset=quantization_preset,
        progress=progress,
    )
    progress_bar.update_absolute(expected_count, expected_count)
    summary_data = {
        "status": report.get("status"),
        "output_checkpoint": str(paths.checkpoint),
        "quantization_preset": quantization_preset,
        "quantized_tensor_count": report.get("quantized_tensor_count"),
        "rotated_tensor_count": report.get("rotated_tensor_count"),
        "copied_tensor_count": report.get("copied_tensor_count"),
        "output_bytes": report.get("output_bytes"),
        "execution_device": report.get("execution_device"),
        "output_hash": report.get("output_hash", ""),
    }
    summary = json.dumps(summary_data, ensure_ascii=False, sort_keys=True)
    LOGGER.info("Completed %s INT8 ConvRot export: %s", profile.model_name, summary)
    return io.NodeOutput(str(paths.checkpoint), report_path, summary)


class ComfyQuantsAnimaInt8ConvRotSave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _define_schema(_ANIMA_PROFILE)

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        filename_prefix: str,
        quantization_preset: str = DEFAULT_QUANTIZATION_PRESET,
        device: str,
        overwrite: bool,
        write_report: bool,
        hash_output: bool,
    ) -> io.NodeOutput:
        return _execute_export(
            cls,
            _ANIMA_PROFILE,
            model=model,
            filename_prefix=filename_prefix,
            quantization_preset=quantization_preset,
            device=device,
            overwrite=overwrite,
            write_report=write_report,
            hash_output=hash_output,
            extract_state_dict=extract_anima_state_dict,
            expected_count_for_preset=expected_quantized_tensors,
            export_checkpoint=export_anima_int8_convrot,
        )


class ComfyQuantsKrea2Int8ConvRotSave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _define_schema(_KREA2_PROFILE)

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        filename_prefix: str,
        quantization_preset: str = DEFAULT_KREA2_PRESET,
        device: str,
        overwrite: bool,
        write_report: bool,
        hash_output: bool,
    ) -> io.NodeOutput:
        return _execute_export(
            cls,
            _KREA2_PROFILE,
            model=model,
            filename_prefix=filename_prefix,
            quantization_preset=quantization_preset,
            device=device,
            overwrite=overwrite,
            write_report=write_report,
            hash_output=hash_output,
            extract_state_dict=extract_krea2_state_dict,
            expected_count_for_preset=expected_krea2_quantized_tensors,
            export_checkpoint=export_krea2_int8_convrot,
        )


class ComfyQuantsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ComfyQuantsAnimaInt8ConvRotSave,
            ComfyQuantsKrea2Int8ConvRotSave,
        ]


async def comfy_entrypoint() -> ComfyQuantsExtension:
    _register_output_models_root()
    return ComfyQuantsExtension()
