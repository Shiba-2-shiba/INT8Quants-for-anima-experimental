from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import folder_paths
from comfy.utils import ProgressBar
from comfy_api.v0_0_2 import ComfyExtension, io

from .service import (
    DEFAULT_QUANTIZATION_PRESET,
    QUANTIZATION_PRESETS,
    estimate_state_dict_bytes,
    expected_quantized_tensors,
    export_anima_int8_convrot,
    extract_anima_state_dict,
    resolve_output_paths,
)


LOGGER = logging.getLogger(__name__)


def _register_output_models_root() -> Path:
    output_root = (
        Path(folder_paths.get_output_directory()) / "diffusion_models"
    ).resolve(strict=False)
    folder_paths.add_model_folder_path("diffusion_models", str(output_root))
    return output_root


class ComfyQuantsAnimaInt8ConvRotSave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ComfyQuantsAnimaInt8ConvRotSave",
            display_name="Anima INT8 ConvRot Save",
            category="Comfy Quants/quantization",
            description=(
                "Quantize an unpatched Anima 2B MODEL to stock ComfyUI INT8 ConvRot "
                "and save it as safetensors."
            ),
            search_aliases=["anima", "int8", "convrot", "comfy quants", "quantize"],
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="Unpatched Anima 2B MODEL from Load Diffusion Model.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="anima_int8_convrot",
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
                    options=list(QUANTIZATION_PRESETS),
                    default=DEFAULT_QUANTIZATION_PRESET,
                    tooltip=(
                        "quality_keep: keep block 0 and block 1 AdaLN in source precision (426 INT8). "
                        "public_examples: match the inspected public checkpoints (448 INT8)."
                    ),
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
        state_dict = extract_anima_state_dict(model)
        output_root = _register_output_models_root()
        paths = resolve_output_paths(output_root, filename_prefix)
        source_bytes = estimate_state_dict_bytes(state_dict)
        LOGGER.info(
            "Starting Anima INT8 ConvRot export: tensors=%d, source_bytes=%d, device=%s, output=%s",
            len(state_dict),
            source_bytes,
            device,
            paths.checkpoint,
        )

        expected_count = expected_quantized_tensors(quantization_preset)
        progress_bar = ProgressBar(expected_count, node_id=cls.hidden.unique_id)

        def progress(event: dict[str, Any]) -> None:
            if event.get("stage") == "quantize_tensor":
                current = int(event.get("quantized_tensor_count", 0))
                total = int(event.get("selected_tensor_count", expected_count))
                progress_bar.update_absolute(current, total)

        report, report_path = export_anima_int8_convrot(
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
        LOGGER.info("Completed Anima INT8 ConvRot export: %s", summary)
        return io.NodeOutput(str(paths.checkpoint), report_path, summary)


class ComfyQuantsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ComfyQuantsAnimaInt8ConvRotSave]


async def comfy_entrypoint() -> ComfyQuantsExtension:
    _register_output_models_root()
    return ComfyQuantsExtension()
