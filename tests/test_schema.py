from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_comfy_quants_node"


def load_package():
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


def test_package_import_is_lazy_and_host_independent():
    package = load_package()
    assert callable(package.comfy_entrypoint)


def test_v3_schema_and_registration():
    if importlib.util.find_spec("folder_paths") is None:
        pytest.skip("ComfyUI checkout is required for the Backend V3 integration test")
    package = load_package()
    extension = asyncio.run(package.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    assert len(nodes) == 1
    schema = nodes[0].define_schema()
    assert schema.node_id == "ComfyQuantsAnimaInt8ConvRotSave"
    assert schema.is_output_node is True
    assert schema.not_idempotent is True
    preset = next(item for item in schema.inputs if item.id == "quantization_preset")
    assert preset.options == ["quality_keep", "public_examples"]
    assert preset.default == "quality_keep"
    assert [item.get_io_type() for item in schema.outputs] == ["STRING", "STRING", "STRING"]
    assert "exported INT8" in schema.outputs[0].tooltip
    assert "empty string" in schema.outputs[1].tooltip
    assert "JSON summary" in schema.outputs[2].tooltip
    schema.validate()
    info = schema.get_v1_info(nodes[0])
    assert info.name == "ComfyQuantsAnimaInt8ConvRotSave"
    assert info.display_name == "Anima INT8 ConvRot Save"
    assert info.category == "Comfy Quants/quantization"
    assert info.input_order["required"] == [
        "model",
        "filename_prefix",
        "device",
        "overwrite",
        "write_report",
        "hash_output",
        "quantization_preset",
    ]
    assert info.output == ["STRING", "STRING", "STRING"]
    assert info.output_name == ["checkpoint path", "report path", "summary"]
    assert info.output_node is True
    assert info.search_aliases == ["anima", "int8", "convrot", "comfy quants", "quantize"]


def test_execute_reports_progress_and_returns_stable_summary(tmp_path, monkeypatch):
    if importlib.util.find_spec("folder_paths") is None:
        pytest.skip("ComfyUI checkout is required for the Backend V3 integration test")
    package = load_package()
    extension = asyncio.run(package.comfy_entrypoint())
    node_class = asyncio.run(extension.get_node_list())[0]
    node_module = sys.modules[node_class.__module__]
    node_class.hidden = types.SimpleNamespace(unique_id="node-1")
    progress_updates = []

    class FakeProgressBar:
        def __init__(self, total, node_id=None):
            progress_updates.append(("init", total, node_id))

        def update_absolute(self, current, total):
            progress_updates.append(("update", current, total))

    output_paths = node_module.resolve_output_paths(
        [tmp_path / "diffusion_models"], "anima"
    )
    monkeypatch.setattr(node_module, "ProgressBar", FakeProgressBar)
    monkeypatch.setattr(node_module, "extract_anima_state_dict", lambda _model: {"x": object()})
    monkeypatch.setattr(node_module, "estimate_state_dict_bytes", lambda _state: 123)
    monkeypatch.setattr(node_module, "expected_quantized_tensors", lambda _preset: 2)
    monkeypatch.setattr(node_module, "resolve_output_paths", lambda _roots, _prefix: output_paths)

    def fake_export(**kwargs):
        kwargs["progress"](
            {
                "stage": "quantize_tensor",
                "quantized_tensor_count": 1,
                "selected_tensor_count": 2,
            }
        )
        return (
            {
                "status": "model_written",
                "quantized_tensor_count": 2,
                "rotated_tensor_count": 2,
                "copied_tensor_count": 3,
                "output_bytes": 456,
                "execution_device": "cpu",
                "output_hash": "abc",
            },
            str(output_paths.report),
        )

    monkeypatch.setattr(node_module, "export_anima_int8_convrot", fake_export)
    result = node_class.execute(
        model=object(),
        filename_prefix="anima",
        quantization_preset="quality_keep",
        device="cpu",
        overwrite=False,
        write_report=True,
        hash_output=True,
    )

    checkpoint, report, summary_text = result.result
    summary = json.loads(summary_text)
    assert checkpoint == str(output_paths.checkpoint)
    assert report == str(output_paths.report)
    assert summary == {
        "copied_tensor_count": 3,
        "execution_device": "cpu",
        "output_bytes": 456,
        "output_checkpoint": str(output_paths.checkpoint),
        "output_hash": "abc",
        "quantization_preset": "quality_keep",
        "quantized_tensor_count": 2,
        "rotated_tensor_count": 2,
        "status": "model_written",
    }
    assert progress_updates[0][:2] == ("init", 2)
    assert progress_updates[-2:] == [("update", 1, 2), ("update", 2, 2)]
