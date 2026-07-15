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
    assert len(nodes) == 2
    nodes_by_id = {node.define_schema().node_id: node for node in nodes}
    assert set(nodes_by_id) == {
        "ComfyQuantsAnimaInt8ConvRotSave",
        "ComfyQuantsKrea2Int8ConvRotSave",
    }
    anima_node = nodes_by_id["ComfyQuantsAnimaInt8ConvRotSave"]
    krea2_node = nodes_by_id["ComfyQuantsKrea2Int8ConvRotSave"]
    node_module = sys.modules[anima_node.__module__]
    output_models_root = str(
        (Path(node_module.folder_paths.get_output_directory()) / "diffusion_models").resolve(
            strict=False
        )
    )
    assert output_models_root in node_module.folder_paths.get_folder_paths("diffusion_models")
    schema = anima_node.define_schema()
    assert schema.node_id == "ComfyQuantsAnimaInt8ConvRotSave"
    assert schema.is_output_node is True
    assert schema.not_idempotent is True
    filename_prefix = next(item for item in schema.inputs if item.id == "filename_prefix")
    assert "output/diffusion_models" in filename_prefix.tooltip
    preset = next(item for item in schema.inputs if item.id == "quantization_preset")
    assert preset.options == ["quality_keep", "public_examples"]
    assert preset.default == "quality_keep"
    assert [item.get_io_type() for item in schema.outputs] == ["STRING", "STRING", "STRING"]
    assert "exported INT8" in schema.outputs[0].tooltip
    assert "empty string" in schema.outputs[1].tooltip
    assert "JSON summary" in schema.outputs[2].tooltip
    schema.validate()
    info = schema.get_v1_info(anima_node)
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

    krea2_schema = krea2_node.define_schema()
    assert krea2_schema.node_id == "ComfyQuantsKrea2Int8ConvRotSave"
    assert krea2_schema.display_name == "Krea2 INT8 ConvRot Save"
    krea2_preset = next(item for item in krea2_schema.inputs if item.id == "quantization_preset")
    assert krea2_preset.options == ["quality_keep"]
    assert krea2_preset.default == "quality_keep"
    krea2_schema.validate()
    krea2_info = krea2_schema.get_v1_info(krea2_node)
    assert krea2_info.category == "Comfy Quants/quantization"
    assert krea2_info.output_node is True
    assert krea2_info.search_aliases == ["krea2", "krea 2", "int8", "convrot", "comfy quants", "quantize"]


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

    output_directory = tmp_path / "output"
    output_root = output_directory / "diffusion_models"
    output_paths = node_module.resolve_output_paths(output_root, "anima")
    registered_paths = []
    monkeypatch.setattr(node_module, "ProgressBar", FakeProgressBar)
    monkeypatch.setattr(
        node_module.folder_paths,
        "get_output_directory",
        lambda: str(output_directory),
    )
    monkeypatch.setattr(
        node_module.folder_paths,
        "add_model_folder_path",
        lambda category, path: registered_paths.append((category, path)),
    )
    monkeypatch.setattr(node_module, "extract_anima_state_dict", lambda _model: {"x": object()})
    monkeypatch.setattr(node_module, "estimate_state_dict_bytes", lambda _state: 123)
    monkeypatch.setattr(node_module, "expected_quantized_tensors", lambda _preset: 2)

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
    assert registered_paths == [("diffusion_models", str(output_root))]


def test_krea2_execute_uses_krea2_contract_and_default_prefix(tmp_path, monkeypatch):
    if importlib.util.find_spec("folder_paths") is None:
        pytest.skip("ComfyUI checkout is required for the Backend V3 integration test")
    package = load_package()
    extension = asyncio.run(package.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    node_class = next(
        node for node in nodes
        if node.define_schema().node_id == "ComfyQuantsKrea2Int8ConvRotSave"
    )
    node_module = sys.modules[node_class.__module__]
    node_class.hidden = types.SimpleNamespace(unique_id="krea2-node")
    output_directory = tmp_path / "output"
    output_root = output_directory / "diffusion_models"
    output_paths = node_module.resolve_output_paths(output_root, "krea2_int8_convrot")
    captured = {}

    class FakeProgressBar:
        def __init__(self, total, node_id=None):
            captured["progress"] = (total, node_id)

        def update_absolute(self, current, total):
            captured["last_progress"] = (current, total)

    monkeypatch.setattr(node_module, "ProgressBar", FakeProgressBar)
    monkeypatch.setattr(
        node_module.folder_paths,
        "get_output_directory",
        lambda: str(output_directory),
    )
    monkeypatch.setattr(node_module.folder_paths, "add_model_folder_path", lambda *_args: None)
    monkeypatch.setattr(node_module, "extract_krea2_state_dict", lambda _model: {"x": object()})
    monkeypatch.setattr(node_module, "estimate_state_dict_bytes", lambda _state: 123)
    monkeypatch.setattr(node_module, "expected_krea2_quantized_tensors", lambda _preset: 224)

    def fake_export(**kwargs):
        captured["export"] = kwargs
        return (
            {
                "status": "model_written",
                "quantized_tensor_count": 224,
                "rotated_tensor_count": 224,
                "copied_tensor_count": 10,
                "output_bytes": 456,
                "execution_device": "cpu",
            },
            "",
        )

    monkeypatch.setattr(node_module, "export_krea2_int8_convrot", fake_export)
    result = node_class.execute(
        model=object(),
        filename_prefix="krea2_int8_convrot",
        device="cpu",
        overwrite=False,
        write_report=False,
        hash_output=False,
    )

    checkpoint, report, summary_text = result.result
    assert checkpoint == str(output_paths.checkpoint)
    assert report == ""
    assert json.loads(summary_text)["quantized_tensor_count"] == 224
    assert captured["progress"] == (224, "krea2-node")
    assert captured["last_progress"] == (224, 224)
    assert captured["export"]["quantization_preset"] == "quality_keep"


def test_stock_comfyui_detects_the_exported_krea2_namespace_and_quant_markers():
    comfy_spec = importlib.util.find_spec("comfy")
    if comfy_spec is None or not comfy_spec.submodule_search_locations:
        pytest.skip("ComfyUI checkout is required for stock loader contract validation")

    import ast
    import logging
    import math

    comfy_root = Path(next(iter(comfy_spec.submodule_search_locations)))

    def load_functions(path, names, namespace):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        selected = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
        ]
        assert {node.name for node in selected} == set(names)
        module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
        return namespace

    detection = load_functions(
        comfy_root / "model_detection.py",
        {
            "any_suffix_in",
            "calculate_transformer_depth",
            "count_blocks",
            "detect_unet_config",
        },
        {"math": math},
    )
    quantization = load_functions(
        comfy_root / "utils.py",
        {"detect_layer_quantization"},
        {"logging": logging},
    )

    class ShapeOnlyTensor:
        def __init__(self, shape):
            self.shape = tuple(shape)

    state = {
        "first.weight": ShapeOnlyTensor((6144, 64)),
        "txtfusion.projector.weight": ShapeOnlyTensor((1, 12)),
        "txtfusion.layerwise_blocks.0.prenorm.scale": ShapeOnlyTensor((2560,)),
        "blocks.0.attn.wq.weight": ShapeOnlyTensor((6144, 6144)),
        "blocks.0.attn.wk.weight": ShapeOnlyTensor((1536, 6144)),
        "blocks.0.attn.wq.comfy_quant": ShapeOnlyTensor((80,)),
    }
    for block in range(1, 28):
        state[f"blocks.{block}.attn.wq.weight"] = ShapeOnlyTensor((6144, 6144))

    config = detection["detect_unet_config"](state, "")
    assert config == {
        "image_model": "krea2",
        "features": 6144,
        "channels": 16,
        "patch": 2,
        "layers": 28,
        "heads": 48,
        "kvheads": 12,
        "txtlayers": 12,
        "txtdim": 2560,
    }
    assert quantization["detect_layer_quantization"](state, "") == {"mixed_ops": True}
