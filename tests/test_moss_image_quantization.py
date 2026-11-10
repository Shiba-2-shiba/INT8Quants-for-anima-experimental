from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "moss_image_tests"
spec = importlib.util.spec_from_file_location(
    PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE] = package
spec.loader.exec_module(package)
REFERENCE = json.loads((ROOT / "tests/fixtures/moss_image_reference.json").read_text())


def source_state(*, fused=True):
    state = {
        name: SimpleNamespace(shape=tuple(item["shape"]), dtype="torch.bfloat16")
        for name, item in REFERENCE["tensors"].items()
    }
    # ComfyUI drops this serialized buffer and recomputes RoPE at runtime.
    state.pop("rope.bands")
    if not fused:
        for i in range(42):
            state.pop(f"layers.{i}.proj.qkv.weight")
            for component in ("q", "k", "v"):
                state[f"layers.{i}.proj.{component}.weight"] = SimpleNamespace(
                    shape=(3072, 3072), dtype="torch.bfloat16",
                )
    return state


def contract():
    return importlib.import_module(f"{PACKAGE}.quantization.moss_image")


@pytest.mark.parametrize("fused,count", [(True, 168), (False, 252)])
def test_moss_selection_preserves_native_qkv_layout_and_reference_boundary(fused, count):
    specs = contract().validate_moss_image_state_dict(source_state(fused=fused))
    actual = {item.name: list(item.shape) for item in specs}
    expected = {
        name: item["shape"] for name, item in REFERENCE["tensors"].items()
        if item["dtype"] == "I8"
    }
    if not fused:
        for i in range(42):
            expected.pop(f"layers.{i}.proj.qkv.weight")
            for part in ("q", "k", "v"):
                expected[f"layers.{i}.proj.{part}.weight"] = [3072, 3072]
    assert actual == expected
    assert len(specs) == count
    assert all(item.shape[1] % 256 == 0 for item in specs)


@pytest.mark.parametrize("mutation", ["missing_layer", "extra_layer", "missing_keep", "wrong_shape", "mixed_qkv", "already_quantized", "wrong_dtype"])
def test_moss_rejects_incomplete_or_unsupported_models(mutation):
    state = source_state()
    if mutation == "missing_layer":
        state = {k: v for k, v in state.items() if not k.startswith("layers.41.")}
    elif mutation == "extra_layer":
        state["layers.42.norm1.weight"] = state["layers.0.norm1.weight"]
    elif mutation == "missing_keep":
        state.pop("text_embedder.blocks.1.ffn_gate.bias")
    elif mutation == "wrong_shape":
        state["layers.0.proj.ffn_down.weight"].shape = (3072, 4096)
    elif mutation == "mixed_qkv":
        state["layers.0.proj.q.weight"] = SimpleNamespace(shape=(3072, 3072))
    elif mutation == "already_quantized":
        state["layers.0.proj.qkv.comfy_quant"] = SimpleNamespace(shape=(72,))
    else:
        state["layers.0.proj.qkv.weight"].dtype = "torch.float8_e4m3fn"
    with pytest.raises(contract().MossImageContractError):
        contract().validate_moss_image_state_dict(state)


def test_moss_rejects_unknown_preset():
    with pytest.raises(contract().MossImageContractError, match="preset"):
        contract().validate_moss_image_state_dict(source_state(), preset="unknown")


def test_moss_accepts_optional_rope_and_preserves_attention_configuration():
    state = source_state()
    state["rope.bands"] = SimpleNamespace(shape=(16,), dtype="torch.float32")
    state["layers.0.comfy_attention.config"] = SimpleNamespace(shape=(35,), dtype="torch.uint8")
    assert len(contract().validate_moss_image_state_dict(state)) == 168


@pytest.mark.parametrize("fused,count", [(True, 168), (False, 252)])
def test_model_extraction_and_progress_use_the_detected_layout(fused, count):
    service = importlib.import_module(f"{PACKAGE}.service")
    state = source_state(fused=fused)
    model = SimpleNamespace(model_state_dict=lambda **_: {f"diffusion_model.{k}": v for k, v in state.items()})
    extracted = service.extract_moss_image_state_dict(model)
    assert extracted == state
    assert service.expected_moss_image_quantized_tensors("quality_keep", extracted) == count
    model.patches = {"layer": []}
    with pytest.raises(service.QuantizationNodeError, match="patches"):
        service.extract_moss_image_state_dict(model)


def test_moss_streaming_export_roundtrip_preserves_keep_tensors_and_input(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    export = importlib.import_module(f"{PACKAGE}.quantization.export")
    tensor_spec = importlib.import_module(f"{PACKAGE}.quantization.contracts").TensorSpec
    name = "layers.0.proj.qkv.weight"
    weight = torch.randn(12, 256, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    attention = torch.tensor(list(json.dumps(REFERENCE["markers"][".comfy_attention.config"]).encode()), dtype=torch.uint8)
    state = {name: weight, "final_norm_video.weight": torch.ones(256, dtype=torch.bfloat16), "layers.0.comfy_attention.config": attention}
    originals = {k: v.clone() for k, v in state.items()}
    # Only the full-size architecture check is replaced; run real quantization,
    # streaming safetensors serialization.
    monkeypatch.setattr(export, "validate_moss_image_state_dict", lambda *_a, **_kw: (tensor_spec(name, (12, 256)),))
    path = tmp_path / "moss.safetensors"
    report = export.export_moss_image_int8_convrot_from_state_dict(state_dict=state, output_checkpoint=path)
    output = safetensors.load_file(path)
    assert report.quantized_tensor_count == report.rotated_tensor_count == 1
    assert output[name].dtype == torch.int8
    assert output["layers.0.proj.qkv.weight_scale"].shape == (12, 1)
    assert output["layers.0.proj.qkv.weight_scale"].dtype == torch.float32
    assert json.loads(bytes(output["layers.0.proj.qkv.comfy_quant"].tolist())) == REFERENCE["markers"][".comfy_quant"]
    for key in ("final_norm_video.weight", "layers.0.comfy_attention.config"):
        assert torch.equal(output[key], originals[key])
    for key in state:
        assert torch.equal(state[key], originals[key])
    convrot = importlib.import_module(f"{PACKAGE}.quantization.convrot")
    rotated = convrot.rotate_weight(weight.float(), convrot.build_hadamard(256), 256)
    reconstructed = output[name].float() * output["layers.0.proj.qkv.weight_scale"]
    assert torch.allclose(rotated, reconstructed, atol=0.03, rtol=0)


@pytest.mark.parametrize("reported_count", [168, 167])
def test_moss_publication_checks_counts_before_committing(tmp_path, reported_count):
    service = importlib.import_module(f"{PACKAGE}.service")
    paths = service.resolve_output_paths(tmp_path / "output/diffusion_models", "moss")

    def exporter(**kwargs):
        assert kwargs["family"] == "moss_image"
        assert kwargs["convrot_groupsize"] == 256
        Path(kwargs["output_checkpoint"]).write_bytes(b"checkpoint")
        return {"quantized_tensor_count": reported_count, "rotated_tensor_count": reported_count}

    kwargs = dict(state_dict=source_state(), paths=paths, device="cpu", overwrite=False,
                  write_report=True, hash_output=False, exporter=exporter)
    if reported_count == 167:
        with pytest.raises(service.QuantizationNodeError, match="expected_quantized=168"):
            service.export_moss_image_int8_convrot(**kwargs)
        assert not paths.checkpoint.exists()
        assert not paths.report.exists()
    else:
        report, report_path = service.export_moss_image_int8_convrot(**kwargs)
        assert paths.checkpoint.read_bytes() == b"checkpoint"
        assert json.loads(Path(report_path).read_text())["quantized_tensor_count"] == 168
        assert report["quantized_tensor_count"] == 168


@pytest.mark.parametrize("fused,count", [(True, 168), (False, 252)])
def test_stock_comfyui_detects_moss_layout_and_quant_markers(fused, count):
    comfy_spec = importlib.util.find_spec("comfy")
    if comfy_spec is None or not comfy_spec.submodule_search_locations:
        pytest.skip("ComfyUI checkout required for stock detection validation")
    import ast
    import logging
    import math
    torch = pytest.importorskip("torch")
    comfy_root = Path(next(iter(comfy_spec.submodule_search_locations)))

    def functions(path, names, namespace):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in selected} == names
        exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), "exec"), namespace)
        return namespace

    detection = functions(comfy_root / "model_detection.py",
                          {"any_suffix_in", "calculate_transformer_depth", "count_blocks", "detect_unet_config"}, {"math": math})
    if "moss_image" not in (comfy_root / "model_detection.py").read_text(encoding="utf-8"):
        pytest.skip("This ComfyUI predates native MOSS-Image support")
    quantization = functions(comfy_root / "utils.py", {"detect_layer_quantization"}, {"logging": logging})
    state = source_state(fused=fused)
    marker = torch.tensor(list(json.dumps(REFERENCE["markers"][".comfy_quant"]).encode()), dtype=torch.uint8)
    for item in contract().validate_moss_image_state_dict(state):
        layer = item.name.removesuffix(".weight")
        state[layer + ".comfy_quant"] = marker
        state[layer + ".weight_scale"] = SimpleNamespace(shape=(item.shape[0], 1), dtype=torch.float32)
    config = detection["detect_unet_config"](state, "")
    assert config == {"image_model": "moss_image", "dim": 3072, "ffn_dim": 8192,
                      "num_heads": 24, "num_layers": 42, "fused_qkv": fused}
    quant_config = quantization["detect_layer_quantization"](state, "")
    assert quant_config == {"mixed_ops": True}
    assert sum(key.endswith(".comfy_quant") for key in state) == count
