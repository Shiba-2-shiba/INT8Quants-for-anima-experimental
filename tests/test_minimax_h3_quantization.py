from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_comfy_quants_minimax_h3_tests"


def _load_quantization_package():
    existing = sys.modules.get(PACKAGE_NAME)
    if existing is not None:
        return existing

    package_dir = ROOT / "quantization"
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


_load_quantization_package()
minimax_h3 = importlib.import_module(f"{PACKAGE_NAME}.minimax_h3")
export = importlib.import_module(f"{PACKAGE_NAME}.export")


BLOCK_SUFFIXES = (
    "attn.qkv_proj",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)


class _ShapeOnlyTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


def _shape_only_minimax_h3_state():
    return {
        name: _ShapeOnlyTensor(shape)
        for name, shape in minimax_h3.get_minimax_h3_source_tensor_shapes().items()
    }


def test_minimax_h3_selection_matches_reference_exactly():
    expected = tuple(
        f"blocks.{block}.{suffix}.weight"
        for block in range(50)
        for suffix in BLOCK_SUFFIXES
    )
    names = minimax_h3.get_minimax_h3_tensor_names()

    assert names == expected
    assert len(names) == minimax_h3.expected_quantized_tensors() == 200
    assert len(set(names)) == len(names)
    assert len(minimax_h3.get_minimax_h3_source_tensor_shapes()) == 532
    assert minimax_h3.EXPECTED_KEEP_TENSOR_COUNT == 332


def test_minimax_h3_selected_shapes_match_reference_and_rotate_by_256():
    specs = {
        spec.name: spec.shape for spec in minimax_h3.get_minimax_h3_tensor_specs()
    }

    assert specs["blocks.0.attn.qkv_proj.weight"] == (21504, 5376)
    assert specs["blocks.0.attn.out_proj.weight"] == (5376, 7168)
    assert specs["blocks.0.mlp.fc1.weight"] == (28672, 5376)
    assert specs["blocks.0.mlp.fc2.weight"] == (5376, 14336)
    assert all(shape[1] % 256 == 0 for shape in specs.values())


def test_minimax_h3_selection_keeps_reference_quality_layers():
    names = minimax_h3.get_minimax_h3_tensor_names()
    kept_prefixes = (
        "adaln_t_table",
        "audio_patch_proj.",
        "condition_proj.",
        "final_layer.",
        "rope.",
        "token_refiner.",
        "video_patch_proj.",
    )

    assert all(name.startswith("blocks.") for name in names)
    assert all(not name.startswith(kept_prefixes) for name in names)
    assert all("adaln" not in name and "norm" not in name for name in names)
    assert all(not name.endswith(".bias") for name in names)


def test_minimax_h3_rejects_unknown_preset():
    with pytest.raises(minimax_h3.MiniMaxH3ContractError, match="unsupported.*preset"):
        minimax_h3.get_minimax_h3_tensor_specs("unknown")


def test_minimax_h3_validation_requires_curve_form_and_exact_block_contracts():
    state = _shape_only_minimax_h3_state()
    assert len(minimax_h3.validate_minimax_h3_state_dict(state)) == 200

    without_curve = dict(state)
    del without_curve["adaln_t_table"]
    with pytest.raises(minimax_h3.MiniMaxH3ContractError, match="adaln_t_table"):
        minimax_h3.validate_minimax_h3_state_dict(without_curve)

    wrong_block_count = {
        key: value for key, value in state.items() if not key.startswith("blocks.49.")
    }
    with pytest.raises(minimax_h3.MiniMaxH3ContractError, match="block contract mismatch"):
        minimax_h3.validate_minimax_h3_state_dict(wrong_block_count)

    extra_block = dict(state)
    extra_block["blocks.50.attn.qkv_proj.weight"] = _ShapeOnlyTensor((21504, 5376))
    with pytest.raises(minimax_h3.MiniMaxH3ContractError, match="block contract mismatch"):
        minimax_h3.validate_minimax_h3_state_dict(extra_block)

    wrong_refiner_count = {
        key: value
        for key, value in state.items()
        if not key.startswith("token_refiner.blocks.1.")
    }
    with pytest.raises(minimax_h3.MiniMaxH3ContractError, match="token-refiner.*mismatch"):
        minimax_h3.validate_minimax_h3_state_dict(wrong_refiner_count)

    extra_refiner = dict(state)
    extra_refiner["token_refiner.blocks.2.attn.qkv_proj.weight"] = _ShapeOnlyTensor(
        (21504, 5376)
    )
    with pytest.raises(minimax_h3.MiniMaxH3ContractError, match="token-refiner.*mismatch"):
        minimax_h3.validate_minimax_h3_state_dict(extra_refiner)

    wrong_shape = dict(state)
    wrong_shape["blocks.0.attn.out_proj.weight"] = _ShapeOnlyTensor((5376, 5376))
    with pytest.raises(minimax_h3.MiniMaxH3ContractError, match="shape mismatch"):
        minimax_h3.validate_minimax_h3_state_dict(wrong_shape)


def test_minimax_h3_validation_requires_every_keep_tensor_and_no_extras():
    state = _shape_only_minimax_h3_state()
    missing_keep = dict(state)
    del missing_keep["token_refiner.final_norm.weight"]
    with pytest.raises(minimax_h3.MiniMaxH3ContractError, match="complete source.*missing"):
        minimax_h3.validate_minimax_h3_state_dict(missing_keep)

    unexpected = dict(state)
    unexpected["future_adapter.weight"] = _ShapeOnlyTensor((1, 1))
    with pytest.raises(minimax_h3.MiniMaxH3ContractError, match="complete source.*unexpected"):
        minimax_h3.validate_minimax_h3_state_dict(unexpected)


def test_minimax_h3_rejects_time_embedder_variant():
    state = _shape_only_minimax_h3_state()
    state["time_embedder.proj_in.weight"] = _ShapeOnlyTensor((1, 1))

    with pytest.raises(minimax_h3.MiniMaxH3ContractError, match="curve-form"):
        minimax_h3.validate_minimax_h3_state_dict(state)


def test_public_minimax_h3_export_uses_internal_contract(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    safe_open = pytest.importorskip("safetensors").safe_open
    spec = minimax_h3.MiniMaxH3TensorSpec(
        name="blocks.0.attn.qkv_proj.weight",
        shape=(4, 256),
    )
    keep_name = "token_refiner.final_norm.weight"
    state = {
        spec.name: torch.zeros(spec.shape, dtype=torch.bfloat16),
        keep_name: torch.arange(4, dtype=torch.float32),
    }
    original_keys = tuple(state)
    original_keep = state[keep_name].clone()
    validated = {}

    def fake_validate(candidate, *, preset, require_selected_tensors):
        validated.update(
            candidate=candidate,
            preset=preset,
            require_selected_tensors=require_selected_tensors,
        )
        return (spec,)

    monkeypatch.setattr(export, "validate_minimax_h3_state_dict", fake_validate)
    output = tmp_path / "minimax-h3.safetensors"
    report = export.export_minimax_h3_int8_convrot_from_state_dict(
        state_dict=state,
        output_checkpoint=output,
    )

    assert validated == {
        "candidate": state,
        "preset": "strict_reference",
        "require_selected_tensors": True,
    }
    assert report.quantized_tensor_count == 1
    assert report.quantization_preset == "strict_reference"
    exported = safetensors_torch.load_file(str(output))
    assert exported[spec.name].dtype == torch.int8
    assert exported["blocks.0.attn.qkv_proj.weight_scale"].dtype == torch.float32
    assert exported["blocks.0.attn.qkv_proj.comfy_quant"].dtype == torch.uint8
    assert torch.equal(exported[keep_name], original_keep)
    assert tuple(state) == original_keys
    assert torch.equal(state[keep_name], original_keep)
    with safe_open(str(output), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
    assert metadata is not None
    assert "config" not in metadata


def test_public_minimax_h3_export_rejects_other_family(tmp_path):
    with pytest.raises(export.QuantizationExportError, match="only family='minimax_h3'"):
        export.export_minimax_h3_int8_convrot_from_state_dict(
            state_dict={},
            output_checkpoint=tmp_path / "wrong-family.safetensors",
            family="minimax_h4",
        )


def test_public_minimax_h3_export_accepts_fp32_and_rejects_fp16(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    spec = minimax_h3.MiniMaxH3TensorSpec(
        name="blocks.0.attn.qkv_proj.weight",
        shape=(4, 256),
    )
    monkeypatch.setattr(
        export,
        "validate_minimax_h3_state_dict",
        lambda *_args, **_kwargs: (spec,),
    )

    export.export_minimax_h3_int8_convrot_from_state_dict(
        state_dict={spec.name: torch.zeros(spec.shape, dtype=torch.float32)},
        output_checkpoint=tmp_path / "fp32.safetensors",
    )
    with pytest.raises(export.QuantizationExportError, match="bfloat16 or float32"):
        export.export_minimax_h3_int8_convrot_from_state_dict(
            state_dict={spec.name: torch.zeros(spec.shape, dtype=torch.float16)},
            output_checkpoint=tmp_path / "fp16.safetensors",
        )
