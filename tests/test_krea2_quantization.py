from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_comfy_quants_krea2_tests"


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
krea2 = importlib.import_module(f"{PACKAGE_NAME}.krea2")
export = importlib.import_module(f"{PACKAGE_NAME}.export")


BLOCK_SUFFIXES = (
    "attn.wq",
    "attn.wk",
    "attn.wv",
    "attn.gate",
    "attn.wo",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
)


def _block_names(block: int) -> tuple[str, ...]:
    return tuple(f"blocks.{block}.{suffix}.weight" for suffix in BLOCK_SUFFIXES)


def _shape_only_krea2_state():
    state = {
        spec.name: _ShapeOnlyTensor(spec.shape)
        for spec in krea2.get_krea2_tensor_specs()
    }
    state["first.weight"] = _ShapeOnlyTensor((6144, 64))
    state["txtfusion.projector.weight"] = _ShapeOnlyTensor((1, 12))
    state["txtfusion.layerwise_blocks.0.prenorm.scale"] = _ShapeOnlyTensor((2560,))
    state["last.linear.weight"] = _ShapeOnlyTensor((64, 6144))
    return state


class _ShapeOnlyTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def dim(self):
        return len(self.shape)


def test_krea2_selection_is_exact_and_deterministic():
    expected = tuple(name for block in range(28) for name in _block_names(block))
    names = krea2.get_krea2_tensor_names()

    assert names == expected
    assert len(names) == krea2.expected_quantized_tensors() == 224
    assert len(set(names)) == len(names)


def test_krea2_selection_keeps_non_block_layers_high_precision():
    names = krea2.get_krea2_tensor_names()

    assert all(name.startswith("blocks.") for name in names)
    assert all("qnorm" not in name and "knorm" not in name for name in names)
    assert all(not name.startswith(("first.", "last.", "tmlp.", "txtfusion.", "txtmlp.", "tproj.")) for name in names)


def test_krea2_tensor_shapes_match_the_open_weight_contract():
    specs = {spec.name: spec.shape for spec in krea2.get_krea2_tensor_specs()}

    assert specs["blocks.0.attn.wq.weight"] == (6144, 6144)
    assert specs["blocks.0.attn.wk.weight"] == (1536, 6144)
    assert specs["blocks.0.attn.wv.weight"] == (1536, 6144)
    assert specs["blocks.0.attn.gate.weight"] == (6144, 6144)
    assert specs["blocks.0.attn.wo.weight"] == (6144, 6144)
    assert specs["blocks.0.mlp.gate.weight"] == (16384, 6144)
    assert specs["blocks.0.mlp.up.weight"] == (16384, 6144)
    assert specs["blocks.0.mlp.down.weight"] == (6144, 16384)
    assert all(shape[1] % 256 == 0 for shape in specs.values())


def test_krea2_rejects_unknown_preset():
    with pytest.raises(krea2.Krea2ContractError, match="unsupported.*preset"):
        krea2.get_krea2_tensor_specs("unknown")


def test_krea2_validation_requires_signature_blocks_and_selected_shapes():
    state = _shape_only_krea2_state()
    specs = krea2.validate_krea2_state_dict(state)
    assert len(specs) == 224

    without_signature = dict(state)
    del without_signature["txtfusion.projector.weight"]
    with pytest.raises(krea2.Krea2ContractError, match="txtfusion.projector"):
        krea2.validate_krea2_state_dict(without_signature)

    without_block = {key: value for key, value in state.items() if not key.startswith("blocks.27.")}
    with pytest.raises(krea2.Krea2ContractError, match="block contract mismatch"):
        krea2.validate_krea2_state_dict(without_block)

    wrong_shape = dict(state)
    wrong_shape["blocks.0.attn.wk.weight"] = _ShapeOnlyTensor((6144, 6144))
    with pytest.raises(krea2.Krea2ContractError, match="shape mismatch"):
        krea2.validate_krea2_state_dict(wrong_shape)


def test_public_krea2_export_uses_internal_contract(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    spec = krea2.Krea2TensorSpec(
        name="blocks.0.attn.wq.weight",
        shape=(4, 256),
    )
    state = {spec.name: torch.zeros(spec.shape, dtype=torch.bfloat16)}
    validated = {}

    def fake_validate(candidate, *, preset, require_selected_tensors):
        validated.update(
            candidate=candidate,
            preset=preset,
            require_selected_tensors=require_selected_tensors,
        )
        return (spec,)

    monkeypatch.setattr(export, "validate_krea2_state_dict", fake_validate)
    report = export.export_krea2_int8_convrot_from_state_dict(
        state_dict=state,
        output_checkpoint=tmp_path / "krea2.safetensors",
    )

    assert validated == {
        "candidate": state,
        "preset": "quality_keep",
        "require_selected_tensors": True,
    }
    assert report.quantized_tensor_count == 1
    assert report.quantization_preset == "quality_keep"
