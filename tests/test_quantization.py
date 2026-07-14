from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_comfy_quants_quantization_tests"


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
anima = importlib.import_module(f"{PACKAGE_NAME}.anima")
convrot = importlib.import_module(f"{PACKAGE_NAME}.convrot")
export = importlib.import_module(f"{PACKAGE_NAME}.export")


BLOCK_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.k_proj",
    "cross_attn.v_proj",
    "cross_attn.output_proj",
    "mlp.layer1",
    "mlp.layer2",
    "adaln_modulation_self_attn.1",
    "adaln_modulation_self_attn.2",
    "adaln_modulation_cross_attn.1",
    "adaln_modulation_cross_attn.2",
    "adaln_modulation_mlp.1",
    "adaln_modulation_mlp.2",
)
BLOCK1_QUALITY_SUFFIXES = BLOCK_SUFFIXES[:10]


def _block_names(block: int, suffixes=BLOCK_SUFFIXES) -> tuple[str, ...]:
    return tuple(f"net.blocks.{block}.{suffix}.weight" for suffix in suffixes)


def _expected_public_names() -> tuple[str, ...]:
    return tuple(name for block in range(28) for name in _block_names(block))


def _expected_quality_names() -> tuple[str, ...]:
    return _block_names(1, BLOCK1_QUALITY_SUFFIXES) + tuple(
        name for block in range(2, 28) for name in _block_names(block)
    )


def test_anima_presets_are_exact_and_deterministic():
    quality = anima.get_anima_tensor_names("quality_keep")
    public = anima.get_anima_tensor_names("public_examples")

    assert quality == _expected_quality_names()
    assert public == _expected_public_names()
    assert len(quality) == anima.expected_quantized_tensors("quality_keep") == 426
    assert len(public) == anima.expected_quantized_tensors("public_examples") == 448
    assert len(set(quality)) == len(quality)
    assert len(set(public)) == len(public)


def test_public_examples_adds_only_the_documented_22_weights():
    quality = set(anima.get_anima_tensor_names("quality_keep"))
    public = set(anima.get_anima_tensor_names("public_examples"))
    expected_extra = set(_block_names(0)) | set(
        _block_names(1, BLOCK_SUFFIXES[10:])
    )

    assert public - quality == expected_extra
    assert quality < public
    assert len(public - quality) == 22


@pytest.mark.parametrize("preset", ["quality_keep", "public_examples"])
def test_anima_selection_boundaries_keep_non_block_weights_high_precision(preset):
    names = anima.get_anima_tensor_names(preset)

    assert all(name.startswith("net.blocks.") for name in names)
    assert all(".q_norm." not in name and ".k_norm." not in name for name in names)
    assert all("net.llm_adapter" not in name for name in names)
    assert all("net.final_layer" not in name for name in names)
    assert all("net.t_embedder" not in name for name in names)
    assert all("net.x_embedder" not in name for name in names)


def test_anima_tensor_shapes_match_the_2b_contract():
    specs = {spec.name: spec.shape for spec in anima.get_anima_tensor_specs("public_examples")}

    assert specs["net.blocks.0.self_attn.q_proj.weight"] == (2048, 2048)
    assert specs["net.blocks.0.cross_attn.k_proj.weight"] == (2048, 1024)
    assert specs["net.blocks.0.cross_attn.v_proj.weight"] == (2048, 1024)
    assert specs["net.blocks.0.mlp.layer1.weight"] == (8192, 2048)
    assert specs["net.blocks.0.mlp.layer2.weight"] == (2048, 8192)
    assert specs["net.blocks.0.adaln_modulation_self_attn.1.weight"] == (256, 2048)
    assert specs["net.blocks.0.adaln_modulation_self_attn.2.weight"] == (6144, 256)


def test_anima_rejects_unknown_preset():
    with pytest.raises(anima.AnimaContractError, match="unsupported.*preset"):
        anima.get_anima_tensor_specs("unknown")


class _ShapeOnlyTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def dim(self):
        return len(self.shape)


def _shape_only_anima_state(preset="quality_keep"):
    state = {
        spec.name: _ShapeOnlyTensor(spec.shape)
        for spec in anima.get_anima_tensor_specs(preset)
    }
    state["net.x_embedder.proj.1.weight"] = _ShapeOnlyTensor((2048, 16))
    state["net.llm_adapter.blocks.0.cross_attn.q_proj.weight"] = _ShapeOnlyTensor((8, 8))
    state["net.blocks.0.self_attn.q_norm.weight"] = _ShapeOnlyTensor((128,))
    return state


def test_anima_validation_requires_standard_signature_and_all_selected_weights():
    state = _shape_only_anima_state()
    specs = anima.validate_anima_state_dict(state, preset="quality_keep")
    assert len(specs) == 426

    without_llm = dict(state)
    del without_llm["net.llm_adapter.blocks.0.cross_attn.q_proj.weight"]
    with pytest.raises(anima.AnimaContractError, match="llm_adapter"):
        anima.validate_anima_state_dict(without_llm, preset="quality_keep")

    without_selected = dict(state)
    del without_selected["net.blocks.2.self_attn.q_proj.weight"]
    with pytest.raises(anima.AnimaContractError, match="missing"):
        anima.validate_anima_state_dict(without_selected, preset="quality_keep")


def test_anima_validation_rejects_14b_and_wrong_selected_shape():
    state = _shape_only_anima_state()
    state["net.x_embedder.proj.1.weight"] = _ShapeOnlyTensor((5120, 16))
    with pytest.raises(anima.AnimaContractError, match="2048|2B"):
        anima.validate_anima_state_dict(state)

    state = _shape_only_anima_state()
    state["net.blocks.2.self_attn.q_proj.weight"] = _ShapeOnlyTensor((2048, 1024))
    with pytest.raises(anima.AnimaContractError, match="shape"):
        anima.validate_anima_state_dict(state)


def test_regular_hadamard_is_normalized_orthogonal_and_cached():
    torch = pytest.importorskip("torch")
    expected = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
    ) / 2

    actual = convrot.build_hadamard(4, dtype=torch.float32)
    assert torch.equal(actual, expected)
    assert torch.allclose(actual @ actual.T, torch.eye(4))
    assert convrot.build_hadamard(4, dtype=torch.float32) is actual


@pytest.mark.parametrize("size", [0, 2, 8, 32])
def test_hadamard_rejects_non_power_of_four(size):
    with pytest.raises(convrot.ConvRotError, match="power of four"):
        convrot.build_hadamard(size)


def test_rotate_weight_matches_grouped_matrix_multiplication_without_mutation():
    torch = pytest.importorskip("torch")
    weight = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    snapshot = weight.clone()
    hadamard = convrot.build_hadamard(4, dtype=weight.dtype)

    actual = convrot.rotate_weight(weight, hadamard, 4)
    expected = (weight.reshape(2, 2, 4) @ hadamard.T).reshape_as(weight)

    assert torch.equal(actual, expected)
    assert torch.equal(weight, snapshot)
    assert actual.shape == weight.shape


def test_rotate_weight_rejects_invalid_rank_and_group_size():
    torch = pytest.importorskip("torch")
    hadamard = convrot.build_hadamard(4)
    with pytest.raises(convrot.ConvRotError, match="rank-2"):
        convrot.rotate_weight(torch.zeros(8), hadamard, 4)
    with pytest.raises(convrot.ConvRotError, match="divisible"):
        convrot.rotate_weight(torch.zeros(2, 6), hadamard, 4)


def _small_spec(in_features=256):
    return anima.AnimaTensorSpec(
        name="net.blocks.2.self_attn.q_proj.weight",
        shape=(4, in_features),
    )


@pytest.mark.parametrize("source_dtype", ["float16", "bfloat16"])
def test_small_export_matches_int8_rowwise_contract_and_preserves_input(tmp_path, source_dtype):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    dtype = getattr(torch, source_dtype)
    generator = torch.Generator().manual_seed(7301)
    name = _small_spec().name
    state = {
        name: torch.randn(4, 256, dtype=dtype, generator=generator),
        "net.final_layer.linear.weight": torch.randn(2, 3, generator=generator),
    }
    original_keys = tuple(state)
    snapshots = {key: tensor.clone() for key, tensor in state.items()}
    output = tmp_path / f"small-{source_dtype}.safetensors"

    report = export._write_int8_convrot_checkpoint_from_specs(
        state_dict=state,
        output_checkpoint=output,
        tensor_specs=(_small_spec(),),
        convrot=True,
        convrot_groupsize=256,
        device="cpu",
        validate_source_dtype=source_dtype == "bfloat16",
        require_all_rotated=True,
        hash_output=True,
        metadata={"model_family": "anima", "quantization_preset": "quality_keep"},
    )
    exported = safetensors_torch.load_file(str(output))

    hadamard = convrot.build_hadamard(256, dtype=dtype)
    rotated = convrot.rotate_weight(snapshots[name], hadamard, 256)
    expected_scale = (rotated.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    expected_quant = (
        rotated / expected_scale.to(dtype)
    ).round().clamp(-128.0, 127.0).to(torch.int8)

    assert torch.equal(exported[name], expected_quant)
    assert torch.equal(exported["net.blocks.2.self_attn.q_proj.weight_scale"], expected_scale)
    marker = exported["net.blocks.2.self_attn.q_proj.comfy_quant"]
    assert marker.dtype == torch.uint8
    assert bytes(marker.tolist()) == (
        b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
    )
    assert json.loads(bytes(marker.tolist()).decode("utf-8")) == {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
    }
    assert torch.equal(exported["net.final_layer.linear.weight"], state["net.final_layer.linear.weight"])

    assert tuple(state) == original_keys
    assert all(torch.equal(state[key], snapshot) for key, snapshot in snapshots.items())
    assert report.quantized_tensor_count == 1
    assert report.rotated_tensor_count == 1
    assert report.nonrotated_tensor_count == 0
    assert report.copied_tensor_count == 1
    assert report.output_tensor_count == 4
    assert report.quant_metadata_tensor_count == 1
    assert report.scale_tensor_count == 1
    assert report.output_hash_state == "written"
    assert report.output_hash


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_small_export_rejects_nonfinite_selected_weight(tmp_path, nonfinite):
    torch = pytest.importorskip("torch")
    spec = _small_spec()
    weight = torch.zeros(spec.shape, dtype=torch.float16)
    weight[0, 0] = nonfinite

    with pytest.raises(export.QuantizationExportError, match="non-finite"):
        export._write_int8_convrot_checkpoint_from_specs(
            state_dict={spec.name: weight},
            output_checkpoint=tmp_path / "nonfinite.safetensors",
            tensor_specs=(spec,),
            validate_source_dtype=False,
        )


@pytest.mark.parametrize(
    ("state_factory", "message"),
    [
        (lambda torch, name: {}, "missing"),
        (lambda torch, name: {name: torch.zeros(1024, dtype=torch.bfloat16)}, "rank"),
        (lambda torch, name: {name: torch.zeros(5, 256, dtype=torch.bfloat16)}, "shape"),
        (lambda torch, name: {name: torch.zeros(4, 256, dtype=torch.float32)}, "dtype"),
        (lambda torch, name: {name: torch.zeros(4, 256, dtype=torch.int8)}, "dtype"),
    ],
)
def test_small_export_rejects_missing_rank_shape_and_dtype(tmp_path, state_factory, message):
    torch = pytest.importorskip("torch")
    spec = _small_spec()
    with pytest.raises(export.QuantizationExportError, match=message):
        export._write_int8_convrot_checkpoint_from_specs(
            state_dict=state_factory(torch, spec.name),
            output_checkpoint=tmp_path / f"invalid-{message}.safetensors",
            tensor_specs=(spec,),
            validate_source_dtype=True,
        )


def test_small_export_rejects_nonrotatable_weight_when_rotation_is_required(tmp_path):
    torch = pytest.importorskip("torch")
    spec = _small_spec(192)
    state = {spec.name: torch.randn(4, 192, dtype=torch.bfloat16)}

    with pytest.raises(export.QuantizationExportError, match="required for every selected tensor"):
        export._write_int8_convrot_checkpoint_from_specs(
            state_dict=state,
            output_checkpoint=tmp_path / "invalid-rotation.safetensors",
            tensor_specs=(spec,),
            convrot=True,
            convrot_groupsize=256,
            require_all_rotated=True,
        )


def test_small_export_report_and_header_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors")
    spec = _small_spec()
    output = tmp_path / "metadata.safetensors"
    report = export._write_int8_convrot_checkpoint_from_specs(
        state_dict={spec.name: torch.zeros(spec.shape, dtype=torch.bfloat16)},
        output_checkpoint=output,
        tensor_specs=(spec,),
        metadata={"model_family": "anima", "custom_number": 7},
    )

    with safetensors.safe_open(str(output), framework="pt") as handle:
        metadata = handle.metadata()

    assert report.quantized_tensor_count == report.rotated_tensor_count == 1
    assert report.output_tensor_count == 3
    assert metadata["artifact_target"] == "comfyui_diffusion_model"
    assert metadata["artifact_contract"] == "int8_tensorwise_inference_checkpoint.v1"
    assert metadata["target_dtype"] == "int8_tensorwise"
    assert metadata["quantized_tensor_count"] == "1"
    assert metadata["model_family"] == "anima"
    assert metadata["custom_number"] == "7"


def test_export_reuses_cpu_storage_but_clones_shared_aliases(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    spec = _small_spec()
    shared = torch.randn(2, 4)
    state = {
        spec.name: torch.zeros(spec.shape, dtype=torch.bfloat16),
        "net.final_layer.first": shared,
        "net.final_layer.second": shared,
    }
    captured = {}

    def fake_save_file(tensors, path, metadata):
        captured.update(tensors)
        Path(path).write_bytes(b"safetensors fixture")

    monkeypatch.setattr(export, "_require_safetensors_save_file", lambda: fake_save_file)
    export._write_int8_convrot_checkpoint_from_specs(
        state_dict=state,
        output_checkpoint=tmp_path / "shared-storage.safetensors",
        tensor_specs=(spec,),
    )

    first = captured["net.final_layer.first"]
    second = captured["net.final_layer.second"]
    assert first.untyped_storage().data_ptr() == shared.untyped_storage().data_ptr()
    assert second.untyped_storage().data_ptr() != shared.untyped_storage().data_ptr()
    assert torch.equal(first, shared)
    assert torch.equal(second, shared)


def test_public_export_uses_internal_anima_contract(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    spec = _small_spec()
    state = {spec.name: torch.zeros(spec.shape, dtype=torch.bfloat16)}
    validated = {}

    def fake_validate(candidate, *, preset, require_selected_tensors):
        validated.update(
            candidate=candidate,
            preset=preset,
            require_selected_tensors=require_selected_tensors,
        )
        return (spec,)

    monkeypatch.setattr(export, "validate_anima_state_dict", fake_validate)
    report = export.export_anima_int8_convrot_from_state_dict(
        state_dict=state,
        output_checkpoint=tmp_path / "public-api.safetensors",
        family="anima",
        quantization_preset="public_examples",
    )

    assert validated == {
        "candidate": state,
        "preset": "public_examples",
        "require_selected_tensors": True,
    }
    assert report.quantized_tensor_count == 1
    assert report.quantization_preset == "public_examples"


@pytest.mark.parametrize("family", ["anima_14b", "qwen_image"])
def test_public_export_rejects_unsupported_family(tmp_path, family):
    with pytest.raises(export.QuantizationExportError, match="unsupported INT8"):
        export.export_anima_int8_convrot_from_state_dict(
            state_dict={},
            output_checkpoint=tmp_path / "unsupported.safetensors",
            family=family,
        )
