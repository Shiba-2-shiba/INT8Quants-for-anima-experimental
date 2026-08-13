from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_comfy_quants_service_tests"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE_NAME] = package
SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.service",
    ROOT / "service.py",
)
assert SPEC is not None and SPEC.loader is not None
service = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = service
SPEC.loader.exec_module(service)


class FakeTensor:
    def __init__(self, shape, dtype="torch.float32", element_bytes=4):
        self.shape = shape
        self.dtype = dtype
        self.element_bytes = element_bytes

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def element_size(self):
        return self.element_bytes


def anima_state_dict(channels=2048):
    state = {
        "diffusion_model.x_embedder.proj.1.weight": FakeTensor((channels, 16)),
        "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight": FakeTensor((8, 8)),
    }
    for block in range(28):
        state[f"diffusion_model.blocks.{block}.self_attn.q_proj.weight"] = FakeTensor((8, 256))
    return state


def krea2_state_dict(features=6144, layers=28):
    state = {
        "diffusion_model.first.weight": FakeTensor((features, 64)),
        "diffusion_model.txtfusion.projector.weight": FakeTensor((1, 12)),
        "diffusion_model.txtfusion.layerwise_blocks.0.prenorm.scale": FakeTensor((2560,)),
    }
    for block in range(layers):
        state[f"diffusion_model.blocks.{block}.attn.wq.weight"] = FakeTensor((features, features))
    return state


def minimax_h3_state_dict(layers=50, token_refiner_layers=2):
    state = {
        "diffusion_model.video_patch_proj.weight": FakeTensor((5376, 96)),
        "diffusion_model.audio_patch_proj.weight": FakeTensor((5376, 32)),
        "diffusion_model.condition_proj.weight": FakeTensor((5376, 5120)),
        "diffusion_model.final_layer.video_out.weight": FakeTensor((96, 5376)),
        "diffusion_model.final_layer.video_out.bias": FakeTensor((96,)),
        "diffusion_model.final_layer.audio_out.weight": FakeTensor((32, 5376)),
        "diffusion_model.final_layer.audio_out.bias": FakeTensor((32,)),
        "diffusion_model.final_layer.adaln_proj.linear.weight": FakeTensor((10752, 8)),
        "diffusion_model.final_layer.adaln_proj.linear.bias": FakeTensor((10752,)),
        "diffusion_model.final_layer.norm.weight": FakeTensor((5376,)),
        "diffusion_model.adaln_t_table": FakeTensor((1025, 8)),
        "diffusion_model.rope.inv_freq": FakeTensor((16,)),
        "diffusion_model.video_patch_proj.bias": FakeTensor((5376,)),
        "diffusion_model.audio_patch_proj.bias": FakeTensor((5376,)),
        "diffusion_model.condition_proj.bias": FakeTensor((5376,)),
        "diffusion_model.token_refiner.final_norm.weight": FakeTensor((5376,)),
    }
    for block in range(layers):
        state[f"diffusion_model.blocks.{block}.adaln_proj.linear.weight"] = FakeTensor(
            (96768, 8)
        )
        state[f"diffusion_model.blocks.{block}.adaln_proj.linear.bias"] = FakeTensor(
            (96768,)
        )
        state[f"diffusion_model.blocks.{block}.attn.q_norm.weight"] = FakeTensor((128,))
        state[f"diffusion_model.blocks.{block}.attn.k_norm.weight"] = FakeTensor((128,))
        state[f"diffusion_model.blocks.{block}.norm1.weight"] = FakeTensor((5376,))
        state[f"diffusion_model.blocks.{block}.norm2.weight"] = FakeTensor((5376,))
        state[f"diffusion_model.blocks.{block}.attn.qkv_proj.weight"] = FakeTensor(
            (21504, 5376)
        )
        state[f"diffusion_model.blocks.{block}.attn.out_proj.weight"] = FakeTensor(
            (5376, 7168)
        )
        state[f"diffusion_model.blocks.{block}.mlp.fc1.weight"] = FakeTensor(
            (28672, 5376)
        )
        state[f"diffusion_model.blocks.{block}.mlp.fc2.weight"] = FakeTensor(
            (5376, 14336)
        )
    for block in range(token_refiner_layers):
        state[f"diffusion_model.token_refiner.blocks.{block}.attn.q_norm.weight"] = FakeTensor(
            (128,)
        )
        state[f"diffusion_model.token_refiner.blocks.{block}.attn.k_norm.weight"] = FakeTensor(
            (128,)
        )
        state[f"diffusion_model.token_refiner.blocks.{block}.norm1.weight"] = FakeTensor(
            (5376,)
        )
        state[f"diffusion_model.token_refiner.blocks.{block}.norm2.weight"] = FakeTensor(
            (5376,)
        )
        state[f"diffusion_model.token_refiner.blocks.{block}.attn.qkv_proj.weight"] = FakeTensor(
            (21504, 5376)
        )
        state[f"diffusion_model.token_refiner.blocks.{block}.attn.out_proj.weight"] = FakeTensor(
            (5376, 7168)
        )
        state[f"diffusion_model.token_refiner.blocks.{block}.mlp.fc1.weight"] = FakeTensor(
            (28672, 5376)
        )
        state[f"diffusion_model.token_refiner.blocks.{block}.mlp.fc2.weight"] = FakeTensor(
            (5376, 14336)
        )
    return state


MINIMAX_H3_EFFECTIVE_CONFIG = {
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


class FakeModel:
    def __init__(
        self,
        state=None,
        patches=None,
        object_patches=None,
        weight_wrapper_patches=None,
        **patcher_state,
    ):
        self.state = state if state is not None else anima_state_dict()
        self.patches = patches if patches is not None else {}
        self.object_patches = object_patches if object_patches is not None else {}
        self.weight_wrapper_patches = (
            weight_wrapper_patches if weight_wrapper_patches is not None else {}
        )
        self.hook_patches = patcher_state.pop("hook_patches", {})
        self.hook_patches_backup = patcher_state.pop("hook_patches_backup", None)
        self.cached_hook_patches = patcher_state.pop("cached_hook_patches", {})
        self.current_hooks = patcher_state.pop("current_hooks", None)
        self.injections = patcher_state.pop("injections", {})
        self.model_options = patcher_state.pop(
            "model_options", {"transformer_options": {}}
        )
        self.model = patcher_state.pop("base_model", None)
        assert not patcher_state
        self.filter_prefix = None

    def model_state_dict(self, filter_prefix=None):
        self.filter_prefix = filter_prefix
        return {key: value for key, value in self.state.items() if key.startswith(filter_prefix or "")}


def minimax_h3_model(*, state=None, config=None):
    unet_config = dict(MINIMAX_H3_EFFECTIVE_CONFIG if config is None else config)
    unet_config.setdefault("dtype", "torch.bfloat16")
    model_config = types.SimpleNamespace(unet_config=unet_config)
    base_model = types.SimpleNamespace(model_config=model_config)
    return FakeModel(state=state or minimax_h3_state_dict(), base_model=base_model)


def test_extract_normalizes_diffusion_prefix():
    model = FakeModel()
    result = service.extract_anima_state_dict(model)
    assert model.filter_prefix == "diffusion_model."
    assert "net.x_embedder.proj.1.weight" in result
    assert all(not key.startswith("diffusion_model.") for key in result)


def test_extract_does_not_duplicate_existing_net_prefix():
    state = {
        key.replace("diffusion_model.", "diffusion_model.net.", 1): value
        for key, value in anima_state_dict().items()
    }
    result = service.extract_anima_state_dict(FakeModel(state=state))
    assert "net.x_embedder.proj.1.weight" in result
    assert all(not key.startswith("net.net.") for key in result)


def test_extract_rejects_patched_model():
    with pytest.raises(service.QuantizationNodeError, match="contains patches"):
        service.extract_anima_state_dict(FakeModel(patches={"diffusion_model.net.blocks.1.weight": []}))


@pytest.mark.parametrize(
    "field",
    [
        "object_patches",
        "weight_wrapper_patches",
        "hook_patches",
        "hook_patches_backup",
        "cached_hook_patches",
        "current_hooks",
        "injections",
    ],
)
def test_extract_rejects_other_model_patcher_mutations(field):
    with pytest.raises(service.QuantizationNodeError, match=field):
        service.extract_anima_state_dict(FakeModel(**{field: {"active": object()}}))


@pytest.mark.parametrize(
    "model_options",
    [
        {"transformer_options": {"patches_replace": {"active": object()}}},
        {"transformer_options": {}, "model_function_wrapper": object()},
    ],
)
def test_extract_rejects_modified_model_options(model_options):
    with pytest.raises(service.QuantizationNodeError, match="model_options"):
        service.extract_anima_state_dict(FakeModel(model_options=model_options))


def test_extract_rejects_quantized_model():
    state = anima_state_dict()
    state["diffusion_model.net.blocks.1.self_attn.q_proj.comfy_quant"] = FakeTensor((20,), "torch.uint8")
    with pytest.raises(service.QuantizationNodeError, match="already quantized"):
        service.extract_anima_state_dict(FakeModel(state=state))


def test_extract_rejects_anima_14b():
    with pytest.raises(service.QuantizationNodeError, match="Anima 14B"):
        service.extract_anima_state_dict(FakeModel(state=anima_state_dict(5120)))


def test_extract_rejects_block_contract_mismatch():
    state = anima_state_dict()
    state = {key: value for key, value in state.items() if ".blocks.27." not in key}
    with pytest.raises(service.QuantizationNodeError, match="block contract mismatch"):
        service.extract_anima_state_dict(FakeModel(state=state))


def test_extract_rejects_non_anima_without_llm_adapter_signature():
    state = anima_state_dict()
    del state["diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight"]
    with pytest.raises(service.QuantizationNodeError, match="llm_adapter"):
        service.extract_anima_state_dict(FakeModel(state=state))


def test_extract_krea2_strips_diffusion_prefix_without_adding_net():
    model = FakeModel(state=krea2_state_dict())
    result = service.extract_krea2_state_dict(model)

    assert model.filter_prefix == "diffusion_model."
    assert "first.weight" in result
    assert "txtfusion.projector.weight" in result
    assert all(not key.startswith(("diffusion_model.", "net.")) for key in result)


def test_extract_krea2_rejects_wrong_width_and_missing_blocks():
    with pytest.raises(service.QuantizationNodeError, match="6144|Krea2"):
        service.extract_krea2_state_dict(FakeModel(state=krea2_state_dict(features=4096)))

    with pytest.raises(service.QuantizationNodeError, match="block contract mismatch"):
        service.extract_krea2_state_dict(FakeModel(state=krea2_state_dict(layers=27)))


def test_extract_krea2_rejects_anima_model():
    with pytest.raises(service.QuantizationNodeError, match="Krea2|txtfusion.projector"):
        service.extract_krea2_state_dict(FakeModel(state=anima_state_dict()))


def test_extract_minimax_h3_preserves_native_namespace_and_exact_effective_config():
    model = minimax_h3_model()
    result = service.extract_minimax_h3_state_dict(model)

    assert model.filter_prefix == "diffusion_model."
    assert "video_patch_proj.weight" in result
    assert "blocks.49.mlp.fc2.weight" in result
    assert all(not key.startswith(("diffusion_model.", "net.")) for key in result)


def test_extract_minimax_h3_rejects_non_reference_curve_or_effective_config():
    time_embedder = minimax_h3_state_dict()
    del time_embedder["diffusion_model.adaln_t_table"]
    time_embedder["diffusion_model.time_embedder.proj_in.weight"] = FakeTensor((256, 256))
    with pytest.raises(service.QuantizationNodeError, match="adaln_t_table|curve"):
        service.extract_minimax_h3_state_dict(minimax_h3_model(state=time_embedder))

    changed_config = dict(MINIMAX_H3_EFFECTIVE_CONFIG, num_layers=49)
    with pytest.raises(service.QuantizationNodeError, match="effective config|num_layers"):
        service.extract_minimax_h3_state_dict(minimax_h3_model(config=changed_config))

    metadata_override = dict(MINIMAX_H3_EFFECTIVE_CONFIG, patch_size=(1, 2, 2))
    with pytest.raises(service.QuantizationNodeError, match="unexpected transformer config"):
        service.extract_minimax_h3_state_dict(minimax_h3_model(config=metadata_override))

    unsupported_dtype = dict(MINIMAX_H3_EFFECTIVE_CONFIG, dtype="torch.float16")
    with pytest.raises(service.QuantizationNodeError, match="bfloat16 or torch.float32"):
        service.extract_minimax_h3_state_dict(minimax_h3_model(config=unsupported_dtype))


def test_extract_minimax_h3_rejects_missing_model_config_and_quantized_input():
    with pytest.raises(service.QuantizationNodeError, match="effective config"):
        service.extract_minimax_h3_state_dict(FakeModel(state=minimax_h3_state_dict()))

    missing_dtype = minimax_h3_model()
    del missing_dtype.model.model_config.unet_config["dtype"]
    with pytest.raises(service.QuantizationNodeError, match="bfloat16 or torch.float32"):
        service.extract_minimax_h3_state_dict(missing_dtype)

    quantized = minimax_h3_state_dict()
    quantized["diffusion_model.blocks.0.attn.qkv_proj.comfy_quant"] = FakeTensor(
        (80,), "torch.uint8"
    )
    with pytest.raises(service.QuantizationNodeError, match="already quantized"):
        service.extract_minimax_h3_state_dict(minimax_h3_model(state=quantized))


def test_resolve_output_uses_output_diffusion_models_root_and_strips_extension(tmp_path):
    output_root = tmp_path / "output" / "diffusion_models"
    paths = service.resolve_output_paths(output_root, "variants/anima.safetensors")
    assert paths.checkpoint == output_root / "variants" / "anima.safetensors"
    assert paths.report.name == "anima.export_report.json"


def test_resolve_output_preserves_dots_in_prefix(tmp_path):
    paths = service.resolve_output_paths(tmp_path / "output" / "diffusion_models", "anima.v1")
    assert paths.checkpoint.name == "anima.v1.safetensors"


def test_resolve_output_rejects_non_diffusion_models_root(tmp_path):
    with pytest.raises(service.QuantizationNodeError, match="output/diffusion_models"):
        service.resolve_output_paths(tmp_path / "models" / "diffusion_models_old", "anima")


@pytest.mark.parametrize(
    "prefix",
    ["../escape", "C:/escape", "/escape", "bad:name", "CON", "bad."],
)
def test_resolve_output_rejects_unsafe_prefix(tmp_path, prefix):
    with pytest.raises(service.QuantizationNodeError):
        service.resolve_output_paths(tmp_path / "output" / "diffusion_models", prefix)


def test_export_publishes_checkpoint_and_report_atomically(tmp_path):
    paths = service.resolve_output_paths(tmp_path / "output" / "diffusion_models", "anima")

    def exporter(**kwargs):
        Path(kwargs["output_checkpoint"]).write_bytes(b"checkpoint")
        assert kwargs["family"] == "anima"
        assert kwargs["quantization_preset"] == "quality_keep"
        assert kwargs["require_all_rotated"] is True
        assert kwargs["validate_source_dtype"] is False
        return {
            "status": "model_written",
            "output_checkpoint": str(kwargs["output_checkpoint"]),
            "quantized_tensor_count": 426,
            "rotated_tensor_count": 426,
            "copied_tensor_count": 12,
            "output_bytes": 10,
            "execution_device": "cpu",
            "written_files": [
                {
                    "kind": "int8_tensorwise_inference_checkpoint",
                    "path": str(kwargs["output_checkpoint"]),
                }
            ],
        }

    report, report_path = service.export_anima_int8_convrot(
        state_dict={"x": FakeTensor((1,))},
        paths=paths,
        device="cpu",
        overwrite=False,
        write_report=True,
        hash_output=False,
        exporter=exporter,
    )
    assert paths.checkpoint.read_bytes() == b"checkpoint"
    assert Path(report_path) == paths.report
    saved_report = json.loads(paths.report.read_text(encoding="utf-8"))
    assert saved_report["output_checkpoint"] == str(paths.checkpoint)
    assert report["written_files"][0]["path"] == str(paths.checkpoint)


def test_export_supports_public_examples_448_preset(tmp_path):
    paths = service.resolve_output_paths(
        tmp_path / "output" / "diffusion_models", "anima-public"
    )

    def exporter(**kwargs):
        Path(kwargs["output_checkpoint"]).write_bytes(b"checkpoint")
        assert kwargs["quantization_preset"] == "public_examples"
        return {
            "status": "model_written",
            "quantized_tensor_count": 448,
            "rotated_tensor_count": 448,
            "copied_tensor_count": 237,
            "written_files": [],
        }

    report, _report_path = service.export_anima_int8_convrot(
        state_dict={},
        paths=paths,
        device="cpu",
        overwrite=False,
        write_report=False,
        hash_output=False,
        quantization_preset="public_examples",
        exporter=exporter,
    )

    assert report["quantized_tensor_count"] == 448
    assert paths.checkpoint.read_bytes() == b"checkpoint"


def test_krea2_export_uses_shared_publication_with_krea2_metadata(tmp_path):
    paths = service.resolve_output_paths(
        tmp_path / "output" / "diffusion_models",
        "krea2",
    )
    captured = {}

    def exporter(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_checkpoint"]).write_bytes(b"krea2 checkpoint")
        return {
            "status": "model_written",
            "quantized_tensor_count": 224,
            "rotated_tensor_count": 224,
            "written_files": [
                {
                    "path": str(kwargs["output_checkpoint"]),
                    "kind": "int8_tensorwise_inference_checkpoint",
                }
            ],
        }

    report, report_path = service.export_krea2_int8_convrot(
        state_dict={"first.weight": FakeTensor((6144, 64))},
        paths=paths,
        device="cpu",
        overwrite=False,
        write_report=True,
        hash_output=False,
        exporter=exporter,
    )

    assert paths.checkpoint.read_bytes() == b"krea2 checkpoint"
    assert Path(report_path).is_file()
    assert captured["family"] == "krea2"
    assert captured["metadata"] == {
        "model_family": "krea2",
        "project": "krea2-int8-convrot",
        "source": "comfyui_model_state_dict",
    }
    assert report["quantized_tensor_count"] == 224


def test_minimax_h3_export_uses_reference_preset_and_shared_publication(tmp_path):
    paths = service.resolve_output_paths(
        tmp_path / "output" / "diffusion_models",
        "minimax_h3_int8",
    )
    captured = {}

    def exporter(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_checkpoint"]).write_bytes(b"minimax h3 checkpoint")
        return {
            "status": "model_written",
            "quantized_tensor_count": 200,
            "rotated_tensor_count": 200,
            "copied_tensor_count": 332,
            "written_files": [],
        }

    report, report_path = service.export_minimax_h3_int8_convrot(
        state_dict={
            "blocks.0.attn.qkv_proj.weight": FakeTensor((4, 256))
        },
        paths=paths,
        device="cpu",
        overwrite=False,
        write_report=True,
        hash_output=False,
        exporter=exporter,
    )

    assert paths.checkpoint.read_bytes() == b"minimax h3 checkpoint"
    assert Path(report_path).is_file()
    assert captured["family"] == "minimax_h3"
    assert captured["quantization_preset"] == "strict_reference"
    assert captured["convrot"] is True
    assert captured["convrot_groupsize"] == 256
    assert captured["validate_source_dtype"] is True
    assert captured["metadata"] == {
        "model_family": "minimax_h3",
        "project": "minimax-h3-int8-convrot",
        "source": "comfyui_model_state_dict",
    }
    assert report["quantized_tensor_count"] == 200
    assert report["estimated_peak_memory_bytes"] > 0
    assert report["required_additional_memory_bytes"] > 0
    written_report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert written_report["estimated_peak_memory_bytes"] == report[
        "estimated_peak_memory_bytes"
    ]


def test_minimax_h3_export_rejects_incomplete_keep_report(tmp_path):
    paths = service.resolve_output_paths(
        tmp_path / "output" / "diffusion_models",
        "minimax_h3_incomplete",
    )

    def exporter(**kwargs):
        Path(kwargs["output_checkpoint"]).write_bytes(b"incomplete")
        return {
            "status": "model_written",
            "quantized_tensor_count": 200,
            "rotated_tensor_count": 200,
            "copied_tensor_count": 331,
            "written_files": [],
        }

    with pytest.raises(service.QuantizationNodeError, match="expected_copied=332"):
        service.export_minimax_h3_int8_convrot(
            state_dict={"video_patch_proj.weight": FakeTensor((5376, 96))},
            paths=paths,
            device="cpu",
            overwrite=False,
            write_report=False,
            hash_output=False,
            exporter=exporter,
        )
    assert not paths.checkpoint.exists()


def test_minimax_h3_memory_estimate_and_available_memory_boundary(monkeypatch):
    state = {
        "blocks.0.attn.qkv_proj.weight": FakeTensor(
            (4, 256), "torch.bfloat16", element_bytes=2
        ),
        "token_refiner.final_norm.weight": FakeTensor((4,), element_bytes=4),
    }
    estimate = service.estimate_minimax_h3_memory(state)

    assert estimate.source_bytes == 2064
    assert estimate.output_file_bytes == 1128
    assert estimate.workspace_bytes == 12288
    assert estimate.estimated_peak_bytes == 16810
    assert estimate.required_additional_bytes == 14746

    monkeypatch.setattr(service, "_available_memory_bytes", lambda: 14745)
    with pytest.raises(service.QuantizationNodeError, match="Insufficient.*memory"):
        service._preflight_minimax_h3_memory(state)

    monkeypatch.setattr(service, "_available_memory_bytes", lambda: 14746)
    checked, available = service._preflight_minimax_h3_memory(state)
    assert checked == estimate
    assert available == 14746


def test_minimax_h3_disk_preflight_uses_streaming_output_size(tmp_path, monkeypatch):
    paths = service.resolve_output_paths(
        tmp_path / "output" / "diffusion_models",
        "minimax_h3_streaming_disk",
    )
    state = {
        "blocks.0.attn.qkv_proj.weight": FakeTensor(
            (4, 256), "torch.bfloat16", element_bytes=2
        ),
    }
    available = 1024 + 16 + 72 + service.DISK_HEADROOM_BYTES
    monkeypatch.setattr(
        service.shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(total=available, used=0, free=available),
    )

    def exporter(**kwargs):
        Path(kwargs["output_checkpoint"]).write_bytes(b"checkpoint")
        return {
            "status": "model_written",
            "quantized_tensor_count": 200,
            "rotated_tensor_count": 200,
            "copied_tensor_count": 332,
            "written_files": [],
        }

    service.export_minimax_h3_int8_convrot(
        state_dict=state,
        paths=paths,
        device="cpu",
        overwrite=False,
        write_report=False,
        hash_output=False,
        exporter=exporter,
    )

    assert paths.checkpoint.is_file()


def test_minimax_h3_service_export_rejects_fp16_selected_tensor(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    export_module = sys.modules[
        service.export_minimax_h3_int8_convrot_from_state_dict.__module__
    ]
    spec = export_module.TensorSpec(
        name="blocks.0.attn.qkv_proj.weight",
        shape=(4, 256),
    )
    monkeypatch.setattr(
        export_module,
        "validate_minimax_h3_state_dict",
        lambda *_args, **_kwargs: (spec,),
    )
    monkeypatch.setattr(service, "_available_memory_bytes", lambda: 1_000_000)
    paths = service.resolve_output_paths(
        tmp_path / "output" / "diffusion_models",
        "minimax_h3_fp16",
    )

    with pytest.raises(service.QuantizationNodeError, match="bfloat16 or float32"):
        service.export_minimax_h3_int8_convrot(
            state_dict={
                spec.name: torch.zeros(spec.shape, dtype=torch.float16),
            },
            paths=paths,
            device="cpu",
            overwrite=False,
            write_report=False,
            hash_output=False,
        )
    assert not paths.checkpoint.exists()


def test_export_rejects_unknown_quantization_preset(tmp_path):
    paths = service.resolve_output_paths(tmp_path / "output" / "diffusion_models", "invalid")
    with pytest.raises(service.QuantizationNodeError, match="Unsupported quantization_preset"):
        service.export_anima_int8_convrot(
            state_dict={},
            paths=paths,
            device="cpu",
            overwrite=False,
            write_report=False,
            hash_output=False,
            quantization_preset="unknown",
        )
    assert not paths.checkpoint.parent.exists()


def test_export_rejects_insufficient_free_disk_before_quantization(tmp_path, monkeypatch):
    paths = service.resolve_output_paths(tmp_path / "output" / "diffusion_models", "too-large")
    monkeypatch.setattr(
        service.shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(total=1024, used=1023, free=1),
    )

    with pytest.raises(service.QuantizationNodeError, match="free disk space"):
        service.export_anima_int8_convrot(
            state_dict={"x": FakeTensor((1024, 1024))},
            paths=paths,
            device="cpu",
            overwrite=False,
            write_report=False,
            hash_output=False,
            exporter=lambda **_kwargs: pytest.fail("exporter must not run"),
        )


def test_export_wraps_output_directory_preflight_errors(tmp_path, monkeypatch):
    paths = service.resolve_output_paths(
        tmp_path / "output" / "diffusion_models", "unavailable"
    )

    def fail_preflight(*_args, **_kwargs):
        raise OSError("disk status unavailable")

    monkeypatch.setattr(service, "_ensure_output_capacity", fail_preflight)

    with pytest.raises(
        service.QuantizationNodeError,
        match="prepare output directory.*disk status unavailable",
    ):
        service.export_anima_int8_convrot(
            state_dict={},
            paths=paths,
            device="cpu",
            overwrite=False,
            write_report=False,
            hash_output=False,
            exporter=lambda **_kwargs: pytest.fail("exporter must not run"),
        )


def test_export_rejects_unexpected_quantized_count(tmp_path):
    paths = service.resolve_output_paths(tmp_path / "output" / "diffusion_models", "anima")

    def exporter(**kwargs):
        Path(kwargs["output_checkpoint"]).write_bytes(b"partial")
        return {"quantized_tensor_count": 425, "rotated_tensor_count": 425}

    with pytest.raises(service.QuantizationNodeError, match="Unexpected quantization result"):
        service.export_anima_int8_convrot(
            state_dict={},
            paths=paths,
            device="cpu",
            overwrite=False,
            write_report=False,
            hash_output=False,
            exporter=exporter,
        )
    assert not paths.checkpoint.exists()
    assert not list(paths.checkpoint.parent.glob("*.tmp.safetensors"))


def test_export_translates_internal_quantization_errors(tmp_path):
    paths = service.resolve_output_paths(tmp_path / "output" / "diffusion_models", "anima")

    def exporter(**_kwargs):
        raise service.QuantizationExportError("invalid selected tensor")

    with pytest.raises(service.QuantizationNodeError, match="invalid selected tensor"):
        service.export_anima_int8_convrot(
            state_dict={},
            paths=paths,
            device="cpu",
            overwrite=False,
            write_report=False,
            hash_output=False,
            exporter=exporter,
        )


def test_publish_without_overwrite_never_replaces_existing_file(tmp_path):
    source = tmp_path / "new.tmp"
    destination = tmp_path / "model.safetensors"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")

    with pytest.raises(service.QuantizationNodeError, match="overwrite was disabled"):
        service._publish_without_overwrite(source, destination)

    assert destination.read_bytes() == b"old"
    assert source.read_bytes() == b"new"


def test_posix_publish_uses_atomic_hard_link(tmp_path, monkeypatch):
    source = tmp_path / "new.tmp"
    destination = tmp_path / "model.safetensors"
    source.write_bytes(b"new")
    calls = []
    original_link = service.os.link

    def tracked_link(link_source, link_destination):
        calls.append((link_source, link_destination))
        return original_link(link_source, link_destination)

    monkeypatch.setattr(service.os, "name", "posix")
    monkeypatch.setattr(service.os, "link", tracked_link)
    service._publish_without_overwrite(source, destination)

    assert calls == [(source, destination)]
    assert destination.read_bytes() == b"new"
    assert source.exists()  # the exporter's finally block removes the temporary link


def test_export_rolls_back_checkpoint_and_report_when_report_publish_fails(tmp_path, monkeypatch):
    paths = service.resolve_output_paths(tmp_path / "output" / "diffusion_models", "anima")
    paths.checkpoint.parent.mkdir(parents=True)
    paths.checkpoint.write_bytes(b"old checkpoint")
    paths.report.write_text('{"old": true}\n', encoding="utf-8")

    def exporter(**kwargs):
        Path(kwargs["output_checkpoint"]).write_bytes(b"new checkpoint")
        return {
            "quantized_tensor_count": 426,
            "rotated_tensor_count": 426,
            "written_files": [],
        }

    original_replace = service.os.replace

    def fail_new_report(source, destination):
        source_path = Path(source)
        if Path(destination) == paths.report and source_path.name.endswith(".tmp"):
            raise OSError("simulated report publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(service.os, "replace", fail_new_report)
    with pytest.raises(service.QuantizationNodeError, match="simulated report publish failure"):
        service.export_anima_int8_convrot(
            state_dict={},
            paths=paths,
            device="cpu",
            overwrite=True,
            write_report=True,
            hash_output=False,
            exporter=exporter,
        )

    assert paths.checkpoint.read_bytes() == b"old checkpoint"
    assert json.loads(paths.report.read_text(encoding="utf-8")) == {"old": True}
    assert not list(paths.checkpoint.parent.glob("*.bak"))


def test_export_does_not_replace_report_that_appears_during_no_overwrite_publish(
    tmp_path, monkeypatch
):
    paths = service.resolve_output_paths(
        tmp_path / "output" / "diffusion_models", "anima-race"
    )

    def exporter(**kwargs):
        Path(kwargs["output_checkpoint"]).write_bytes(b"checkpoint")
        return {
            "quantized_tensor_count": 426,
            "rotated_tensor_count": 426,
            "written_files": [],
        }

    original_publish = service._publish_without_overwrite

    def create_competing_report(source, destination):
        if Path(destination) == paths.report:
            paths.report.write_text('{"owner": "other"}\n', encoding="utf-8")
        return original_publish(source, destination)

    monkeypatch.setattr(service, "_publish_without_overwrite", create_competing_report)
    with pytest.raises(service.QuantizationNodeError, match="overwrite was disabled"):
        service.export_anima_int8_convrot(
            state_dict={},
            paths=paths,
            device="cpu",
            overwrite=False,
            write_report=True,
            hash_output=False,
            exporter=exporter,
        )

    assert not paths.checkpoint.exists()
    assert json.loads(paths.report.read_text(encoding="utf-8")) == {"owner": "other"}


def test_export_wraps_unexpected_exporter_errors_with_output_context(tmp_path):
    paths = service.resolve_output_paths(tmp_path / "output" / "diffusion_models", "failure")

    def exporter(**_kwargs):
        raise OSError("simulated writer failure")

    with pytest.raises(
        service.QuantizationNodeError,
        match=r"writing or publishing .*failure\.safetensors.*simulated writer failure",
    ):
        service.export_anima_int8_convrot(
            state_dict={},
            paths=paths,
            device="cpu",
            overwrite=False,
            write_report=False,
            hash_output=False,
            exporter=exporter,
        )


def test_default_exporter_is_the_internal_relative_api():
    exporter = service.export_anima_int8_convrot_from_state_dict
    assert exporter.__name__ == "export_anima_int8_convrot_from_state_dict"
    assert exporter.__module__.startswith(f"{PACKAGE_NAME}.quantization")


def test_service_wires_the_internal_exporter_through_publication(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    export_module = sys.modules[service.export_anima_int8_convrot_from_state_dict.__module__]
    anima_module = sys.modules[f"{PACKAGE_NAME}.quantization.anima"]
    name = "net.blocks.2.self_attn.q_proj.weight"
    spec = anima_module.AnimaTensorSpec(name=name, shape=(4, 256))
    state = {
        name: torch.randn(4, 256, dtype=torch.bfloat16),
        "net.final_layer.linear.bias": torch.randn(4),
    }
    paths = service.resolve_output_paths(tmp_path / "output" / "diffusion_models", "internal")

    monkeypatch.setattr(
        export_module,
        "validate_anima_state_dict",
        lambda *_args, **_kwargs: (spec,),
    )
    monkeypatch.setattr(service, "expected_quantized_tensors", lambda _preset: 1)

    report, report_path = service.export_anima_int8_convrot(
        state_dict=state,
        paths=paths,
        device="cpu",
        overwrite=False,
        write_report=True,
        hash_output=True,
    )

    tensors = safetensors_torch.load_file(str(paths.checkpoint))
    assert tensors[name].dtype == torch.int8
    assert report["quantized_tensor_count"] == report["rotated_tensor_count"] == 1
    assert Path(report_path) == paths.report
