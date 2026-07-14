# Compatibility

## Supported contract

| Component | Supported |
| --- | --- |
| Model | Anima 2B diffusion model with 28 transformer blocks |
| Source dtype | BF16 or FP16 in-memory weights |
| Output | Stock ComfyUI `int8_tensorwise` safetensors |
| Output directory | Configured ComfyUI `output/diffusion_models/` |
| ConvRot | Regular Hadamard, group size 256 |
| Presets | `quality_keep` (426), `public_examples` (448) |
| Python | 3.10 or newer |
| ComfyUI API | Backend V3 `v0_0_2` |

## Validation environment

| Component | Version |
| --- | --- |
| ComfyUI checkout | 0.27.0, commit `917faef771a2fd2f14f44af94f17da3d0b2803a3` |
| ComfyUI pinned comfy-kitchen | 0.2.18 |
| Python used for unit tests | 3.10.11 |
| PyTorch used for unit tests | 2.10.0+cpu |
| safetensors used for unit tests | 0.7.0 |

Artifact generation, stock loader compatibility, and optimized GPU runtime are
separate compatibility levels. CUDA and ROCm runtime support must only be
claimed for environments recorded in release validation evidence.
