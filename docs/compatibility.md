# Compatibility

## Supported contract

| Component | Supported |
| --- | --- |
| Models | Anima 2B; Krea2 Raw; Krea2 Turbo |
| Krea2 architecture | Native ComfyUI 28-block, width-6144 open-weight model |
| Source dtype | BF16 or FP16 in-memory weights |
| Output | Stock ComfyUI `int8_tensorwise` safetensors |
| Output directory | Configured ComfyUI `output/diffusion_models/` |
| ConvRot | Regular Hadamard, group size 256 |
| Anima presets | `quality_keep` (426), `public_examples` (448) |
| Krea2 presets | `quality_keep` (224) |
| Python | 3.10 or newer |
| ComfyUI API | Backend V3 `v0_0_2` |

FLUX.1 Krea [dev], hosted Krea API models, Anima 14B, Diffusers folders as a
direct node input, and already-quantized `MODEL` inputs are not supported.

## Validation environment

| Component | Version |
| --- | --- |
| ComfyUI checkout | 0.28.0, commit `700821e1364eaab0e8f21c538a2131719fec57bf` |
| ComfyUI pinned comfy-kitchen | 0.2.20 |
| Python used for unit tests | 3.10.11 |
| PyTorch used for unit tests | 2.10.0+cpu |
| safetensors used for unit tests | 0.7.0 |

Artifact generation, stock loader detection, optimized GPU runtime, and image
quality are separate compatibility levels. CUDA and ROCm runtime support must
only be claimed for environments recorded in release validation evidence.

Krea2 checkpoint distribution is also governed by the Krea 2 Community
License; code compatibility does not grant additional model-weight rights.
