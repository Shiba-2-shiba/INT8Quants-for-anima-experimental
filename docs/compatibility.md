# Compatibility

## Supported contract

| Component | Supported |
| --- | --- |
| Models | Anima 2B; Krea2 Raw; Krea2 Turbo; MiniMax H3 reference architecture |
| Krea2 architecture | Native ComfyUI 28-block, width-6144 open-weight model |
| MiniMax H3 architecture | Native ComfyUI 50-block, two-token-refiner, width-5376 curve form |
| Source dtype | Anima/Krea2: BF16 or FP16; MiniMax H3: BF16 or FP32 in-memory weights |
| Output | Stock ComfyUI `int8_tensorwise` safetensors |
| Output directory | Configured ComfyUI `output/diffusion_models/` |
| ConvRot | Regular Hadamard, group size 256 |
| Anima presets | `quality_keep` (426), `public_examples` (448) |
| Krea2 presets | `quality_keep` (224) |
| MiniMax H3 presets | `strict_reference` (200) |
| MiniMax H3 RAM | 96 GB recommended for the reference BF16 layout; 160 GB for a fully FP32 source (provisional) |
| Python | 3.10 or newer |
| ComfyUI API | Backend V3 `v0_0_2` |
| Minimum ComfyUI | 0.31.0 (MiniMax H3 plus corrected AV sampling settings) |

FLUX.1 Krea [dev], hosted Krea API models, Anima 14B, MiniMax H3 time-embedder
or non-50/2 variants, Diffusers folders as a direct node input, and
already-quantized `MODEL` inputs are not supported.

## Validation environment

| Component | Version |
| --- | --- |
| ComfyUI checkout | `v0.32.0-6-g725e6ec6`, commit `725e6ec60621c6f001af04769173e7dbb3c53541` |
| comfy-kitchen used for tests | 0.2.8 |
| Python used for unit tests | 3.10.11 |
| PyTorch used for unit tests | 2.10.0+cpu |
| safetensors used for unit tests | 0.7.0 |

Artifact generation, stock loader detection, optimized GPU runtime, and image
quality are separate compatibility levels. CUDA and ROCm runtime support must
only be claimed for environments recorded in release validation evidence.

Krea2 checkpoint distribution is also governed by the Krea 2 Community
License; code compatibility does not grant additional model-weight rights.
