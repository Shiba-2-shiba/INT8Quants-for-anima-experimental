# Anima & Krea2 INT8 ConvRot for ComfyUI

Self-contained ComfyUI Backend V3 output nodes that export unpatched Anima 2B
and Krea2 Raw/Turbo `MODEL` inputs as stock ComfyUI `int8_tensorwise`
checkpoints with Regular ConvRot group size 256.

The custom node contains only the two model contracts and the INT8 export code
they share. It does not require a `third_party` checkout, Git submodule,
external `comfy-quants` or `convert_to_quant` package, or runtime `sys.path`
modification.

## Nodes

| Node | Quantized matrices | High-precision boundary |
| --- | ---: | --- |
| Anima INT8 ConvRot Save | 426 or 448 | Preset-dependent; embedders, final layer, LLM adapter, and norms remain unchanged |
| Krea2 INT8 ConvRot Save | 224 | `first`, `tmlp`, `txtmlp`, `txtfusion`, `last`, `tproj`, modulation, and norms remain unchanged |

Krea2 is a separate node so the existing
`ComfyQuantsAnimaInt8ConvRotSave` node ID and saved workflows remain compatible.
No frontend JavaScript or tab patching is needed.

## Requirements

- ComfyUI 0.27.0 or newer with Backend V3 and native INT8 ConvRot support
- Python 3.10 or newer
- An original floating-point Anima 2B, Krea2 Raw, or Krea2 Turbo diffusion model
- `safetensors` (installed from this repository's `requirements.txt`)

ComfyUI supplies PyTorch and the runtime INT8 loader. The current local
validation checkout is ComfyUI 0.28.0 with `comfy-kitchen==0.2.20`.

Krea2 here means the local open-weight Krea2 Raw/Turbo model. It is not
FLUX.1 Krea [dev], and it is not the hosted Krea API's Medium/Large model
selector.

## Install

Clone the repository below `ComfyUI/custom_nodes`, then install its direct
dependency with the same Python interpreter that starts ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Shiba-2-shiba/INT8Quants-for-anima-experimental.git
python -m pip install -r INT8Quants-for-anima-experimental/requirements.txt
```

Restart ComfyUI after installation. No recursive clone or sibling repository is
required.

## Workflow

```text
Load Diffusion Model -> Anima INT8 ConvRot Save
Load Diffusion Model -> Krea2 INT8 ConvRot Save
```

The checkpoint and optional JSON report are written together below ComfyUI's
configured `output/diffusion_models/` directory. This directory is registered
as a `diffusion_models` search path when the custom node loads, so the exported
checkpoint can be reloaded with the stock `Load Diffusion Model` node after
refreshing the model list.

API workflow examples:

- [`examples/workflows/anima_int8_convrot.json`](examples/workflows/anima_int8_convrot.json)
- [`examples/workflows/krea2_int8_convrot.json`](examples/workflows/krea2_int8_convrot.json)

Each node returns the checkpoint path, optional report path, and a compact JSON
summary containing counts, size, device, and optional hash.

## Presets

Anima:

- `quality_keep` (default): quantizes 426 weights. Block 0 and block 1 AdaLN
  modulation weights stay in source precision.
- `public_examples`: quantizes all 448 eligible transformer weights.

Krea2:

- `quality_keep` (default): quantizes the eight eligible linear weights in each
  of 28 transformer blocks, for 224 INT8 matrices. All selected input dimensions
  are divisible by the ConvRot group size 256.

The Krea2 selection follows the high-precision boundary used by the local
`convert_to_quant --krea2` implementation, but uses explicit native Krea2 keys
and exact shapes instead of short substring filters. See
[`docs/krea2.md`](docs/krea2.md) for model and format details.

## Safety checks

Both nodes reject patched ModelPatchers, already-quantized inputs, invalid
model signatures, missing blocks, invalid tensor shapes/dtypes, non-finite
selected weights, unsafe output paths, and insufficient free disk space.
Checkpoint/report publication uses temporary files and rollback.

`cuda` is also the PyTorch device name on ROCm builds. Successful offline
export does not by itself guarantee that an optimized runtime INT8 kernel is
available or stable on every GPU.

## Development

Run host-independent tests:

```bash
python -m pytest -q
```

Run Backend V3 and stock Krea2 detection integration against a ComfyUI checkout:

```powershell
$env:PYTHONPATH='C:\path\to\ComfyUI'
python -m pytest -q tests/test_schema.py
```

See [`docs/development.md`](docs/development.md) and
[`docs/validation.md`](docs/validation.md) for the complete verification
matrix.

## License and provenance

This project's code is licensed under GPL-3.0-only. The self-contained
quantization core contains code derived from `Comfy-Org/comfy-quants`; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Krea2 model weights use the separate Krea 2 Community License. Quantized
checkpoints are modified weight artifacts and remain subject to the applicable
model license, notice, naming, use-policy, and commercial-use conditions. Review
the official Krea license before distributing an exported checkpoint.
