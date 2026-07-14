# Anima INT8 ConvRot for ComfyUI

Self-contained ComfyUI Backend V3 output node that exports an unpatched Anima
2B `MODEL` as a stock ComfyUI `int8_tensorwise` checkpoint with Regular
ConvRot group size 256.

The custom node contains only the Anima contract and INT8 export code it uses.
It does not require a `third_party` checkout, Git submodule, external
`comfy-quants` package, or runtime `sys.path` modification.

## Requirements

- ComfyUI 0.27.0 or newer with Backend V3 support
- Python 3.10 or newer
- An original floating-point Anima 2B diffusion model
- `safetensors` (installed from this repository's `requirements.txt`)

ComfyUI supplies PyTorch and the runtime INT8 loader. The ComfyUI version used
for validation pins `comfy-kitchen==0.2.18`.

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
```

The checkpoint and JSON report are written together below ComfyUI's configured
`output/diffusion_models/` directory. This directory is registered as a
`diffusion_models` search path when the custom node loads, so the checkpoint
can be reloaded with the stock `Load Diffusion Model` node after refreshing the
model list. A generic API workflow is available at
[`examples/workflows/anima_int8_convrot.json`](examples/workflows/anima_int8_convrot.json).

The node returns:

- `checkpoint path`: absolute path to the exported `.safetensors` file.
- `report path`: optional `.export_report.json`, or an empty string.
- `summary`: compact JSON containing counts, size, device, and optional hash.

## Presets

- `quality_keep` (default): quantizes 426 weights. Block 0 and block 1 AdaLN
  modulation weights stay in source precision.
- `public_examples`: quantizes all 448 eligible transformer weights.

Both presets keep normalization weights, embedders, the final layer, and the
LLM adapter in source precision. The selected tensor names, shapes, marker
bytes, rowwise INT8 math, and ConvRot transform are regression-tested.

## Safety checks

The node rejects patched ModelPatchers, already-quantized inputs, Anima 14B,
models without the Anima LLM-adapter signature, invalid tensor shapes/dtypes,
non-finite selected weights, unsafe output paths, and insufficient free disk
space. Checkpoint/report publication uses temporary files and rollback.

`cuda` is also the PyTorch device name on ROCm builds. Successful offline
export does not by itself guarantee that an optimized runtime INT8 kernel is
available or stable on every GPU.

## Development

Run host-independent tests:

```bash
python -m pytest -q
```

Run Backend V3 integration against a ComfyUI checkout:

```powershell
$env:PYTHONPATH='C:\path\to\ComfyUI'
python -m pytest -q tests/test_schema.py
```

See [`docs/development.md`](docs/development.md) and
[`docs/validation.md`](docs/validation.md) for the complete verification
matrix.

## License and provenance

This project is licensed under GPL-3.0-only. The self-contained quantization
core contains code derived from `Comfy-Org/comfy-quants`; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the pinned upstream
commit and migration details.
