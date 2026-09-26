# Anima 2B and 2.9B

`Anima INT8 ConvRot Save` accepts an unpatched floating-point Anima `MODEL`
from `Load Diffusion Model`. It detects contiguous main-transformer blocks:
28 for 2B, or 40 for 2.9B. Both have model width 2048, context width 1024,
MLP width 8192, and AdaLN low-rank width 256. Other depths, missing blocks,
and incorrect selected matrix shapes are rejected.

ComfyUI 0.33.1 or newer is required. Older loaders may instantiate only 28
blocks for a 40-block checkpoint. The exporter can inspect only the loaded
model, so a newer exporter cannot recover blocks discarded by an old loader.

## Presets

| Preset | 2B | 2.9B | High-precision boundary |
| --- | ---: | ---: | --- |
| `quality_keep` (default) | 426 | 618 | All of block 0 and the six block 1 AdaLN matrices are excluded |
| `public_examples` | 448 | 640 | All 16 eligible matrices in every transformer block are selected |

Both presets preserve embedders, the final layer, LLM adapter, norms, and
other unselected tensors. The `quality_keep` boundary is inherited from 2B;
its image-quality advantage for 2.9B has not been measured. The existing node
ID, preset names, default preset, output fields, and 2B selections are unchanged.

## Inspected reference

Reference: [`Anima-2.9B-preview-v1_int8_convrot.safetensors`](https://huggingface.co/Gazingstars123/Anima-2.9B/blob/main/Anima-2.9B-preview-v1_int8_convrot.safetensors).
The user-supplied local copy was inspected on 2026-09-26. Its file size is
3,082,486,758 bytes, with a 254,344-byte JSON header and 2,208 tensors:

- 640 INT8 matrices: the same 16 eligible names/shapes in each of 40 blocks.
- 640 FP32 `weight_scale` tensors, each shaped `[out_features, 1]`.
- 640 UINT8 `comfy_quant` markers containing exactly
  `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`.
- 288 BF16 tensors outside the quantization selection.

`public_examples` matches this selection and format, including the extra
blocks 28 through 39. This does not claim byte-identical quantized weights
from a new export: the original floating-point checkpoint and numeric
comparison are needed to establish that.

The supplied file is already INT8 and must not be fed back into the exporter.
Use the original BF16/FP16 checkpoint for conversion; load the INT8 file
directly for inference with a compatible ComfyUI runtime.

## Repeatable checks

Run all host-independent tests and the optional real-reference gate:

```powershell
$env:ANIMA_29B_REFERENCE='C:\path\to\Anima-2.9B-preview-v1_int8_convrot.safetensors'
python -B -m pytest -q
```

The reference test reads only the header and small marker tensors. No weights
are copied, converted, or modified. Without the environment variable, this
test is skipped; synthetic contract tests still run on every platform.

For Backend V3 execution and stock depth detection, also set `PYTHONPATH` to
a ComfyUI 0.33.1+ checkout and run `tests/test_schema.py`. Full real-model
export/reload and fixed-seed image comparisons are separate validation gates.
