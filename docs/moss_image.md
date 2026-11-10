# Moss Image INT8 ConvRot

## Usage

Restart ComfyUI, then connect:

```text
Load Diffusion Model (floating-point Moss Image, weight_dtype=default)
  -> Moss Image INT8 ConvRot Save
```

Use an unpatched 42-layer, width-3072 source model. The preset is `quality_keep`;
`device=cpu` is the default. Output is
`output/diffusion_models/moss_image_int8_convrot.safetensors`, with an optional
JSON report beside it. The filename prefix can be changed. Existing files are
protected unless `overwrite` is enabled. Reload the result using the stock
Load Diffusion Model node after refreshing its model list.

The [API workflow example](../examples/workflows/moss_image_int8_convrot.json)
uses `moss_image_base_bf16.safetensors` as an example source filename. Replace
that filename with your installed floating-point model. Already-INT8/FP8
checkpoints and MODEL inputs with LoRA or other runtime patches are rejected.

## Model and format contract

The reference is Comfy-Org's
[`moss_image_base_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_base_int8_convrot.safetensors).
Its header and small configuration tensors were inspected; model weights are
not included in the tests. The checked-in
[`moss_image_reference.json`](../tests/fixtures/moss_image_reference.json)
records the reference tensor names, shapes, dtypes, and marker contents.

For every `layers.0` through `layers.41`, the fused layout quantizes:

| Weight suffix | Shape |
| --- | --- |
| `proj.qkv.weight` | 9216 × 3072 |
| `proj.o.weight` | 3072 × 3072 |
| `proj.ffn_up_gate.weight` | 16384 × 3072 |
| `proj.ffn_down.weight` | 3072 × 8192 |

These are 168 INT8 matrices, each accompanied by an FP32
`weight_scale` shaped `(out_features, 1)` and a uint8 JSON `comfy_quant` marker:

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

ComfyUI also supports separate `proj.q`, `proj.k`, and `proj.v` weights.
The exporter detects that layout, validates all three 3072 × 3072 matrices
per layer, and preserves their names, yielding 252 INT8 matrices. Mixed
fused/separate layouts are rejected.

Text projection and both text-refiner blocks, timestep and shared modulation
layers, video input/output projections, all biases, and all norms retain their
source dtype and values. All 378 learned source tensors (462 for separate QKV)
are checked for exact names, shapes, and BF16/FP16/FP32 dtype before export.
Streaming serialization reuses the existing 32 MiB chunk writer, so the output
checkpoint is not accumulated in memory. Loading the floating-point source
still requires sufficient memory.

## Differences from the reference file

The official reference also contains `rope.bands` and 42
`comfy_attention.config` tensors selecting `comfy_kitchen_int8` attention.
These are separate from weight ConvRot quantization:

- Current ComfyUI removes `rope.bands` on load and computes RoPE itself.
  Export accepts and preserves this buffer when present, but does not invent
  one when the loaded MODEL lacks it.
- Existing attention settings are preserved. The exporter does not add an
  INT8 attention preference to a source model that has none.

The output follows the official weight-selection and serialization format;
it is not claimed to be byte-identical to the official file or to use the same
attention backend. Native behavior is defined by ComfyUI's
[Moss implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/moss_image/model.py)
and [model handling](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/supported_models.py).

## Validation scope

Automated checks cover reference selection/shapes, both QKV layouts, complete
source validation, rejection of quantized/patched inputs, small-tensor streaming
roundtrip, scale/marker format, non-mutation, count validation, and publication.
Source-level model/INT8 detection was checked against the local ComfyUI
`v0.43.0-32-gab08f7b2c` checkout. An optional integration test checks Backend V3
registration and execution when a ComfyUI environment is supplied.

Full-model export, stock reload, GPU inference, and image-quality comparison
are left to the user's real-machine validation. Expected report counts are
`quantized_tensor_count == rotated_tensor_count == 168` for fused QKV, or
`252` for separate QKV. Use a current ComfyUI with native Moss support.
