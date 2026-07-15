# Krea2 INT8 support

## Model scope

This project supports the native ComfyUI in-memory representation of the
open-weight Krea2 Raw and Krea2 Turbo diffusion models. Both variants use the
same supported diffusion architecture:

- 12B dense single-stream DiT
- 28 transformer blocks
- hidden width 6144
- 48 query heads and 12 key/value heads, head dimension 128
- 16 latent channels with patch size 2
- Qwen3-VL-4B text features aggregated from 12 layers at width 2560

The stock ComfyUI detector identifies this family through
`txtfusion.projector.weight`. `first.weight`, block attention shapes, and the
text-fusion norm shape determine the remaining configuration.

Official sources:

- <https://www.krea.ai/krea-2-open-source>
- <https://github.com/krea-ai/krea-2>
- <https://huggingface.co/krea/Krea-2-Raw>
- <https://huggingface.co/krea/Krea-2-Turbo>
- <https://github.com/Comfy-Org/ComfyUI/pull/14589>

FLUX.1 Krea [dev] is a different model family. Krea's hosted Medium, Medium
Turbo, and Large API choices are also outside this local checkpoint exporter.

## Quantization contract

Only these native Krea2 block matrices are selected:

```text
blocks.0..27.attn.{wq,wk,wv,gate,wo}.weight
blocks.0..27.mlp.{gate,up,down}.weight
```

That produces 28 × 8 = 224 INT8 matrices. Every selected input dimension is
divisible by 256, so all selected weights use Regular ConvRot rather than a
mixed rotated/non-rotated fallback.

The exporter keeps `first`, `tmlp`, `txtmlp`, `txtfusion`, `last`, `tproj`,
normalization, modulation, bias, and all non-matrix tensors in their source
dtype. This matches the intent of the local `convert_to_quant --krea2` filter
while avoiding its deliberately short substring patterns.

Each selected layer is written as:

```text
<layer>.weight       torch.int8 [out_features, in_features]
<layer>.weight_scale torch.float32 [out_features, 1]
<layer>.comfy_quant  uint8 JSON marker
```

The marker contract is:

```json
{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
```

The node consumes a ComfyUI `MODEL`, not a Diffusers directory. Source file
namespace conversion remains the responsibility of the stock model loader;
the node validates and exports the resulting native Krea2 namespace.

## Existing official INT8 files

Comfy-Org already publishes Raw and Turbo BF16, FP8, and INT8 ConvRot diffusion
models at <https://huggingface.co/Comfy-Org/Krea-2/tree/main/diffusion_models>.
This node is primarily useful for locally modified checkpoints, merges,
fine-tunes, or reproducible offline conversion inside ComfyUI.

## Runtime and licensing

ComfyUI 0.27.0 added native INT8 ConvRot execution. Loading an exported model
and actually accelerating it are separate checks: GPU/backend support is owned
by ComfyUI and `comfy-kitchen`.

Krea2 code is published under Apache-2.0, but the weights use the Krea 2
Community License. Exported quantized weights remain subject to the official
license and use policy:

- <https://www.krea.ai/krea-2-licensing>
- <https://www.krea.ai/krea-2-use-policy>
