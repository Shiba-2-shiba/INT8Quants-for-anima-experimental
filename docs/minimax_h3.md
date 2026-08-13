# MiniMax H3 INT8 support

## Supported model contract

The MiniMax H3 node intentionally supports one architecture: the native
ComfyUI 50-block, two-token-refiner, width-5376 curve form represented by the
official `minimax_h3_fl2va_pruned_int8_convrot.safetensors` reference. It
rejects time-embedder form, different block/refiner counts, or incompatible
tensor shapes instead of guessing checkpoint metadata.

The node consumes the unpatched, floating-point `MODEL` produced by stock
`Load Diffusion Model`. It does not load a checkpoint path itself and does not
replace parameters in the running model.

The reference header establishes these detector values:

| Property | Value |
| --- | ---: |
| Main DiT blocks | 50 |
| Token-refiner blocks | 2 |
| Hidden width | 5376 |
| Attention head dimension | 128 |
| Query heads | 56 |
| FFN width | 14336 |
| Text width | 5120 |
| Video/audio latent dimensions | 24 / 32 |
| `adaln_t_table` | `[1025, 8]` |
| `rope.inv_freq` | `[16]` |

The official reference has no `config.transformer` metadata override. The
exporter therefore verifies the effective input model config against this
contract and lets stock ComfyUI reconstruct it from tensors after reload.

## Quantization and keep boundary

`strict_reference` selects exactly four weights in every main block:

```text
blocks.0..49.attn.qkv_proj.weight  [21504, 5376]
blocks.0..49.attn.out_proj.weight  [5376, 7168]
blocks.0..49.mlp.fc1.weight        [28672, 5376]
blocks.0..49.mlp.fc2.weight        [5376, 14336]
```

This produces exactly 200 INT8 weights. Each input dimension is divisible by
the fixed ConvRot group size 256. Every selected layer gets an FP32
`weight_scale` and this stock ComfyUI marker:

```json
{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
```

Everything else remains in source precision. In particular, the complete
token refiner, condition projection, video/audio input projections, final
layer, AdaLN projections, all biases and normalization tensors,
`adaln_t_table`, and RoPE buffers are kept. The reference contains 274 kept
weights in addition to the 200 quantized weights.

The validator fixes the entire 532-tensor floating-point source manifest, not
only the selected matrices. Export fails if any of the 332 keep tensors is
missing, has the wrong shape, or if an unknown extra tensor is present. The
publication service also requires the completed writer report to contain
exactly 200 quantized and 332 copied tensors.

## Resource and validation status

Header-only analysis of the 20,970,379,616-byte reference found 932 tensors,
200 one-to-one INT8/scale/marker triplets, no orphan quantization tensors, and
no data-offset gaps or overlaps. The selected INT8 weights occupy
19,267,584,000 bytes; scales occupy 12,185,600 bytes; kept tensors occupy
1,690,500,192 bytes.

For the official mixed-precision/BF16 source layout, the floating-point state
dict is about 40.23 GB (37.46 GiB). MiniMax H3 uses a dedicated streaming
writer: selected matrices are processed in source chunks of at most 32 MiB,
INT8 bytes are written immediately, and only the small FP32 row scales remain
live until their adjacent output entry is written. Kept tensors are copied
directly instead of being accumulated in a second in-memory state dict.

The RAM preflight therefore excludes the 20.97 GB output file from live memory.
It budgets six 32 MiB work buffers plus 20% headroom, or about 230.4 MiB of
additional RAM for the reference model. The estimated output size, streaming
chunk size, peak memory, required additional memory, and preflight availability
are recorded in the report and node summary. Loading the source model remains
the dominant requirement; 64 GB system RAM for the BF16 layout and 96 GB for a
fully FP32 source remain provisional recommendations until full-model peak RSS
has been measured.

Release validation still requires a high-memory machine to record peak RSS,
export/reload success, fixed-input forward error, and a short fixed-seed sample.
When the matching floating-point source is available, generated INT8 tensors
and scales should be compared exactly with the official reference on CPU. If
an exact comparison is not possible, generated relative L2 error must be no
more than 5% worse than the reference and cosine similarity must not fall more
than `1e-4` below it.
