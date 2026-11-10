# Architecture

The repository is one self-contained ComfyUI custom-node package with four
model-specific output nodes and one shared INT8 writer.

```text
ComfyUI Backend V3
  __init__.py -> nodes.py
                     |
                     v
                  service.py
                     |
                     v
  quantization/{contracts,anima,krea2,minimax_h3,moss_image,convrot,export}.py
```

## ComfyUI adapter

`nodes.py` registers separate Anima, Krea2, MiniMax H3, and Moss Image schemas. A small immutable node
profile shares the identical controls, progress handling, and summary shaping
without merging the public node IDs or model-specific descriptions. Existing
Anima workflows continue to reference `ComfyQuantsAnimaInt8ConvRotSave`.

There is no custom frontend extension. All nodes use only Backend V3 metadata,
which survives the stock `/object_info` to Nodes 2.0 transformation.

## Application service

`service.py` owns ModelPatcher mutation checks, already-quantized rejection,
disk capacity checks, output path validation, and the checkpoint/report
publication transaction.

The extraction boundary is deliberately model-specific:

- Anima converts the in-memory `diffusion_model.*` keys to the checkpoint's
  `net.*` namespace.
- Krea2 strips only the in-memory `diffusion_model.` wrapper and retains the
  native `first`, `blocks`, and `txtfusion` namespace expected by stock ComfyUI.
- MiniMax H3 also strips only `diffusion_model.` and retains its native
  `video_patch_proj`, `audio_patch_proj`, `token_refiner`, and `blocks`
  namespace. Its effective ComfyUI model config is checked against the exact
  reference architecture before export.

## Quantization core

- `contracts.py` contains the shared name/shape value object.
- `anima.py` owns the Anima width-2048 signature, contiguous 28/40-block
  detection, exact shapes, and 426/448 (2B) or 618/640 (2.9B) presets. Public
  selection helpers default to 28 blocks for existing callers; validation
  detects depth from the state dict. Node progress and publication count checks
  use this same detected selection.
- `krea2.py` owns the Krea2 signature, fixed open-weight architecture, and 224
  block-linear selection.
- `minimax_h3.py` owns the supported 50/2 curve-form signature and the exact
  200 main-block matrix selection inferred from the official reference header.
- `convrot.py` contains regular-Hadamard generation and offline weight rotation.
- `moss_image.py` validates the complete 42-layer source layout and selects
  168 fused-QKV or 252 separate-QKV matrices, preserving the native namespace.
- `export.py` validates selected tensors, performs rowwise INT8 quantization,
  writes marker/scale tensors, and copies non-selected tensors unchanged.

Public export functions are thin model-contract wrappers around shared writers.
Anima/Krea2 use `_write_int8_convrot_checkpoint_from_specs`; MiniMax H3 and
Moss Image use the bounded-memory streaming writer. The old
`AnimaInt8ExportReport` name remains as an alias for compatibility.

The core has no generic registry, CLI, graph/policy framework, submodule, or
external quantization-package import.
