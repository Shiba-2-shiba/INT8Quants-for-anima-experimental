# Architecture

The repository is one self-contained ComfyUI custom-node package with two
model-specific output nodes and one shared INT8 writer.

```text
ComfyUI Backend V3
  __init__.py -> nodes.py
                     |
                     v
                  service.py
                     |
                     v
  quantization/{contracts,anima,krea2,convrot,export}.py
```

## ComfyUI adapter

`nodes.py` registers separate Anima and Krea2 schemas. A small immutable node
profile shares the identical controls, progress handling, and summary shaping
without merging the public node IDs or model-specific descriptions. Existing
Anima workflows continue to reference `ComfyQuantsAnimaInt8ConvRotSave`.

There is no custom frontend extension. Both nodes use only Backend V3 metadata,
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

## Quantization core

- `contracts.py` contains the shared name/shape value object.
- `anima.py` owns the Anima 2B signature, 28-block shapes, and 426/448 presets.
- `krea2.py` owns the Krea2 signature, fixed open-weight architecture, and 224
  block-linear selection.
- `convrot.py` contains regular-Hadamard generation and offline weight rotation.
- `export.py` validates selected tensors, performs rowwise INT8 quantization,
  writes marker/scale tensors, and copies non-selected tensors unchanged.

Public Anima and Krea2 export functions are thin model-contract wrappers around
the same `_write_int8_convrot_checkpoint_from_specs` implementation. The old
`AnimaInt8ExportReport` name remains as an alias for compatibility.

The core has no generic registry, CLI, graph/policy framework, submodule, or
external quantization-package import.
