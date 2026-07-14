# Architecture

The repository is one self-contained ComfyUI custom node. Runtime code has
three boundaries:

```text
ComfyUI Backend V3
  __init__.py -> nodes.py
                     |
                     v
                  service.py
                     |
                     v
        quantization/{anima,convrot,export}.py
```

## ComfyUI adapter

`nodes.py` defines the V3 schema, obtains ComfyUI model directories, reports
progress, and converts the result to `NodeOutput`. `__init__.py` is lazy so the
package can be inspected without importing ComfyUI.

## Application service

`service.py` normalizes the ModelPatcher state dict to `net.*`, rejects patched
or incompatible models, validates output paths and disk capacity, and owns the
checkpoint/report publication transaction. It translates internal domain
errors to actionable node errors.

## Quantization core

- `anima.py` is the single source of truth for the Anima 2B signature, 28-block
  shapes, and the 426/448 preset selections.
- `convrot.py` contains only regular-Hadamard generation and offline weight
  rotation.
- `export.py` validates selected tensors, performs per-row INT8 quantization,
  writes the stock ComfyUI marker/scale tensors, and creates a safetensors
  artifact report.

The core has no registry, CLI, graph/policy framework, submodule, or external
`comfy_quants` import.
