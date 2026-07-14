# Third-party notices

## Comfy Quants

The files below contain code derived from the Comfy Quants project:

- Upstream: <https://github.com/Comfy-Org/comfy-quants>
- Base commit: `1e0d481f24847c4914578f5468917902ad53ea46`
- License: GPL-3.0-only
- Copyright: Comfy Quants Project contributors

| Local file | Principal upstream source |
| --- | --- |
| `quantization/anima.py` | `model_adapters/anima.py`, `model_adapters/anima_contracts/` |
| `quantization/convrot.py` | `formats/convrot.py` |
| `quantization/export.py` | `api.py`, `backends/int8_tensorwise_model_export.py` |

The local implementation is limited to Anima 2B state-dict export. Generic
registries, graph/policy layers, CLI commands, file-input exporters, and all
non-INT8 backends were removed. Anima detection was aligned with ComfyUI's LLM
adapter signature, non-finite input validation and disk preflight were added,
and publication is owned by the custom-node service. These changes were made
for this custom node on 2026-07-14; the derived source files carry matching
modification notices.

Per-file SPDX and derivation comments are retained in the self-contained
quantization modules. The upstream commit above is the reproducible reference
for reviewing the extracted implementation.
