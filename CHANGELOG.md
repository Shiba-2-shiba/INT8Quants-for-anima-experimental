# Changelog

All notable changes are documented here. Versions follow semantic versioning.

## 0.1.0 - Unreleased

- Provide the Anima 2B INT8 ConvRot Backend V3 output node.
- Integrate the minimal Anima contract, ConvRot, and state-dict writer directly
  into the custom node.
- Remove the former `third_party/comfy-quants` snapshot and runtime path
  manipulation.
- Add exact 426/448 preset, marker, numeric, transaction, and V3 registration
  regression tests.
- Add Registry metadata, GPL-3.0 provenance, portable installation guidance,
  and Windows/Linux CI.
- Store generated checkpoints and reports together under the configured
  `output/diffusion_models/` directory and register it for stock model loading.
- Add a separate Krea2 Raw/Turbo INT8 ConvRot output node with an exact
  28-block, 224-matrix contract.
- Share ModelPatcher checks, atomic publication, node controls, and the INT8
  writer while preserving the existing Anima node ID and public export API.
- Validate Krea2's native namespace against stock ComfyUI model detection and
  document the official checkpoints and model-license boundary.
