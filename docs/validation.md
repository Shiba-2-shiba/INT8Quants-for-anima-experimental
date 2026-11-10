# Validation

## Automated gates

- Anima signature, contiguous 28/40-block coverage, exact tensor shapes
- Ordered 426/448 (2B) and 618/640 (2.9B) selections with their 22-weight difference
- Detected Anima counts in progress and publication, rejection of truncated 2B reports for 2.9B
- Optional local Anima 2.9B reference-header and ConvRot-marker comparison (see `anima.md`)
- Krea2 signature, width 6144, 28-block coverage, and exact 224-weight selection
- Krea2 native checkpoint namespace and stock ComfyUI model detection
- MiniMax H3 50/2 curve-form signature, exact 200-weight selection, keep
  boundary, complete 532-source/332-copied tensor contract, and stock ComfyUI
  model detection
- Hadamard orthogonality and ConvRot math
- Moss Image official reference manifest, 168 fused / 252 separate QKV
  selections, complete source validation, and optional stock model detection
- FP16/BF16 rowwise INT8, FP32 scales, exact marker bytes
- Input mapping/tensor non-mutation
- Missing/rank/shape/dtype/non-finite rejection
- Path traversal, Windows reserved names, and disk preflight
- Checkpoint/report publication and rollback
- Lazy package import and Backend V3 schema registration

During the self-contained migration, both preset selections matched the old
vendor in order and shape. A deterministic BF16 fixture produced identical
weight, scale, marker, and copied tensors with the old and new writers.

## Release gate

Before publishing each release:

1. Confirm the Registry `PublisherId`, immutable project name, and GitHub URLs;
   `Shiba-2-shiba` is currently provisional from the intended GitHub owner and
   Registry ownership cannot be verified offline.
2. Export real Anima 2B and 2.9B BF16 and FP16 sources with both presets.
3. Export real Krea2 Raw and Turbo sources with `quality_keep`.
4. Export the supported real MiniMax H3 source with `strict_reference` on a
   machine with sufficient host memory.
5. Confirm `quantized_tensor_count == rotated_tensor_count == 426/448` for
   Anima 2B, `618/640` for Anima 2.9B, `224` for Krea2, and `200` for MiniMax H3.
6. Record output size, SHA-256, elapsed time, and peak host memory.
7. Reload the artifact with stock `Load Diffusion Model`.
8. For Krea2, use `CLIPLoader` type `krea2`, Qwen3-VL-4B text weights, and the
   compatible Qwen Image VAE before fixed-seed inference.
9. For MiniMax H3, compare generated INT8 weights/scales with the official
   reference when the matching floating-point source is available, then run
   the fixed-input numerical and short-sample gates in `docs/minimax_h3.md`.
10. Run fixed seed/prompt workflows and check for NaN, Inf, or crashes.
11. Record CPU export separately from CUDA/ROCm runtime results.
12. Run `python tools/validate_release_archive.py` and install the resulting
   `.comfyignore`-filtered file set into a clean ComfyUI checkout.
13. Publish from a clean public root commit; do not publish the local migration
   baseline, which contains pre-refactor development paths and Git identity.
14. Confirm Krea2 distribution naming, Notice, license, and use-policy
    obligations before publishing quantized Krea2 weights.

Do not commit source models, generated checkpoints, or private local paths.

For Moss Image, full-model export/reload/inference are pending user-run
validation. See [moss_image.md](moss_image.md) for expected counts and the
reference's optional attention/RoPE metadata differences.

## Anima 2.9B validation scope

The supplied public INT8 checkpoint was inspected without loading its large
weight payload. All 640 selected names/shapes, 640 FP32 row scales, and 640
ConvRot markers match `public_examples`; 288 BF16 tensors remain unselected.
Host-independent regressions cover both depths and presets. Backend V3 node
execution/progress was checked with the local Desktop host (ComfyUI 0.22.3).
Stock detection for Anima 28/40, Krea2, and MiniMax H3 was checked separately
against the official ComfyUI v0.33.1 `model_detection.py` and `utils.py` sources.
This source-level check is not a complete v0.33.1 runtime reload/inference test.

The older Desktop host hard-codes 28 Anima blocks and lacks Krea2/MiniMax H3
detection; it cannot serve as the full supported-runtime validation target.
CI uses v0.33.1 for the complete Backend V3 and stock-detection suite.
Full 2.9B floating-point export, stock reload, GPU inference, and image-quality
comparison remain release gates; the provided INT8 reference alone cannot
establish them. See [`anima.md`](anima.md) for the repeatable reference test.
