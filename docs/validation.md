# Validation

## Automated gates

- Anima signature, 28-block coverage, exact tensor shapes
- Ordered 426/448 selections and their documented 22-weight difference
- Krea2 signature, width 6144, 28-block coverage, and exact 224-weight selection
- Krea2 native checkpoint namespace and stock ComfyUI model detection
- Hadamard orthogonality and ConvRot math
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
2. Export a real Anima 2B BF16 and FP16 source with both presets.
3. Export real Krea2 Raw and Turbo sources with `quality_keep`.
4. Confirm `quantized_tensor_count == rotated_tensor_count == 426/448` for
   Anima and `224` for Krea2.
5. Record output size, SHA-256, elapsed time, and peak host memory.
6. Reload the artifact with stock `Load Diffusion Model`.
7. For Krea2, use `CLIPLoader` type `krea2`, Qwen3-VL-4B text weights, and the
   compatible Qwen Image VAE before fixed-seed inference.
8. Run fixed seed/prompt workflows and check for NaN, Inf, or crashes.
9. Record CPU export separately from CUDA/ROCm runtime results.
10. Run `python tools/validate_release_archive.py` and install the resulting
   `.comfyignore`-filtered file set into a clean ComfyUI checkout.
11. Publish from a clean public root commit; do not publish the local migration
   baseline, which contains pre-refactor development paths and Git identity.
12. Confirm Krea2 distribution naming, Notice, license, and use-policy
    obligations before publishing quantized Krea2 weights.

Do not commit source models, generated checkpoints, or private local paths.
