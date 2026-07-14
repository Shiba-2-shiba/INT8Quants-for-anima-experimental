# Validation

## Automated gates

- Anima signature, 28-block coverage, exact tensor shapes
- Ordered 426/448 selections and their documented 22-weight difference
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
3. Confirm `quantized_tensor_count == rotated_tensor_count == 426/448`.
4. Record output size, SHA-256, elapsed time, and peak host memory.
5. Reload the artifact with stock `Load Diffusion Model`.
6. Run a fixed seed/prompt workflow and check for NaN, Inf, or crashes.
7. Record CPU export separately from CUDA/ROCm runtime results.
8. Run `python tools/validate_release_archive.py` and install the resulting
   `.comfyignore`-filtered file set into a clean ComfyUI checkout.
9. Publish from a clean public root commit; do not publish the local migration
   baseline, which contains pre-refactor development paths and Git identity.

Do not commit source models, generated checkpoints, or private local paths.
