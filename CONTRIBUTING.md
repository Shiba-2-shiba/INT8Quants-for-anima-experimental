# Contributing

Keep changes focused on the self-contained Anima 2B and Krea2 INT8 export contracts.

1. Open an issue for behavior or compatibility changes.
2. Add a failing regression test before changing quantization or publication
   behavior.
3. Run `python -m pytest -q`.
4. Run `tests/test_schema.py` with the validated ComfyUI checkout on
   `PYTHONPATH`.
5. Do not add model weights, generated checkpoints, local paths, runtime
   `sys.path` manipulation, or an external `comfy_quants` dependency.
6. Preserve GPL attribution when changing derived quantization code.

Commits should explain intent, constraints, rejected alternatives where useful,
and the verification performed.
