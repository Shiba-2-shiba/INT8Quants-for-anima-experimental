# Development

## Setup

Use Python 3.10 or newer. ComfyUI supplies PyTorch; install direct development
requirements in the same environment:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

## Test layers

Host-independent regression suite:

```bash
python -m pytest -q
```

Backend V3 integration against a local ComfyUI checkout:

```powershell
$env:PYTHONPATH='C:\path\to\ComfyUI'
python -m pytest -q tests/test_schema.py
```

Syntax verification without writing bytecode:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -m pytest -q
```

## Change discipline

1. Add or update a regression test first.
2. Keep Anima names and shapes centralized in `quantization/anima.py`.
3. Do not add runtime path manipulation or an external `comfy_quants`
   dependency.
4. Verify both Anima presets, the Krea2 224-weight contract, marker bytes,
   input non-mutation, and transaction rollback after quantization changes.
5. Record real-model export, stock reload, and inference evidence before a
   release.
6. Keep model-specific namespace normalization and shape validation in the
   matching `quantization/{anima,krea2}.py` module.
