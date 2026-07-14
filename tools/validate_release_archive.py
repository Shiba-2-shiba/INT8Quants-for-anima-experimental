"""Validate and smoke-test the file set selected by ``.comfyignore``."""

from __future__ import annotations

import fnmatch
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    ".comfyignore",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "__init__.py",
    "nodes.py",
    "pyproject.toml",
    "quantization/__init__.py",
    "quantization/anima.py",
    "quantization/convrot.py",
    "quantization/export.py",
    "requirements.txt",
    "service.py",
}
FORBIDDEN_ROOTS = {
    ".agents",
    ".ci",
    ".git",
    ".github",
    ".omx",
    "docs",
    "tests",
    "third_party",
    "tools",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".pyc", ".pt", ".pth", ".safetensors", ".tmp"}


def _tracked_and_untracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        ROOT / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item and (ROOT / item.decode("utf-8")).is_file()
    ]


def _patterns() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _is_excluded(relative: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        normalized = pattern.replace("\\", "/")
        if normalized.endswith("/"):
            prefix = normalized.rstrip("/")
            if relative == prefix or relative.startswith(f"{prefix}/"):
                return True
        elif "/" in normalized:
            if fnmatch.fnmatchcase(relative, normalized):
                return True
        elif fnmatch.fnmatchcase(Path(relative).name, normalized):
            return True
    return False


def _candidate_files() -> list[tuple[Path, str]]:
    patterns = _patterns()
    candidates = []
    for path in _tracked_and_untracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if not _is_excluded(relative, patterns):
            candidates.append((path, relative))
    return sorted(candidates, key=lambda item: item[1])


def _validate_contents(candidates: list[tuple[Path, str]]) -> None:
    names = {relative for _path, relative in candidates}
    missing = sorted(REQUIRED - names)
    if missing:
        raise RuntimeError(f"release archive is missing runtime files: {missing}")

    forbidden = []
    for _path, relative in candidates:
        parts = Path(relative).parts
        suffix = Path(relative).suffix.casefold()
        if parts[0] in FORBIDDEN_ROOTS or suffix in FORBIDDEN_SUFFIXES:
            forbidden.append(relative)
        if relative.endswith((".export_report.json", ".report.json")):
            forbidden.append(relative)
    if forbidden:
        forbidden_names = sorted(set(forbidden))
        raise RuntimeError(f"release archive contains forbidden files: {forbidden_names}")


def _smoke_test(candidates: list[tuple[Path, str]]) -> None:
    with tempfile.TemporaryDirectory(prefix="anima-int8-release-") as directory:
        staging = Path(directory)
        for source, relative in candidates:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        check = """
import importlib.util
import sys
from pathlib import Path

root = Path.cwd()
spec = importlib.util.spec_from_file_location(
    "release_candidate", root / "__init__.py", submodule_search_locations=[str(root)]
)
assert spec is not None and spec.loader is not None
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
assert callable(package.comfy_entrypoint)

quant_spec = importlib.util.spec_from_file_location(
    "release_quantization", root / "quantization" / "__init__.py",
    submodule_search_locations=[str(root / "quantization")],
)
assert quant_spec is not None and quant_spec.loader is not None
quantization = importlib.util.module_from_spec(quant_spec)
sys.modules[quant_spec.name] = quantization
quant_spec.loader.exec_module(quantization)
assert callable(quantization.export_anima_int8_convrot_from_state_dict)
"""
        subprocess.run(
            [sys.executable, "-B", "-c", check],
            cwd=staging,
            check=True,
        )


def main() -> int:
    candidates = _candidate_files()
    _validate_contents(candidates)
    _smoke_test(candidates)
    print(f"Validated {len(candidates)} release archive files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
