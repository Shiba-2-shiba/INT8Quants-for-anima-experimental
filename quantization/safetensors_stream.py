# SPDX-License-Identifier: GPL-3.0-only
"""Bounded-memory writer for the simple, contiguous safetensors file format."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


class SafetensorsStreamError(RuntimeError):
    """Raised when a streaming safetensors file cannot be written safely."""


_DTYPE_CODES = {
    "torch.bool": "BOOL",
    "torch.uint8": "U8",
    "torch.int8": "I8",
    "torch.int16": "I16",
    "torch.int32": "I32",
    "torch.int64": "I64",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float32": "F32",
    "torch.float64": "F64",
}


@dataclass(frozen=True, slots=True)
class TensorLayout:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def byte_length(self) -> int:
        return self.end - self.start


def dtype_code(dtype: Any) -> str:
    try:
        return _DTYPE_CODES[str(dtype)]
    except KeyError as exc:
        raise SafetensorsStreamError(f"unsupported safetensors dtype: {dtype}") from exc


def tensor_byte_length(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def build_layouts(
    tensor_entries: Sequence[tuple[str, Any]],
) -> tuple[TensorLayout, ...]:
    layouts: list[TensorLayout] = []
    offset = 0
    names: set[str] = set()
    for name, tensor in tensor_entries:
        if name in names:
            raise SafetensorsStreamError(f"duplicate safetensors tensor name: {name}")
        names.add(name)
        byte_length = tensor_byte_length(tensor)
        layout = TensorLayout(
            name=name,
            dtype=dtype_code(tensor.dtype),
            shape=tuple(int(dimension) for dimension in tensor.shape),
            start=offset,
            end=offset + byte_length,
        )
        layouts.append(layout)
        offset = layout.end
    return tuple(layouts)


def encode_header(
    layouts: Sequence[TensorLayout],
    metadata: Mapping[str, str] | None,
) -> bytes:
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    for layout in layouts:
        header[layout.name] = {
            "dtype": layout.dtype,
            "shape": list(layout.shape),
            "data_offsets": [layout.start, layout.end],
        }
    encoded = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    padding = (-len(encoded)) % 8
    return encoded + b" " * padding


def write_all(handle: BinaryIO, data: bytes | bytearray | memoryview) -> None:
    view = memoryview(data).cast("B")
    written = 0
    while written < view.nbytes:
        count = handle.write(view[written:])
        if count is None or int(count) <= 0:
            raise SafetensorsStreamError("safetensors stream write made no progress")
        written += int(count)


def tensor_bytes_view(tensor: Any) -> memoryview:
    cpu = tensor.detach().to(device="cpu").contiguous()
    return memoryview(cpu.view(dtype=_require_torch().uint8).numpy())


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - provided by ComfyUI
        raise SafetensorsStreamError("torch is required for safetensors streaming") from exc
    return torch


def write_streaming_file(
    path: str | Path,
    layouts: Sequence[TensorLayout],
    metadata: Mapping[str, str] | None,
    writers: Sequence[Callable[[BinaryIO], None]],
) -> None:
    if len(layouts) != len(writers):
        raise SafetensorsStreamError(
            f"layout/writer count mismatch: {len(layouts)} != {len(writers)}"
        )
    output = Path(path)
    header = encode_header(layouts, metadata)
    created = False
    try:
        with output.open("xb") as handle:
            created = True
            write_all(handle, struct.pack("<Q", len(header)))
            write_all(handle, header)
            for layout, writer in zip(layouts, writers):
                start = handle.tell()
                writer(handle)
                actual = handle.tell() - start
                if actual != layout.byte_length:
                    raise SafetensorsStreamError(
                        f"tensor byte count mismatch for {layout.name}: "
                        f"expected {layout.byte_length}, wrote {actual}"
                    )
            handle.flush()
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise


__all__ = [
    "SafetensorsStreamError",
    "TensorLayout",
    "build_layouts",
    "dtype_code",
    "encode_header",
    "tensor_byte_length",
    "tensor_bytes_view",
    "write_all",
    "write_streaming_file",
]
