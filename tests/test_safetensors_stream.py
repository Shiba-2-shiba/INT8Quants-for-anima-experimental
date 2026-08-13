from __future__ import annotations

import io

import pytest

from quantization import safetensors_stream as stream


class _ShortWriter(io.BytesIO):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def write(self, data):
        return super().write(memoryview(data).cast("B")[: self.limit])


@pytest.mark.parametrize("size", [6, 7, 8, 15])
def test_write_all_handles_short_writes_and_byte_boundaries(size):
    payload = bytes(range(size))
    handle = _ShortWriter(limit=3)

    stream.write_all(handle, memoryview(payload))

    assert handle.getvalue() == payload


def test_streaming_file_does_not_delete_preexisting_output(tmp_path):
    output = tmp_path / "existing.safetensors"
    output.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        stream.write_streaming_file(output, (), {}, ())

    assert output.read_bytes() == b"keep me"
