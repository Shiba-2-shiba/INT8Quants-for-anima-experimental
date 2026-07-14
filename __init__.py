from __future__ import annotations


async def comfy_entrypoint():
    """Load the ComfyUI-dependent node module only when the host calls it."""

    from .nodes import comfy_entrypoint as nodes_entrypoint

    return await nodes_entrypoint()

__all__ = ["comfy_entrypoint"]
