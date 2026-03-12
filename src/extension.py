from __future__ import annotations

from .nodes import ReferenceTrainingExtension


async def comfy_entrypoint() -> ReferenceTrainingExtension:
    return ReferenceTrainingExtension()
