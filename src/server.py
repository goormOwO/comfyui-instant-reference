from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from aiohttp import web
from server import PromptServer

from .runtime import ensure_dir, get_runtime_paths
from .profiles import load_profiles


ROUTES = PromptServer.instance.routes


def _plugin_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _profiles_dir() -> Path:
    return _plugin_root() / "profiles"


def _cache_dirs() -> list[Path]:
    paths = get_runtime_paths()
    return [paths.cache, paths.outputs, paths.datasets, paths.artifacts]


def _last_lora_info_path() -> Path:
    return get_runtime_paths().root / "last_lora.json"


def _is_path_within(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _cache_info_payload() -> dict[str, object]:
    breakdown: dict[str, int] = {}
    total = 0
    for path in _cache_dirs():
        size = _dir_size_bytes(path)
        breakdown[path.name] = size
        total += size
    profiles_dir = ensure_dir(_profiles_dir())
    return {
        "profiles_dir": str(profiles_dir),
        "total_bytes": total,
        "total_human": _format_bytes(total),
        "breakdown_bytes": breakdown,
        "breakdown_human": {name: _format_bytes(size) for name, size in breakdown.items()},
    }


def _profile_slots_payload() -> dict[str, object]:
    profiles = load_profiles(_plugin_root())
    return {
        "profiles": {
            profile.key: {
                "name": profile.name,
                "slots": [{"name": slot.name, "type": slot.slot_type} for slot in profile.slots],
            }
            for profile in profiles
        }
    }


def _clear_dir_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            continue


def read_last_lora_info() -> dict[str, object]:
    path = _last_lora_info_path()
    if not path.exists():
        return {"path": "", "exists": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": "", "exists": False}

    lora_path = str(payload.get("path", ""))
    exists = False
    if lora_path:
        exists = Path(lora_path).exists()
    return {
        "path": lora_path,
        "exists": exists,
    }


@ROUTES.get("/instant-reference-lora/cache-info")
async def instant_reference_lora_cache_info(_request):
    return web.json_response(_cache_info_payload())


@ROUTES.get("/instant-reference-lora/profiles")
async def instant_reference_lora_profiles(_request):
    return web.json_response(_profile_slots_payload())


@ROUTES.get("/instant-reference-lora/last-lora")
async def instant_reference_lora_last_lora(_request):
    return web.json_response(read_last_lora_info())


@ROUTES.post("/instant-reference-lora/open-profiles")
async def instant_reference_lora_open_profiles(_request):
    profiles_dir = ensure_dir(_profiles_dir())
    try:
        os.startfile(str(profiles_dir))  # type: ignore[attr-defined]
    except AttributeError:
        return web.json_response(
            {"success": False, "error": "Opening folders is only supported on this platform.", "profiles_dir": str(profiles_dir)},
            status=400,
        )
    except OSError as exc:
        return web.json_response(
            {"success": False, "error": str(exc), "profiles_dir": str(profiles_dir)},
            status=500,
        )
    return web.json_response({"success": True, "profiles_dir": str(profiles_dir)})


@ROUTES.post("/instant-reference-lora/clear-cache")
async def instant_reference_lora_clear_cache(_request):
    for path in _cache_dirs():
        ensure_dir(path)
        _clear_dir_contents(path)
    payload = _cache_info_payload()
    payload["success"] = True
    return web.json_response(payload)


@ROUTES.get("/instant-reference-lora/download")
async def instant_reference_lora_download(request):
    raw_path = request.query.get("path", "").strip()
    if not raw_path:
        return web.json_response({"error": "Missing LoRA path."}, status=400)

    requested_path = Path(raw_path).expanduser().resolve()
    outputs_root = get_runtime_paths().outputs.resolve()
    if not _is_path_within(outputs_root, requested_path):
        return web.json_response({"error": "Only cached LoRA files can be downloaded."}, status=400)
    if requested_path.suffix.lower() != ".safetensors":
        return web.json_response({"error": "Only .safetensors files can be downloaded."}, status=400)
    if not requested_path.exists() or not requested_path.is_file():
        return web.json_response({"error": "LoRA file was not found."}, status=404)

    return web.FileResponse(path=requested_path, headers={
        "Content-Disposition": f'attachment; filename="{requested_path.name}"'
    })
