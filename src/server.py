from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from aiohttp import web
from server import PromptServer

from .runtime import ensure_dir, get_runtime_paths
from .profiles import load_profiles


ROUTES = PromptServer.instance.routes
LORA_ID_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def _plugin_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _profiles_dir() -> Path:
    return _plugin_root() / "profiles"


def _cache_dirs() -> list[Path]:
    paths = get_runtime_paths()
    return [paths.cache, paths.datasets, paths.artifacts]


def _last_lora_info_path() -> Path:
    return get_runtime_paths().root / "last_lora.json"


def _is_path_within(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sidecar_paths(lora_path: Path) -> list[Path]:
    base_path = lora_path.with_suffix("")
    sidecars = [
        base_path.with_suffix(".metadata.json"),
        lora_path.with_name(f"{lora_path.name}.rgthree-info.json"),
        base_path.with_suffix(".cm-info.json"),
        base_path.with_suffix(".cminfo.json"),
    ]
    sidecars.extend(lora_path.parent.glob(f"{base_path.name}.preview.*"))
    return sidecars


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


def _safe_lora_id(raw_id: str) -> str:
    lora_id = raw_id.strip().lower()
    if not LORA_ID_PATTERN.match(lora_id):
        raise ValueError("Invalid LoRA id.")
    return lora_id


def _manifest_path(lora_id: str) -> Path:
    return _generated_lora_dir(lora_id) / "manifest.json"


def _generated_lora_dir(lora_id: str) -> Path:
    return get_runtime_paths().generated_loras / lora_id


def _latest_generated_lora(lora_id: str) -> Path | None:
    lora_dir = _generated_lora_dir(lora_id)
    if not lora_dir.exists():
        return None
    candidates = sorted(lora_dir.glob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _metadata_path(lora_path: Path) -> Path:
    return lora_path.with_suffix(".metadata.json")


def _resolve_generated_sidecar(raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    sidecar_path = Path(raw_path).expanduser().resolve()
    generated_root = get_runtime_paths().generated_loras.resolve()
    if _is_path_within(generated_root, sidecar_path) and sidecar_path.exists() and sidecar_path.is_file():
        return sidecar_path
    return None


def _resolve_generated_lora(lora_id: str, manifest: dict[str, object]) -> Path | None:
    raw_lora_path = manifest.get("lora_path")
    if isinstance(raw_lora_path, str) and raw_lora_path.strip():
        lora_path = Path(raw_lora_path).expanduser().resolve()
        outputs_root = get_runtime_paths().generated_loras.resolve()
        if _is_path_within(outputs_root, lora_path) and lora_path.exists() and lora_path.suffix.lower() == ".safetensors":
            return lora_path
    return _latest_generated_lora(lora_id)


def _caption_files_payload(subset_dir: Path) -> dict[str, str]:
    captions: dict[str, str] = {}
    for path in sorted(subset_dir.glob("*.txt")):
        captions[path.stem] = path.read_text(encoding="utf-8").strip()
    return captions


def _find_dataset_dir_by_captions(captions: object) -> Path | None:
    if not isinstance(captions, dict) or not captions:
        return None
    expected = {str(key): str(value).strip() for key, value in captions.items()}
    datasets_root = get_runtime_paths().datasets.resolve()
    for dataset_dir in sorted(datasets_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for subset_dir in sorted(dataset_dir.iterdir()):
            if not subset_dir.is_dir():
                continue
            try:
                if _caption_files_payload(subset_dir) == expected:
                    return dataset_dir
            except OSError:
                continue
    return None


def _resolve_dataset_dir(manifest: dict[str, object]) -> Path | None:
    datasets_root = get_runtime_paths().datasets.resolve()
    for key in ("dataset_dir", "dataset_path"):
        raw_path = manifest.get(key)
        if isinstance(raw_path, str) and raw_path.strip():
            dataset_dir = Path(raw_path).expanduser().resolve()
            if _is_path_within(datasets_root, dataset_dir) and dataset_dir.exists() and dataset_dir.is_dir():
                return dataset_dir
    raw_dataset_key = manifest.get("dataset_key")
    if isinstance(raw_dataset_key, str) and LORA_ID_PATTERN.match(raw_dataset_key):
        dataset_dir = datasets_root / raw_dataset_key
        if dataset_dir.exists() and dataset_dir.is_dir():
            return dataset_dir
    return _find_dataset_dir_by_captions(manifest.get("captions"))


def _resolve_preview_sidecar(lora_path: Path, manifest: dict[str, object]) -> Path | None:
    manifest_preview = _resolve_generated_sidecar(manifest.get("preview_path"))
    if manifest_preview is not None:
        return manifest_preview

    metadata = _read_json(_metadata_path(lora_path))
    metadata_preview = _resolve_generated_sidecar(metadata.get("preview_url"))
    if metadata_preview is not None:
        return metadata_preview

    base_path = lora_path.with_suffix("")
    for preview_path in sorted(lora_path.parent.glob(f"{base_path.name}.preview.*")):
        if preview_path.is_file():
            return preview_path
    return None


def _resolve_dataset_thumbnail_path(manifest: dict[str, object]) -> Path | None:
    datasets_root = get_runtime_paths().datasets.resolve()
    raw_thumbnail = manifest.get("thumbnail_path")
    if isinstance(raw_thumbnail, str) and raw_thumbnail.strip():
        thumbnail_path = Path(raw_thumbnail).expanduser().resolve()
        if _is_path_within(datasets_root, thumbnail_path) and thumbnail_path.exists() and thumbnail_path.is_file():
            return thumbnail_path

    dataset_dir = _resolve_dataset_dir(manifest)
    if dataset_dir is None:
        return None
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        candidates = sorted(dataset_dir.rglob(pattern))
        if candidates:
            return candidates[0]
    return None


def _resolve_thumbnail_path(lora_path: Path, manifest: dict[str, object]) -> Path | None:
    preview_path = _resolve_preview_sidecar(lora_path, manifest)
    if preview_path is not None:
        return preview_path
    return _resolve_dataset_thumbnail_path(manifest)


def _profile_name(profile_key: object) -> str:
    if not isinstance(profile_key, str):
        return ""
    for profile in load_profiles(_plugin_root()):
        if profile.key == profile_key:
            return profile.name
    return profile_key


def _library_item(lora_id: str) -> dict[str, object] | None:
    manifest = _read_json(_manifest_path(lora_id))
    lora_path = _resolve_generated_lora(lora_id, manifest)
    if lora_path is None:
        return None

    stat = lora_path.stat()
    thumbnail_path = _resolve_thumbnail_path(lora_path, manifest)
    metadata = _read_json(_metadata_path(lora_path))
    tags = manifest.get("tags")
    metadata_tags = metadata.get("tags")
    if not isinstance(tags, str) and isinstance(metadata_tags, list):
        tags = ", ".join(str(tag) for tag in metadata_tags)
    if not isinstance(tags, str):
        tags = ""
    captions = manifest.get("captions")
    caption_count = len(captions) if isinstance(captions, dict) else 0
    profile = manifest.get("profile", metadata.get("base_model", ""))
    return {
        "id": lora_id,
        "name": lora_path.stem,
        "file_name": lora_path.name,
        "lora_path": str(lora_path),
        "size_bytes": stat.st_size,
        "size_human": _format_bytes(stat.st_size),
        "modified_at": stat.st_mtime,
        "profile": profile,
        "profile_name": _profile_name(profile),
        "tags": tags,
        "caption_count": caption_count,
        "has_thumbnail": thumbnail_path is not None,
        "thumbnail_url": f"/instant-reference-lora/library/thumbnail?id={lora_id}" if thumbnail_path else "",
    }


def _library_payload() -> dict[str, object]:
    paths = get_runtime_paths()
    ids: set[str] = set()
    if paths.generated_loras.exists():
        for lora_dir in paths.generated_loras.iterdir():
            if lora_dir.is_dir() and LORA_ID_PATTERN.match(lora_dir.name):
                ids.add(lora_dir.name)

    items = [item for lora_id in sorted(ids) if (item := _library_item(lora_id)) is not None]
    items.sort(
        key=lambda item: item["modified_at"] if isinstance(item["modified_at"], (int, float)) else 0.0,
        reverse=True,
    )
    return {
        "items": items,
        "generated_loras_dir": str(paths.generated_loras),
        "lora_dir": str(paths.generated_loras.parent),
    }


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
        "generated_loras_dir": str(get_runtime_paths().generated_loras),
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


@ROUTES.get("/instant-reference-lora/library")
async def instant_reference_lora_library(_request):
    return web.json_response(_library_payload())


@ROUTES.get("/instant-reference-lora/library/thumbnail")
async def instant_reference_lora_library_thumbnail(request):
    try:
        lora_id = _safe_lora_id(request.query.get("id", ""))
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    manifest = _read_json(_manifest_path(lora_id))
    lora_path = _resolve_generated_lora(lora_id, manifest)
    if lora_path is None:
        return web.json_response({"error": "LoRA file was not found."}, status=404)
    thumbnail_path = _resolve_thumbnail_path(lora_path, manifest)
    if thumbnail_path is None:
        return web.json_response({"error": "Thumbnail image was not found."}, status=404)
    return web.FileResponse(path=thumbnail_path)


@ROUTES.post("/instant-reference-lora/library/delete")
async def instant_reference_lora_library_delete(request):
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body."}, status=400)

    try:
        lora_id = _safe_lora_id(str(payload.get("id", "")))
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    manifest_path = _manifest_path(lora_id)
    manifest = _read_json(manifest_path)
    lora_path = _resolve_generated_lora(lora_id, manifest)
    if lora_path is None:
        return web.json_response({"error": "LoRA file was not found."}, status=404)

    outputs_root = get_runtime_paths().generated_loras.resolve()
    resolved_lora_path = lora_path.resolve()
    if not _is_path_within(outputs_root, resolved_lora_path):
        return web.json_response({"error": "Only generated LoRA files can be deleted."}, status=400)

    resolved_lora_path.unlink(missing_ok=True)
    for sidecar_path in _sidecar_paths(resolved_lora_path):
        try:
            if sidecar_path.exists() and _is_path_within(outputs_root, sidecar_path.resolve()):
                sidecar_path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        if resolved_lora_path.parent.exists() and not any(resolved_lora_path.parent.iterdir()):
            resolved_lora_path.parent.rmdir()
    except OSError:
        pass

    manifest["deleted_lora_path"] = str(resolved_lora_path)
    if manifest.get("lora_path") == str(resolved_lora_path):
        manifest.pop("lora_path", None)
    _write_json(manifest_path, manifest)
    return web.json_response({"success": True})


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
    outputs_root = get_runtime_paths().generated_loras.resolve()
    if not _is_path_within(outputs_root, requested_path):
        return web.json_response({"error": "Only cached LoRA files can be downloaded."}, status=400)
    if requested_path.suffix.lower() != ".safetensors":
        return web.json_response({"error": "Only .safetensors files can be downloaded."}, status=400)
    if not requested_path.exists() or not requested_path.is_file():
        return web.json_response({"error": "LoRA file was not found."}, status=404)

    return web.FileResponse(path=requested_path, headers={
        "Content-Disposition": f'attachment; filename="{requested_path.name}"'
    })
