from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SLOT_PATTERN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*):(MODEL|CLIP|VAE|STRING)\}\}")
BUILTIN_PATTERN = re.compile(r"\{\{([A-Z_][A-Z0-9_]*)\}\}")
SUPPORTED_SLOT_TYPES = {"MODEL", "CLIP", "VAE", "STRING"}


@dataclass(frozen=True)
class SlotSpec:
    name: str
    slot_type: str


@dataclass(frozen=True)
class ProfileDefinition:
    key: str
    name: str
    script: str
    config: str
    slots: tuple[SlotSpec, ...]
    file_path: Path
    file_hash: str


def profiles_dir(root_dir: Path) -> Path:
    return root_dir / "profiles"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_slots(config: str) -> tuple[SlotSpec, ...]:
    seen: dict[str, str] = {}
    ordered: list[SlotSpec] = []
    for match in SLOT_PATTERN.finditer(config):
        name, slot_type = match.groups()
        existing = seen.get(name)
        if existing is not None and existing != slot_type:
            raise ValueError(f"Profile slot '{name}' uses conflicting types: {existing} vs {slot_type}")
        if existing is None:
            seen[name] = slot_type
            ordered.append(SlotSpec(name=name, slot_type=slot_type))
    return tuple(ordered)


def load_profiles(root_dir: Path) -> list[ProfileDefinition]:
    directory = profiles_dir(root_dir)
    profiles: list[ProfileDefinition] = []
    for path in sorted(directory.glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        name = data["name"]
        script = data["script"]
        config = data["config"]
        slots = _extract_slots(config)
        profiles.append(
            ProfileDefinition(
                key=path.stem,
                name=name,
                script=script,
                config=config,
                slots=slots,
                file_path=path,
                file_hash=_hash_file(path),
            )
        )
    if not profiles:
        raise RuntimeError(f"No profile files were found in {directory}")
    return profiles


def profile_map(root_dir: Path) -> dict[str, ProfileDefinition]:
    return {profile.key: profile for profile in load_profiles(root_dir)}


def profiles_fingerprint(profiles: Iterable[ProfileDefinition]) -> str:
    digest = hashlib.sha256()
    for profile in profiles:
        digest.update(profile.key.encode("utf-8"))
        digest.update(profile.file_hash.encode("utf-8"))
    return digest.hexdigest()


def _toml_safe_value(value: str) -> str:
    normalized = value.replace("\\", "/")
    # Tokens are substituted inside TOML basic strings, so escape only the content.
    return json.dumps(normalized)[1:-1]


def replace_profile_tokens(config: str, slot_values: dict[str, str], builtins: dict[str, str]) -> str:
    def replace_slot(match: re.Match[str]) -> str:
        slot_name = match.group(1)
        return _toml_safe_value(slot_values[slot_name])

    def replace_builtin(match: re.Match[str]) -> str:
        builtin_name = match.group(1)
        return _toml_safe_value(builtins.get(builtin_name, match.group(0)))

    resolved = SLOT_PATTERN.sub(replace_slot, config)
    return BUILTIN_PATTERN.sub(replace_builtin, resolved)
