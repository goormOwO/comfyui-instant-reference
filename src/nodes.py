from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .profiles import ProfileDefinition, SlotSpec, load_profiles, profile_map, profiles_fingerprint, replace_profile_tokens
from .runtime import (
    ensure_dir,
    ensure_sd_scripts_environment,
    export_images,
    get_runtime_paths,
    hash_file,
    hash_tensor_batch,
    hash_text,
    latest_safetensors,
    read_caption_files,
    resolve_sd_scripts_file,
    run_command,
    venv_python,
    write_json,
)


def _read_project_version() -> str:
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', content, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find project version in {pyproject_path}")
    version = match.group(1)
    if version != "0.0.0":
        return version

    git_dir = pyproject_path.parent / ".git"
    if not git_dir.exists():
        return version

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pyproject_path.parent,
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception:
        return version

    return f"{version}+{git_sha}" if git_sha else version


NODE_VERSION = _read_project_version()
MAX_TRAIN_STEPS_PATTERN = re.compile(r"^\s*max_train_steps\s*=\s*(\d+)\s*$", re.MULTILINE)
MIXED_PRECISION_PATTERN = re.compile(r'^\s*mixed_precision\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


@io.comfytype(io_type="LORA_STACK")
class LoRAStack(io.ComfyTypeIO):
    Type = list[tuple[str, float, float]]


@dataclass(frozen=True)
class ResolvedSlot:
    name: str
    replacement: str
    fingerprint: str


@dataclass(frozen=True)
class TaggingOptions:
    general_threshold: float = 0.35
    character_threshold: float = 0.85
    prepend_tags: str = ""
    append_tags: str = ""
    exclude_tags: str = ""
    replace_tags: str = ""
    remove_underscore: bool = True


@dataclass(frozen=True)
class TrainOptions:
    steps_override: int = 0
    learning_rate_override: float = 0.0
    network_dim_override: int = 0
    network_alpha_override: int = 0
    resolution_override: str = ""
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_text_encoder_outputs: bool = True
    seed_override: int = -1
    force_retrain: bool = False


def _plugin_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _last_lora_info_path() -> Path:
    return get_runtime_paths().root / "last_lora.json"


def _hash_options(value: dict[str, Any]) -> str:
    return hash_text(json.dumps(value, sort_keys=True))


def _split_tags(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _tagging_options_from_input(value: Any | None) -> TaggingOptions:
    if not value:
        return TaggingOptions()
    return TaggingOptions(
        general_threshold=float(value.get("general_threshold", 0.35)),
        character_threshold=float(value.get("character_threshold", 0.85)),
        prepend_tags=str(value.get("prepend_tags", "")),
        append_tags=str(value.get("append_tags", "")),
        exclude_tags=str(value.get("exclude_tags", "")),
        replace_tags=str(value.get("replace_tags", "")),
        remove_underscore=bool(value.get("remove_underscore", True)),
    )


def _train_options_from_input(value: Any | None) -> TrainOptions:
    if not value:
        return TrainOptions()
    return TrainOptions(
        steps_override=int(value.get("steps_override", 0)),
        learning_rate_override=float(value.get("learning_rate_override", 0.0)),
        network_dim_override=int(value.get("network_dim_override", 0)),
        network_alpha_override=int(value.get("network_alpha_override", 0)),
        resolution_override=str(value.get("resolution_override", "")),
        gradient_checkpointing=bool(value.get("gradient_checkpointing", True)),
        cache_latents=bool(value.get("cache_latents", True)),
        cache_text_encoder_outputs=bool(value.get("cache_text_encoder_outputs", True)),
        seed_override=int(value.get("seed_override", -1)),
        force_retrain=bool(value.get("force_retrain", False)),
    )


def _tagging_options_fingerprint(options: TaggingOptions) -> str:
    return _hash_options(options.__dict__)


def _train_options_fingerprint(options: TrainOptions) -> str:
    return _hash_options(options.__dict__)


def _effective_max_train_steps(profile: ProfileDefinition, options: TrainOptions) -> int:
    if options.steps_override > 0:
        return options.steps_override
    match = MAX_TRAIN_STEPS_PATTERN.search(profile.config)
    if match:
        return max(1, int(match.group(1)))
    return 50


def _profile_choice_inputs(profile: ProfileDefinition) -> list[Any]:
    inputs: list[Any] = []
    for slot in profile.slots:
        if slot.slot_type in {"MODEL", "CLIP"}:
            continue
        if slot.slot_type == "STRING":
            inputs.append(io.String.Input(slot.name, multiline=False))
        elif slot.slot_type == "VAE":
            inputs.append(io.Vae.Input(slot.name))
    return inputs


def _profile_slots_by_type(profile: ProfileDefinition, slot_type: str) -> list[SlotSpec]:
    return [slot for slot in profile.slots if slot.slot_type == slot_type]


def _primary_profile_slot(profile: ProfileDefinition, slot_type: str) -> SlotSpec | None:
    slots = _profile_slots_by_type(profile, slot_type)
    return slots[0] if slots else None


def _recover_model_checkpoint_path(model: Any) -> str:
    cached = getattr(model, "cached_patcher_init", None)
    if not cached or len(cached) < 2 or not cached[1]:
        raise RuntimeError("This MODEL does not expose a recoverable checkpoint path. Load it with a checkpoint loader first.")
    checkpoint_path = cached[1][0]
    if not isinstance(checkpoint_path, str):
        raise RuntimeError("Recovered checkpoint path was not a string.")
    return checkpoint_path


def _recover_clip_paths(clip: Any) -> list[str]:
    patcher = getattr(clip, "patcher", None)
    cached = getattr(patcher, "cached_patcher_init", None)
    if not cached or len(cached) < 2 or not cached[1]:
        raise RuntimeError("This CLIP input does not expose recoverable checkpoint metadata.")
    ckpt_paths = cached[1][0]
    if isinstance(ckpt_paths, (list, tuple)):
        return [str(path) for path in ckpt_paths]
    raise RuntimeError("Recovered CLIP paths were not a list.")


def _resolve_string_slot(name: str, value: str) -> ResolvedSlot:
    return ResolvedSlot(name=name, replacement=value, fingerprint=hash_text(value))


def _resolve_model_slot(name: str, value: Any) -> ResolvedSlot:
    checkpoint_path = _recover_model_checkpoint_path(value)
    return ResolvedSlot(name=name, replacement=checkpoint_path, fingerprint=hash_text(checkpoint_path))


def _export_state_dict_artifact(state_dict: dict[str, Any], suffix: str) -> tuple[str, Path]:
    paths = get_runtime_paths()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=paths.artifacts) as handle:
        temp_path = Path(handle.name)
    comfy.utils.save_torch_file(state_dict, str(temp_path))
    return hash_file(temp_path), temp_path


def _resolve_clip_slot(name: str, value: Any, ephemeral_artifacts: list[Path]) -> ResolvedSlot:
    try:
        clip_paths = _recover_clip_paths(value)
        fingerprint = hash_text("|".join(clip_paths))
        return ResolvedSlot(name=name, replacement=clip_paths[0], fingerprint=fingerprint)
    except Exception:
        digest, exported_path = _export_state_dict_artifact(value.get_sd(), ".safetensors")
        ephemeral_artifacts.append(exported_path)
        return ResolvedSlot(name=name, replacement=str(exported_path), fingerprint=digest)


def _resolve_vae_slot(name: str, value: Any, ephemeral_artifacts: list[Path]) -> ResolvedSlot:
    digest, exported_path = _export_state_dict_artifact(value.get_sd(), ".safetensors")
    ephemeral_artifacts.append(exported_path)
    return ResolvedSlot(name=name, replacement=str(exported_path), fingerprint=digest)


def _resolve_slot(slot: SlotSpec, raw_value: Any, ephemeral_artifacts: list[Path]) -> ResolvedSlot:
    if slot.slot_type == "STRING":
        return _resolve_string_slot(slot.name, raw_value)
    if slot.slot_type == "MODEL":
        return _resolve_model_slot(slot.name, raw_value)
    if slot.slot_type == "CLIP":
        return _resolve_clip_slot(slot.name, raw_value, ephemeral_artifacts)
    if slot.slot_type == "VAE":
        return _resolve_vae_slot(slot.name, raw_value, ephemeral_artifacts)
    raise RuntimeError(f"Unsupported slot type: {slot.slot_type}")


def _cleanup_ephemeral_artifacts(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _tag_dataset(paths, dataset_dir: Path, log_path: Path, options: TaggingOptions) -> None:
    if any(dataset_dir.glob("*.txt")):
        return
    ensure_sd_scripts_environment(paths, log_path=log_path)
    python_path = venv_python(paths.venv)
    tagger_script = resolve_sd_scripts_file(paths, "tag_images_by_wd14_tagger.py")
    tagger_model_dir = paths.sd_scripts / "wd14_tagger_model"
    onnx_model_path = tagger_model_dir / "SmilingWolf_wd-v1-4-convnext-tagger-v2" / "model.onnx"
    command = [
        str(python_path),
        str(tagger_script),
        "--batch_size",
        "1",
        "--caption_extension",
        ".txt",
        "--general_threshold",
        str(options.general_threshold),
        "--character_threshold",
        str(options.character_threshold),
        "--model_dir",
        str(tagger_model_dir),
        "--onnx",
        "--recursive",
        str(dataset_dir),
    ]
    if options.remove_underscore:
        command.append("--remove_underscore")
    if options.exclude_tags.strip():
        command.extend(["--undesired_tags", options.exclude_tags])
    if options.replace_tags.strip():
        command.extend(["--tag_replacement", options.replace_tags])
    if not onnx_model_path.exists():
        command.append("--force_download")
    run_command(command, cwd=paths.sd_scripts, log_path=log_path)


def _apply_caption_options(dataset_dir: Path, options: TaggingOptions) -> dict[str, str]:
    captions = read_caption_files(dataset_dir)
    prepend_tags = _split_tags(options.prepend_tags)
    append_tags = _split_tags(options.append_tags)
    exclude_tags = set(_split_tags(options.exclude_tags))
    for path in sorted(dataset_dir.glob("*.txt")):
        tags = _split_tags(captions.get(path.stem, ""))
        filtered_tags = [tag for tag in tags if tag not in exclude_tags]
        final_tags: list[str] = []
        for tag in [*prepend_tags, *filtered_tags, *append_tags]:
            if tag and tag not in final_tags:
                final_tags.append(tag)
        path.write_text(", ".join(final_tags), encoding="utf-8")
    return read_caption_files(dataset_dir)


def _prepare_dataset(
    images: Any,
    log_path: Path,
    options: TaggingOptions,
    target_steps: int,
) -> tuple[Path, str, str, dict[str, str]]:
    paths = get_runtime_paths()
    image_hash = hash_tensor_batch(images)
    dataset_key = hash_text(f"{image_hash}|{_tagging_options_fingerprint(options)}|steps:{target_steps}")
    dataset_dir = ensure_dir(paths.datasets / dataset_key)
    image_count = images.shape[0] if hasattr(images, "shape") and len(images.shape) > 0 else 1
    repeat_count = max(1, -(-target_steps // max(1, int(image_count))))
    train_subset_dir = ensure_dir(dataset_dir / f"{repeat_count}_reference")
    export_images(images, train_subset_dir)
    _tag_dataset(paths, train_subset_dir, log_path=log_path, options=options)
    captions = _apply_caption_options(train_subset_dir, options)
    if not captions:
        raise RuntimeError("WD tagging did not produce any caption files.")
    caption_digest = hashlib.sha256()
    for key, value in sorted(captions.items()):
        caption_digest.update(key.encode("utf-8"))
        caption_digest.update(value.encode("utf-8"))
    return dataset_dir, image_hash, caption_digest.hexdigest(), captions


def _merge_run_log(temp_log: Path, final_log: Path) -> None:
    if not temp_log.exists():
        return
    ensure_dir(final_log.parent)
    content = temp_log.read_text(encoding="utf-8")
    with final_log.open("a", encoding="utf-8") as handle:
        handle.write(content)
    temp_log.unlink(missing_ok=True)
    temp_dir = temp_log.parent
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()


def _cache_key(
    checkpoint_path: str,
    profile: ProfileDefinition,
    image_hash: str,
    captions_hash: str,
    slots: dict[str, ResolvedSlot],
    tagging_options: TaggingOptions,
    train_options: TrainOptions,
) -> str:
    digest = hashlib.sha256()
    digest.update(NODE_VERSION.encode("utf-8"))
    digest.update(checkpoint_path.encode("utf-8"))
    digest.update(profile.file_hash.encode("utf-8"))
    digest.update(image_hash.encode("utf-8"))
    digest.update(captions_hash.encode("utf-8"))
    digest.update(_tagging_options_fingerprint(tagging_options).encode("utf-8"))
    digest.update(_train_options_fingerprint(train_options).encode("utf-8"))
    for name in sorted(slots):
        digest.update(name.encode("utf-8"))
        digest.update(slots[name].fingerprint.encode("utf-8"))
    return digest.hexdigest()


def _builtins_for_run(
    dataset_dir: Path,
    output_dir: Path,
    output_name: str,
) -> dict[str, str]:
    return {
        "TRAIN_DIR": str(dataset_dir),
        "OUTPUT_DIR": str(output_dir),
        "OUTPUT_NAME": output_name,
        "CAPTION_EXTENSION": ".txt",
    }


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    return json.dumps(str(value))


def _set_toml_key(config_text: str, key: str, value: Any) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    replacement = f"{key} = {_format_toml_value(value)}"
    if pattern.search(config_text):
        return pattern.sub(replacement, config_text, count=1)
    if not config_text.endswith("\n"):
        config_text += "\n"
    return config_text + replacement + "\n"


def _apply_train_options(config_text: str, options: TrainOptions) -> str:
    overrides: dict[str, Any] = {}
    if options.steps_override > 0:
        overrides["max_train_steps"] = options.steps_override
    if options.learning_rate_override > 0:
        overrides["learning_rate"] = options.learning_rate_override
    if options.network_dim_override > 0:
        overrides["network_dim"] = options.network_dim_override
    if options.network_alpha_override > 0:
        overrides["network_alpha"] = options.network_alpha_override
    if options.resolution_override.strip():
        overrides["resolution"] = options.resolution_override.strip()
    overrides["gradient_checkpointing"] = options.gradient_checkpointing
    overrides["cache_latents"] = options.cache_latents
    overrides["cache_text_encoder_outputs"] = options.cache_text_encoder_outputs
    if os.name == "nt":
        overrides["max_data_loader_n_workers"] = 0
        overrides["persistent_data_loader_workers"] = False
    if options.seed_override >= 0:
        overrides["seed"] = options.seed_override
    for key, value in overrides.items():
        config_text = _set_toml_key(config_text, key, value)
    return config_text


def _accelerate_mixed_precision(config_path: Path) -> str | None:
    match = MIXED_PRECISION_PATTERN.search(config_path.read_text(encoding="utf-8"))
    if not match:
        return None
    value = match.group(1).strip().lower()
    if value in {"fp16", "bf16", "fp8", "no"}:
        return value
    return None


def _write_resolved_config(
    profile: ProfileDefinition,
    slot_values: dict[str, ResolvedSlot],
    builtins: dict[str, str],
    run_dir: Path,
    train_options: TrainOptions,
) -> Path:
    config_path = run_dir / "config.toml"
    rendered = replace_profile_tokens(
        profile.config,
        {name: slot.replacement for name, slot in slot_values.items()},
        builtins,
    )
    rendered = _apply_train_options(rendered, train_options)
    config_path.write_text(rendered, encoding="utf-8")
    return config_path


def _run_training(profile: ProfileDefinition, run_dir: Path, output_dir: Path, config_path: Path, log_path: Path) -> Path:
    paths = get_runtime_paths()
    ensure_sd_scripts_environment(paths, log_path=log_path)
    python_path = venv_python(paths.venv)
    training_script = resolve_sd_scripts_file(paths, profile.script)
    command = [str(python_path)]
    if os.name == "nt":
        command.extend(
            [
                "-m",
                "accelerate.commands.launch",
                "--num_cpu_threads_per_process",
                "2",
            ]
        )
        mixed_precision = _accelerate_mixed_precision(config_path)
        if mixed_precision is not None:
            command.extend(["--mixed_precision", mixed_precision])
    command.extend([str(training_script), "--config_file", str(config_path)])
    run_command(command, cwd=paths.sd_scripts, log_path=log_path)
    trained_lora = latest_safetensors(output_dir)
    if trained_lora is None:
        raise RuntimeError(f"Training completed but no LoRA file was found in {output_dir}")
    return trained_lora


def _apply_lora(model: Any, clip: Any, lora_path: Path, model_strength: float, clip_strength: float) -> tuple[Any, Any]:
    lora = comfy.utils.load_torch_file(str(lora_path), safe_load=True)
    return comfy.sd.load_lora_for_models(model, clip, lora, model_strength, clip_strength)


def _record_last_lora(lora_path: Path) -> None:
    write_json(
        _last_lora_info_path(),
        {
            "path": str(lora_path),
        },
    )


def _ensure_lora_stack_entry(lora_path: Path, model_strength: float, clip_strength: float) -> list[tuple[str, float, float]]:
    lora_dirs = [Path(path) for path in folder_paths.get_folder_paths("loras")]
    for root in lora_dirs:
        try:
            relative = lora_path.resolve().relative_to(root.resolve())
            return [(relative.as_posix(), model_strength, clip_strength)]
        except ValueError:
            continue

    if not lora_dirs:
        raise RuntimeError("No ComfyUI LoRA directories are registered.")

    generated_dir = ensure_dir(lora_dirs[0] / "instant-reference-generated")
    target_path = generated_dir / lora_path.name
    if (
        not target_path.exists()
        or target_path.stat().st_size != lora_path.stat().st_size
        or target_path.stat().st_mtime < lora_path.stat().st_mtime
    ):
        shutil.copy2(lora_path, target_path)

    relative = target_path.relative_to(lora_dirs[0]).as_posix()
    return [(relative, model_strength, clip_strength)]


class InstantReferenceLoRA(io.ComfyNode):
    CATEGORY = "reference/training"

    @classmethod
    def define_schema(cls) -> io.Schema:
        profiles = load_profiles(_plugin_root())
        options = [
            io.DynamicCombo.Option(profile.key, _profile_choice_inputs(profile))
            for profile in profiles
        ]
        return io.Schema(
            node_id="InstantReferenceLoRA",
            display_name="Instant Reference LoRA",
            category=cls.CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Image.Input("images"),
                io.Float.Input("model_strength", default=1.0),
                io.Float.Input("clip_strength", default=1.0),
                io.DynamicCombo.Input("profile", options=options, display_name="profile"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.String.Output(display_name="lora_path"),
                LoRAStack.Output(display_name="lora_stack"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, model=None, clip=None, images=None, profile=None):
        profiles = load_profiles(_plugin_root())
        return profiles_fingerprint(profiles)

    @classmethod
    def execute(cls, model, clip, images, model_strength, clip_strength, profile, tagging_options=None, train_options=None) -> io.NodeOutput:
        return _execute_reference_lora(
            model,
            clip,
            images,
            profile,
            model_strength=model_strength,
            clip_strength=clip_strength,
            tagging_options=tagging_options,
            train_options=train_options,
        )


def _execute_reference_lora(model, clip, images, profile, model_strength=1.0, clip_strength=1.0, tagging_options=None, train_options=None) -> io.NodeOutput:
        ephemeral_artifacts: list[Path] = []
        try:
            profile_key = profile["profile"]
            selected_profile = profile_map(_plugin_root())[profile_key]
            resolved_tagging = _tagging_options_from_input(tagging_options)
            resolved_train = _train_options_from_input(train_options)
            target_steps = _effective_max_train_steps(selected_profile, resolved_train)
            model_slot = _primary_profile_slot(selected_profile, "MODEL")
            if model_slot is None:
                raise RuntimeError(f"Profile '{selected_profile.name}' must define a MODEL slot such as '{{{{model:MODEL}}}}'.")
            checkpoint_path = _recover_model_checkpoint_path(model)
            temp_image_hash = hash_tensor_batch(images)
            paths = get_runtime_paths()
            temp_run_dir = ensure_dir(paths.cache / f"_inflight_{temp_image_hash}")
            temp_run_log = temp_run_dir / "run.log"
            dataset_dir, image_hash, captions_hash, captions = _prepare_dataset(
                images,
                log_path=temp_run_log,
                options=resolved_tagging,
                target_steps=target_steps,
            )

            resolved_slots: dict[str, ResolvedSlot] = {}
            for slot in selected_profile.slots:
                if slot.slot_type == "MODEL":
                    resolved_slots[slot.name] = _resolve_slot(slot, model, ephemeral_artifacts)
                    continue
                if slot.slot_type == "CLIP":
                    resolved_slots[slot.name] = _resolve_slot(slot, clip, ephemeral_artifacts)
                    continue
                if slot.name not in profile:
                    raise RuntimeError(f"Profile '{selected_profile.name}' requires input '{slot.name}'.")
                resolved_slots[slot.name] = _resolve_slot(slot, profile[slot.name], ephemeral_artifacts)

            cache_key = _cache_key(
                checkpoint_path=checkpoint_path,
                profile=selected_profile,
                image_hash=image_hash,
                captions_hash=captions_hash,
                slots=resolved_slots,
                tagging_options=resolved_tagging,
                train_options=resolved_train,
            )

            run_dir = ensure_dir(paths.cache / cache_key)
            run_log = run_dir / "run.log"
            if temp_run_log != run_log:
                _merge_run_log(temp_run_log, run_log)
            output_dir = ensure_dir(paths.outputs / cache_key)
            output_name = f"instant_lora_{cache_key[:12]}"
            manifest = run_dir / "manifest.json"
            cached_lora = None if resolved_train.force_retrain else latest_safetensors(output_dir)

            if cached_lora is None:
                ensure_sd_scripts_environment(paths, log_path=run_log)
                builtins = _builtins_for_run(dataset_dir, output_dir, output_name)
                config_path = _write_resolved_config(selected_profile, resolved_slots, builtins, run_dir, resolved_train)
                write_json(
                    manifest,
                    {
                        "cache_key": cache_key,
                        "checkpoint_path": checkpoint_path,
                        "profile": selected_profile.key,
                        "profile_file": str(selected_profile.file_path),
                        "captions": captions,
                        "tagging_options": resolved_tagging.__dict__,
                        "train_options": resolved_train.__dict__,
                        "resolved_slots": {name: slot.replacement for name, slot in resolved_slots.items()},
                        "config_path": str(config_path),
                    },
                )
                comfy.model_management.unload_all_models()
                soft_empty_cache = getattr(comfy.model_management, "soft_empty_cache", None)
                if callable(soft_empty_cache):
                    soft_empty_cache()
                cached_lora = _run_training(selected_profile, run_dir, output_dir, config_path, log_path=run_log)

            patched_model, patched_clip = _apply_lora(
                model,
                clip,
                cached_lora,
                float(model_strength),
                float(clip_strength),
            )
            _record_last_lora(cached_lora)
            lora_stack = _ensure_lora_stack_entry(cached_lora, float(model_strength), float(clip_strength))
            return io.NodeOutput(patched_model, patched_clip, str(cached_lora), lora_stack)
        finally:
            _cleanup_ephemeral_artifacts(ephemeral_artifacts)


class ReferenceTrainingExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [InstantReferenceLoRA]


def _v1_slot_type(slot_type: str):
    return slot_type


def _all_optional_profile_inputs() -> dict[str, tuple]:
    # Keep V1 inputs stable so existing nodes do not accumulate stale profile-specific sockets.
    merged: dict[str, tuple] = {}
    for profile in load_profiles(_plugin_root()):
        for slot in profile.slots:
            if slot.slot_type in {"MODEL", "CLIP"}:
                continue
            existing = merged.get(slot.name)
            current = _v1_slot_type(slot.slot_type)
            if existing is not None and existing != (current,):
                raise RuntimeError(f"Profile input '{slot.name}' uses conflicting types across profiles.")
            merged[slot.name] = (current,)
    return merged


class InstantReferenceLoRAV1:
    CATEGORY = "reference/training"
    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "LORA_STACK")
    RETURN_NAMES = ("model", "clip", "lora_path", "lora_stack")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        profiles = load_profiles(_plugin_root())
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "images": ("IMAGE",),
                "model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "profile": ([profile.key for profile in profiles],),
            },
            "optional": {
                **_all_optional_profile_inputs(),
                "tagging_options": ("TAGGING_OPTIONS",),
                "train_options": ("TRAIN_OPTIONS",),
            },
        }

    def run(self, model, clip, images, model_strength, clip_strength, profile, tagging_options=None, train_options=None, **kwargs):
        payload = {"profile": profile}
        payload.update(kwargs)
        output = _execute_reference_lora(
            model,
            clip,
            images,
            payload,
            model_strength=model_strength,
            clip_strength=clip_strength,
            tagging_options=tagging_options,
            train_options=train_options,
        )
        return output.result


class TaggingOptionsV1:
    CATEGORY = "reference/training"
    RETURN_TYPES = ("TAGGING_OPTIONS",)
    RETURN_NAMES = ("tagging_options",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "general_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prepend_tags": ("STRING", {"default": "", "multiline": False}),
                "append_tags": ("STRING", {"default": "", "multiline": False}),
                "exclude_tags": ("STRING", {"default": "", "multiline": False}),
                "replace_tags": ("STRING", {"default": "", "multiline": False}),
                "remove_underscore": ("BOOLEAN", {"default": True}),
            }
        }

    def build(self, **kwargs):
        return (kwargs,)


class TrainOptionsV1:
    CATEGORY = "reference/training"
    RETURN_TYPES = ("TRAIN_OPTIONS",)
    RETURN_NAMES = ("train_options",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required": {
                    "steps_override": ("INT", {"default": 0, "min": 0, "max": 100000}),
                    "learning_rate_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001}),
                    "network_dim_override": ("INT", {"default": 0, "min": 0, "max": 1024}),
                    "network_alpha_override": ("INT", {"default": 0, "min": 0, "max": 1024}),
                    "resolution_override": ("STRING", {"default": "", "multiline": False}),
                    "gradient_checkpointing": ("BOOLEAN", {"default": True}),
                    "cache_latents": ("BOOLEAN", {"default": True}),
                    "cache_text_encoder_outputs": ("BOOLEAN", {"default": True}),
                    "seed_override": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                    "force_retrain": ("BOOLEAN", {"default": False}),
            }
        }

    def build(self, **kwargs):
        return (kwargs,)


NODE_CLASS_MAPPINGS = {
    "InstantReferenceLoRA": InstantReferenceLoRAV1,
    "ReferenceTaggingOptions": TaggingOptionsV1,
    "ReferenceTrainOptions": TrainOptionsV1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantReferenceLoRA": "Instant Reference LoRA",
    "ReferenceTaggingOptions": "Reference Tagging Options",
    "ReferenceTrainOptions": "Reference Train Options",
}
