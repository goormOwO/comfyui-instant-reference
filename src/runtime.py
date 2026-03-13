from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import signal
import shutil
import subprocess
import sys
import sysconfig
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import folder_paths
from PIL import Image

try:
    import comfy.model_management as comfy_model_management
except Exception:
    comfy_model_management = None

if os.name == "nt":
    import ctypes
    from ctypes import wintypes


SETUP_VERSION = "11"
SD_SCRIPTS_REPO = "https://github.com/kohya-ss/sd-scripts.git"
SD_SCRIPTS_COMMIT = "1a3ec9ea745fe9883551dfca5c947ea3d6aa68c7"


if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JobObjectExtendedLimitInformation = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def _create_windows_job() -> int | None:
    if os.name != "nt":
        return None

    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    result = _kernel32.SetInformationJobObject(
        job,
        _JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not result:
        error = ctypes.get_last_error()
        _kernel32.CloseHandle(job)
        raise OSError(error, "SetInformationJobObject failed")
    return job


def _assign_process_to_windows_job(process: subprocess.Popen, job_handle: int | None) -> None:
    if os.name != "nt" or job_handle is None:
        return

    access = _PROCESS_SET_QUOTA | _PROCESS_TERMINATE
    process_handle = _kernel32.OpenProcess(access, False, process.pid)
    if not process_handle:
        raise OSError(ctypes.get_last_error(), f"OpenProcess failed for pid {process.pid}")
    try:
        result = _kernel32.AssignProcessToJobObject(job_handle, process_handle)
        if not result:
            raise OSError(ctypes.get_last_error(), f"AssignProcessToJobObject failed for pid {process.pid}")
    finally:
        _kernel32.CloseHandle(process_handle)


def _close_windows_job(job_handle: int | None) -> None:
    if os.name != "nt" or job_handle is None:
        return
    _kernel32.CloseHandle(job_handle)


def plugin_root() -> Path:
    return Path(__file__).resolve().parent.parent


def runtime_root() -> Path:
    return plugin_root() / "runtime"


def uv_executable() -> str | None:
    env_candidates = [
        os.environ.get("COMFYUI_UV"),
        os.environ.get("COMFY_DESKTOP_UV"),
        os.environ.get("UV"),
    ]
    for candidate in env_candidates:
        if candidate and Path(candidate).exists():
            return candidate

    executable_path = Path(sys.executable).resolve()
    parent_candidates = [executable_path.parent, *executable_path.parents]
    relative_candidates = [
        Path("uv.exe"),
        Path("Scripts") / "uv.exe",
        Path("bin") / "uv",
        Path("uv") / "win" / "uv.exe",
        Path("resources") / "uv" / "win" / "uv.exe",
        Path("..") / "resources" / "uv" / "win" / "uv.exe",
        Path("..") / ".." / "resources" / "uv" / "win" / "uv.exe",
    ]
    for parent in parent_candidates:
        for relative in relative_candidates:
            candidate = (parent / relative).resolve()
            if candidate.exists():
                return str(candidate)

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidate = Path(local_appdata) / "Programs" / "ComfyUI" / "resources" / "uv" / "win" / "uv.exe"
        if candidate.exists():
            return str(candidate)

    scripts_dir = Path(sysconfig.get_path("scripts"))
    if os.name == "nt":
        scripts_candidate = scripts_dir / "uv.exe"
    else:
        scripts_candidate = scripts_dir / "uv"
    if scripts_candidate.exists():
        return str(scripts_candidate)

    return shutil.which("uv")


def ensure_uv(paths: RuntimePaths, log_path: Path | None = None) -> str:
    uv = uv_executable()
    if uv is not None:
        return uv

    run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "uv"],
        cwd=paths.root,
        log_path=log_path,
    )

    uv = uv_executable()
    if uv is None:
        raise RuntimeError("Failed to install uv with pip.")
    return uv


def runtime_project_dir() -> Path:
    return plugin_root() / "runtime_env"


def runtime_imports_ready(python_path: Path) -> bool:
    if not python_path.exists():
        return False
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    import_check = "import torch, torchvision"
    if os.name == "nt":
        import_check = "import torch, torchvision, xformers"
    try:
        result = subprocess.run(
            [str(python_path), "-c", import_check],
            cwd=str(plugin_root()),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    sd_scripts: Path
    venv: Path
    cache: Path
    datasets: Path
    outputs: Path
    generated_loras: Path
    artifacts: Path


def get_runtime_paths() -> RuntimePaths:
    root = ensure_dir(runtime_root())
    lora_dirs = folder_paths.get_folder_paths("loras")
    if not lora_dirs:
        raise RuntimeError("No ComfyUI LoRA directories are registered.")
    generated_loras = ensure_dir(Path(lora_dirs[0]) / "instant-reference-generated")
    return RuntimePaths(
        root=root,
        sd_scripts=root / "sd-scripts",
        venv=root / "venv",
        cache=ensure_dir(root / "cache"),
        datasets=ensure_dir(root / "datasets"),
        outputs=generated_loras,
        generated_loras=generated_loras,
        artifacts=ensure_dir(root / "artifacts"),
    )


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tensor_batch(images: Any) -> str:
    array = images.detach().cpu().numpy()
    normalized = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(str(normalized.shape).encode("utf-8"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def export_images(images: Any, target_dir: Path) -> list[Path]:
    ensure_dir(target_dir)
    exported: list[Path] = []
    image_batch = images.detach().cpu().numpy()
    for index, image in enumerate(image_batch):
        clipped = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        path = target_dir / f"image_{index:03d}.png"
        if not path.exists():
            Image.fromarray(clipped).save(path)
        exported.append(path)
    return exported


def _terminate_process_tree(process: subprocess.Popen, log_handle) -> None:
    if process.poll() is not None:
        return
    try:
        if log_handle is not None:
            log_handle.write(f"\n[interrupt] terminating process tree for pid {process.pid}\n")
            log_handle.flush()
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except Exception as exc:
        logging.warning("Failed to terminate child process tree for pid %s: %s", process.pid, exc)


def run_command(command: list[str], cwd: Path, log_path: Path | None = None, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    merged_env.setdefault("PYTHONUTF8", "1")
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")
    merged_env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    if os.name == "nt":
        merged_env.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    if env:
        merged_env.update(env)
    joined_command = " ".join(command)
    log_handle = None
    process: subprocess.Popen | None = None
    job_handle: int | None = None
    recent_output: list[str] = []

    try:
        if log_path is not None:
            ensure_dir(log_path.parent)
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(f"$ {joined_command}\n")
            log_handle.flush()

        creationflags = 0
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        if os.name == "nt":
            job_handle = _create_windows_job()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
            **popen_kwargs,
        )
        _assign_process_to_windows_job(process, job_handle)

        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def _reader_thread() -> None:
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=_reader_thread, daemon=True)
        reader.start()

        while True:
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                line = None

            if comfy_model_management is not None:
                try:
                    comfy_model_management.throw_exception_if_processing_interrupted()
                except Exception as exc:
                    if log_handle is not None:
                        log_handle.write(f"\n[interrupt] comfy processing interrupted: {exc.__class__.__name__}\n")
                        log_handle.flush()
                    raise

            if line is None:
                if process.poll() is not None and not reader.is_alive() and output_queue.empty():
                    break
                continue

            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()
            text = line.rstrip()
            if text:
                logging.info(text)
                recent_output.append(text)
                if len(recent_output) > 80:
                    recent_output = recent_output[-80:]

        return_code = process.wait()
        if log_handle is not None:
            log_handle.write(f"\n[exit_code] {return_code}\n")
            log_handle.flush()

        if return_code != 0:
            tail = "\n".join(recent_output)[-4000:]
            raise RuntimeError(f"Command failed with exit code {return_code}: {joined_command}\n{tail}")
    except BaseException:
        if process is not None:
            _terminate_process_tree(process, log_handle)
        raise
    finally:
        _close_windows_job(job_handle)
        if log_handle is not None:
            log_handle.close()


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_sd_scripts_checkout(paths: RuntimePaths, log_path: Path | None = None) -> None:
    if not paths.sd_scripts.exists():
        run_command(
            ["git", "clone", SD_SCRIPTS_REPO, str(paths.sd_scripts)],
            cwd=paths.root,
            log_path=log_path,
        )


def ensure_sd_scripts_environment(paths: RuntimePaths, log_path: Path | None = None) -> None:
    ensure_sd_scripts_checkout(paths, log_path=log_path)

    python_path = venv_python(paths.venv)
    uv = ensure_uv(paths, log_path=log_path)
    if not python_path.exists():
        run_command(
            [uv, "venv", "--python", sys.executable, "--system-site-packages", str(paths.venv)],
            cwd=paths.root,
            log_path=log_path,
        )

    marker = paths.venv / ".sd_scripts_ready"
    if marker.exists():
        if marker.read_text(encoding="utf-8").strip() == SETUP_VERSION and runtime_imports_ready(python_path):
            return

    sync_env = os.environ.copy()
    sync_env.setdefault("PYTHONUTF8", "1")
    sync_env.setdefault("PYTHONIOENCODING", "utf-8")
    sync_env["VIRTUAL_ENV"] = str(paths.venv)
    scripts_dir = python_path.parent
    sync_env["PATH"] = str(scripts_dir) + os.pathsep + sync_env.get("PATH", "")
    run_command(
        [
            uv,
            "sync",
            "--active",
            "--project",
            str(runtime_project_dir()),
            "--no-install-project",
        ],
        cwd=runtime_project_dir(),
        log_path=log_path,
        env=sync_env,
    )

    if not runtime_imports_ready(python_path):
        raise RuntimeError(f"Managed runtime is missing required packages in {paths.venv}")

    marker.write_text(f"{SETUP_VERSION}\n", encoding="utf-8")


def resolve_sd_scripts_file(paths: RuntimePaths, relative_script: str) -> Path:
    direct = paths.sd_scripts / relative_script
    if direct.exists():
        return direct
    finetune = paths.sd_scripts / "finetune" / relative_script
    if finetune.exists():
        return finetune
    raise FileNotFoundError(f"Could not find sd-scripts file '{relative_script}' in {paths.sd_scripts}")


def read_caption_files(dataset_dir: Path) -> dict[str, str]:
    captions: dict[str, str] = {}
    for path in sorted(dataset_dir.glob("*.txt")):
        captions[path.stem] = path.read_text(encoding="utf-8").strip()
    return captions


def latest_safetensors(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def stable_artifact_path(artifacts_dir: Path, digest: str, suffix: str) -> Path:
    return artifacts_dir / f"{digest}{suffix}"


def move_into_artifacts(temp_path: Path, artifacts_dir: Path, suffix: str) -> tuple[str, Path]:
    digest = hash_file(temp_path)
    final_path = stable_artifact_path(artifacts_dir, digest, suffix)
    if not final_path.exists():
        shutil.move(str(temp_path), str(final_path))
    else:
        temp_path.unlink(missing_ok=True)
    return digest, final_path
