from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re


TEMP_ROOT = Path(__file__).resolve().parents[2] / "dev_temp" / "tensorrt"
PRECISIONS = ("fp32", "fp16", "bf16", "fp8", "nvfp4")


class ManifestError(RuntimeError):
    pass


def source_name(value: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\\:\0"):
        raise ManifestError(f"Invalid source filename: {value!r}")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ManifestError(f"Source filename must be relative to its ComfyUI model folder: {value}")
    return value


def validate_chain(chain: dict) -> dict:
    if not isinstance(chain, dict) or set(chain) != {"base_model", "loras"} or not isinstance(chain["loras"], list):
        raise ManifestError("Expected a base_model and an ordered loras list.")
    source_name(chain["base_model"])
    for item in chain["loras"]:
        if not isinstance(item, dict) or set(item) != {"name", "strength_model", "strength_clip"}:
            raise ManifestError("Each LoRA must contain name, strength_model, and strength_clip.")
        source_name(item["name"])
        for key in ("strength_model", "strength_clip"):
            if isinstance(item[key], bool) or not isinstance(item[key], (int, float)) or not math.isfinite(item[key]):
                raise ManifestError(f"Invalid LoRA {key}: {item[key]!r}")
    return chain


def _slug(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(name).stem).strip("._")[:80] or "model"


def package_name(chain: dict) -> str:
    validate_chain(chain)
    return _slug(chain["base_model"])


def engine_name(chain: dict, width: int, height: int) -> str:
    return f"{_slug(chain['base_model'])}_{width}x{height}.engine"


@contextmanager
def package_lock(root):
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    locks = TEMP_ROOT / "locks" / str(Path(root).resolve()).lstrip("/").replace(":", "_")
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / "active"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ManifestError(f"TensorRT package is busy: {root}. If its process exited, remove stale lock {lock}.") from exc
    try:
        yield
    finally:
        lock.rmdir()


@dataclass(frozen=True)
class EngineSpec:
    path: Path
    image_size: tuple[int, int]
    precision: str
    tensorrt_version: str
    gpu_name: str
    compute_capability: tuple[int, int]

    @property
    def latent_shape(self):
        return (1, 4, self.image_size[1] // 8, self.image_size[0] // 8)


@dataclass(frozen=True)
class PackageManifest:
    root: Path
    chain: dict
    engines: tuple[EngineSpec, ...]


def read_manifest(root: str | Path) -> dict:
    try:
        data = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read TensorRT manifest in {root}: {exc}") from exc
    if not isinstance(data, dict) or data.get("format_version") != 2:
        raise ManifestError("Unsupported TensorRT manifest; rebuild old merged-checkpoint packages using format 2.")
    validate_chain(data.get("source"))
    if not isinstance(data.get("engines"), list) or not data["engines"]:
        raise ManifestError("TensorRT package contains no engines.")
    return data


def load_manifest(root) -> PackageManifest:
    root = Path(root).resolve()
    with package_lock(root):
        data = read_manifest(root)
        engines = []
        sizes = set()
        for item in data.get("engines", []):
            if not isinstance(item, dict) or not isinstance(item.get("image_size"), list):
                raise ManifestError("Invalid TensorRT engine entry.")
            size = tuple(item.get("image_size", ()))
            if len(size) != 2 or any(type(v) is not int or v <= 0 or v % 8 for v in size) or size in sizes:
                raise ManifestError("Invalid or duplicate TensorRT image resolution.")
            sizes.add(size)
            expected = "engines/" + engine_name(data["source"], *size)
            if not isinstance(item.get("path"), str):
                raise ManifestError("Invalid TensorRT engine path.")
            path = (root / item["path"]).resolve()
            if item.get("path") != expected or path.parent != (root / "engines").resolve() or root not in path.parents or not path.is_file():
                raise ManifestError(f"Missing or unsafe engine path: {item.get('path')}")
            requirements = item.get("requirements", {})
            if not isinstance(requirements, dict):
                raise ManifestError("Invalid TensorRT runtime requirements.")
            raw_capability = requirements.get("compute_capability")
            if not isinstance(raw_capability, list):
                raise ManifestError("Invalid engine compute capability.")
            capability = tuple(raw_capability)
            version = requirements.get("tensorrt_version", "")
            gpu = requirements.get("gpu_name", "")
            if item.get("precision") not in PRECISIONS or len(capability) != 2 or not all(type(v) is int for v in capability) or not isinstance(version, str) or not version.startswith("11.") or not isinstance(gpu, str) or not gpu:
                raise ManifestError("Invalid engine precision or runtime requirements.")
            engines.append(EngineSpec(path, size, item["precision"], version, gpu, capability))
        if not engines:
            raise ManifestError("TensorRT package contains no engines.")
        return PackageManifest(root, data["source"], tuple(engines))


def publish_engine(root, chain, candidate, image_size, precision, requirements, staging):
    validate_chain(chain)
    root, staging = Path(root).resolve(), Path(staging)
    with package_lock(root):
        previous = read_manifest(root) if (root / "manifest.json").exists() else None
        if previous is not None and previous["source"] != chain:
            raise ManifestError(f"Package naming collision with a different model chain: {root}")
        if root.exists() and previous is None:
            raise ManifestError(f"Refusing to write into an incomplete package: {root}")
        relative = "engines/" + engine_name(chain, *image_size)
        engine = root / relative
        if engine.resolve().parent != (root / "engines").resolve() or root not in engine.resolve().parents:
            raise ManifestError("Engine directory escapes its package.")
        data = {"format_version": 2, "source": chain, "engines": [] if previous is None else [
            item for item in previous["engines"] if item["image_size"] != list(image_size)
        ]}
        data["engines"].append({"path": relative, "image_size": list(image_size), "precision": precision, "requirements": requirements})
        data["engines"].sort(key=lambda item: item["image_size"])
        manifest = staging / "manifest.json"
        manifest.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        backup_engine = staging / "previous.engine"
        backup_manifest = staging / "previous.json"
        if engine.exists():
            os.link(engine, backup_engine)
        if previous is not None:
            os.link(root / "manifest.json", backup_manifest)
        for path in (candidate, manifest):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        (root / "engines").mkdir(parents=True, exist_ok=True)
        try:
            # Temp storage and the model directory must share a filesystem for atomic replacement.
            os.replace(candidate, engine)
            os.replace(manifest, root / "manifest.json")
        except BaseException:
            if backup_engine.exists():
                os.replace(backup_engine, engine)
            elif not candidate.exists():
                engine.unlink(missing_ok=True)
            if backup_manifest.exists():
                os.replace(backup_manifest, root / "manifest.json")
            elif previous is None:
                (root / "manifest.json").unlink(missing_ok=True)
            if previous is None:
                (root / "engines").rmdir()
                root.rmdir()
            raise
