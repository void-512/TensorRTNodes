from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineSpec:
    path: Path
    image_size: tuple[int, int]
    latent_shape: tuple[int, int, int, int]
    context_min: int
    context_opt: int
    context_max: int
    bindings: tuple[dict, ...]


@dataclass(frozen=True)
class PackageManifest:
    root: Path
    checkpoint: Path
    engines: tuple[EngineSpec, ...]
    tensorrt_major: int
    compute_capability_min: tuple[int, int]


def _package_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path.parent != root and root not in path.parents:
        raise ManifestError(f"Package path escapes its root: {relative}")
    if not path.is_file():
        raise ManifestError(f"Package artifact is missing: {relative}")
    return path


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_manifest(root: str | Path, verify_integrity: bool = False) -> PackageManifest:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ManifestError(f"TensorRT package has no manifest.json: {root}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"TensorRT package manifest is invalid JSON: {manifest_path}") from exc
    if data.get("format_version") != 1:
        raise ManifestError("Unsupported TensorRT package manifest version.")
    checkpoint_data = data.get("checkpoint", {})
    checkpoint = _package_file(root, checkpoint_data.get("path", ""))
    if verify_integrity:
        expected = checkpoint_data.get("sha256")
        if not expected or _digest(checkpoint) != expected:
            raise ManifestError(f"Checkpoint integrity check failed: {checkpoint.name}")
    engines = []
    for item in data.get("engines", []):
        context_profile = item.get("context_profile", {})
        latent_shape = tuple(item.get("latent_shape", ()))
        image_size = tuple(item.get("image_size", ()))
        if (
            len(latent_shape) != 4
            or len(image_size) != 2
            or latent_shape[0:2] != (1, 4)
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in latent_shape + image_size)
            or (latent_shape[3] * 8, latent_shape[2] * 8) != image_size
        ):
            raise ManifestError("TensorRT package contains an invalid fixed-resolution engine interface.")
        profile_values = tuple(context_profile.get(key) for key in ("min", "opt", "max")) if isinstance(context_profile, dict) else ()
        if (
            len(profile_values) != 3
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in profile_values)
            or not profile_values[0] <= profile_values[1] <= profile_values[2]
        ):
            raise ManifestError("TensorRT package has an invalid context optimization profile.")
        engine = _package_file(root, item.get("path", ""))
        if verify_integrity:
            expected = item.get("sha256")
            if not expected or _digest(engine) != expected:
                raise ManifestError(f"TensorRT engine integrity check failed: {engine.name}")
        engines.append(EngineSpec(
            engine, image_size, latent_shape,
            profile_values[0], profile_values[1], profile_values[2], tuple(item.get("bindings", [])),
        ))
    if not engines:
        raise ManifestError("TensorRT package manifest contains no engines.")
    requirements = data.get("requirements", {})
    capability = tuple(requirements.get("compute_capability_min", ()))
    if len(capability) != 2:
        raise ManifestError("TensorRT package has no valid compute-capability requirement.")
    return PackageManifest(root, checkpoint, tuple(engines), int(requirements.get("tensorrt_major", 0)), capability)
