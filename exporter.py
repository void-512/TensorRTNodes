"""One-shot ComfyUI node that packages an already-fused SDXL model for TensorRTLoader.

The node intentionally receives a MODEL, CLIP, and VAE from the workflow instead
of a checkpoint path.  This lets normal ComfyUI nodes apply LoRAs first, then
materializes that exact state into the loader package.  It does not parse workflow
JSON or load/apply LoRA files itself.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import folder_paths
import torch


_CONTEXT_MIN = 77
_PACKAGE_ROOT = Path(folder_paths.models_dir) / "tensorrt"


class _ExportUNet(torch.nn.Module):
    """Raw ComfyUI UNet call; sampler policy remains outside the engine."""

    def __init__(self, unet: torch.nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, adm: torch.Tensor) -> torch.Tensor:
        return self.unet(latent, timestep, context=context, y=adm, transformer_options={})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_package_path(package_name: str) -> Path:
    if not package_name or Path(package_name).name != package_name or package_name in {".", ".."}:
        raise RuntimeError("package_name must be a single directory name.")
    root = _PACKAGE_ROOT.resolve()
    package = (root / package_name).resolve()
    if package.parent != root:
        raise RuntimeError("Package directory must be directly under ComfyUI/models/tensorrt.")
    return package


def _build_engine(
    onnx: Path, engine: Path, report: Path, workspace_mib: int,
    context_profile: dict[str, int], latent_shape: tuple[int, int, int, int], builder_optimization_level: int,
) -> dict[str, Any]:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT Python bindings are required to export an engine.") from exc

    class Logger(trt.ILogger):
        def __init__(self) -> None:
            trt.ILogger.__init__(self)
            self.messages: list[str] = []

        def log(self, severity: Any, message: str) -> None:
            self.messages.append(f"{severity}: {message}")

    logger = Logger()
    builder = trt.Builder(logger)
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network = builder.create_network(0 if explicit_batch is None else 1 << int(explicit_batch))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        details = logger.messages + [str(parser.get_error(index)) for index in range(parser.num_errors)]
        report.write_text("\n".join(details) + "\n", encoding="utf-8")
        raise RuntimeError(f"TensorRT could not parse {onnx.name}; see {report.name}.")
    if not getattr(builder, "platform_has_fast_fp16", True):
        raise RuntimeError("This GPU/TensorRT configuration does not report fast FP16 support.")
    config = builder.create_builder_config()
    if not 0 <= builder_optimization_level <= 5:
        raise RuntimeError("builder_optimization_level must be between 0 and 5.")
    if not hasattr(config, "builder_optimization_level"):
        raise RuntimeError("Installed TensorRT does not expose builder_optimization_level.")
    # NVIDIA defines level 3 as the default.  Set it explicitly as well, so all
    # UI choices follow one builder path while preserving current behavior.
    config.builder_optimization_level = builder_optimization_level
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    if fp16_flag is not None:
        config.set_flag(fp16_flag)
    context = next((network.get_input(index) for index in range(network.num_inputs) if network.get_input(index).name == "context"), None)
    latent = next((network.get_input(index) for index in range(network.num_inputs) if network.get_input(index).name == "latent"), None)
    if context is None or tuple(context.shape) != (1, -1, 2048) or latent is None or tuple(latent.shape) != latent_shape:
        raise RuntimeError("Exported ONNX must expose the requested fixed latent shape and dynamic context [1, tokens, 2048].")
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "context", (1, context_profile["min"], 2048), (1, context_profile["opt"], 2048),
        (1, context_profile["max"], 2048),
    )
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        report.write_text("\n".join(logger.messages) + "\n", encoding="utf-8")
        raise RuntimeError(f"TensorRT could not build {engine.name}; see {report.name}.")
    runtime = trt.Runtime(logger)
    built = runtime.deserialize_cuda_engine(bytes(serialized))
    if built is None:
        raise RuntimeError("TensorRT built an engine that failed immediate deserialization.")
    bindings = []
    for index in range(built.num_io_tensors):
        name = built.get_tensor_name(index)
        bindings.append({
            "name": name,
            "mode": str(built.get_tensor_mode(name)),
            "shape": list(built.get_tensor_shape(name)),
            "dtype": str(built.get_tensor_dtype(name)),
        })
    engine.write_bytes(bytes(serialized))
    result = {
        "status": "pass", "engine": engine.name, "tensorrt_version": trt.__version__,
        "engine_size_bytes": engine.stat().st_size,
        "precision": "fp16 (TensorRT 11 network-inferred)" if fp16_flag is None else "fp16 (builder flag)",
        "workspace_mib": workspace_mib, "builder_optimization_level": builder_optimization_level,
        "context_profile": context_profile, "latent_shape": list(latent_shape), "bindings": bindings,
    }
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _export_onnx(unet: torch.nn.Module, output: Path, width: int, height: int, max_context_length: int) -> None:
    device = torch.device("cuda")
    wrapper = _ExportUNet(unet.eval().to(device=device, dtype=torch.float16)).eval()
    latent = torch.randn((1, 4, height // 8, width // 8), device=device, dtype=torch.float16)
    timestep = torch.full((1,), 500.0, device=device, dtype=torch.float32)
    context = torch.randn((1, max_context_length, 2048), device=device, dtype=torch.float16)
    adm = torch.randn((1, 2816), device=device, dtype=torch.float16)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (latent, timestep, context, adm), str(output),
            input_names=["latent", "timestep", "context", "adm"], output_names=["noise_pred"],
            opset_version=18, dynamo=False, external_data=True,
            dynamic_axes={"context": {1: "context_length"}}, do_constant_folding=True,
        )
    del wrapper, latent, timestep, context, adm
    torch.cuda.empty_cache()


def _export_materialized_checkpoint(checkpoint: Path, output: Path, width: int, height: int, max_context_length: int) -> None:
    """Trace the checkpoint that TensorRTLoader will later load.

    A DynamicVRAM ModelPatcher can carry LoRAs as deferred weight functions rather
    than physical UNet parameters.  Exporting its raw ``diffusion_model`` risks
    tracing the unpatched weights.  Reloading the checkpoint materialized above
    makes the ONNX graph and the package checkpoint one identical fused model.
    """
    import comfy.sd

    exported_model, _clip, _vae, _ = comfy.sd.load_checkpoint_guess_config(
        str(checkpoint), output_vae=False, output_clip=False, embedding_directory=[], disable_dynamic=True,
    )
    if exported_model is None:
        raise RuntimeError(f"Materialized checkpoint did not provide a diffusion MODEL: {checkpoint}")
    try:
        exported_model.patch_model(device_to=torch.device("cuda"), force_patch_weights=True)
        _export_onnx(exported_model.model.diffusion_model, output, width, height, max_context_length)
    finally:
        exported_model.unpatch_model(device_to=torch.device("cpu"))
        del exported_model
        gc.collect()
        torch.cuda.empty_cache()


def _materialize_cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Resolve ComfyUI DynamicVRAM lazy tensors before state-dict conversion.

    ModelPatcherDynamic exposes deferred values with a FakeDevice sentinel.  Most
    checkpoint writers can call their ``.to('cpu')`` implementation lazily, but
    SDXL's CLIP key converter concatenates Q/K/V tensors first and therefore needs
    genuine torch tensors with a genuine CPU device at that point.
    """
    resolved: dict[str, Any] = {}
    for name, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            resolved[name] = value.to(device="cpu").detach().contiguous()
        else:
            resolved[name] = value
    return resolved


def _update_tensor_fingerprint(digest: Any, name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().to(device="cpu").contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())


def _unet_fingerprint(state_dict: dict[str, Any]) -> str:
    """Stable hash of the fused UNet, independent of checkpoint serialization."""
    digest = hashlib.sha256()
    keys = sorted(key for key, value in state_dict.items() if key.startswith("model.diffusion_model.") and isinstance(value, torch.Tensor))
    if not keys:
        raise RuntimeError("Could not find SDXL diffusion-model tensors while creating the package.")
    for key in keys:
        _update_tensor_fingerprint(digest, key, state_dict[key])
    return digest.hexdigest()


def _checkpoint_unet_fingerprint(checkpoint: Path) -> str:
    from safetensors import safe_open

    digest = hashlib.sha256()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as archive:
        keys = sorted(key for key in archive.keys() if key.startswith("model.diffusion_model."))
        if not keys:
            raise RuntimeError(f"Existing package checkpoint has no SDXL diffusion-model tensors: {checkpoint}")
        for key in keys:
            _update_tensor_fingerprint(digest, key, archive.get_tensor(key))
    return digest.hexdigest()


def _materialize_and_write_checkpoint(model: Any, clip: Any, vae: Any, checkpoint: Path) -> str:
    """Materialize a Dynamic-VRAM graph without directly patching its inputs.

    The Dynamic VRAM patcher rejects direct ``patch_model`` calls.  SDXL's CLIP
    checkpoint converter also cannot concatenate deferred FakeDevice tensors,
    so every state-dictionary tensor is resolved to a real CPU tensor before
    the normal ComfyUI model-config conversion runs.
    """
    import comfy.model_management
    import comfy.utils

    managed_models = [model, clip.load_model()]
    vae_patcher = getattr(vae, "patcher", None)
    if vae_patcher is not None:
        managed_models.append(vae_patcher)
    comfy.model_management.load_models_gpu(managed_models)

    unet_state = _materialize_cpu_state_dict(
        model.model_state_dict_for_saving(model.model.diffusion_model, "diffusion_model.")
    )
    clip_state = _materialize_cpu_state_dict(clip.state_dict_for_saving())
    vae_state = _materialize_cpu_state_dict(vae.get_sd())
    state_dict = _materialize_cpu_state_dict(
        model.model.state_dict_for_saving(unet_state, clip_state_dict=clip_state, vae_state_dict=vae_state)
    )
    unet_fingerprint = _unet_fingerprint(state_dict)
    comfy.utils.save_torch_file(state_dict, str(checkpoint), {
        "format": "pt", "modelspec.architecture": "stable-diffusion-xl-v1-base",
        "trt_exporter.source": "ComfyUI MODEL, CLIP, and VAE inputs",
        "trt_exporter.loras": "already applied by the input graph",
        "trt_exporter.materialization": "ComfyUI model manager + resolved CPU tensors",
    })
    # The caller traces this serialized checkpoint, not the deferred input patcher.
    del state_dict
    return unet_fingerprint


class ExportTensorRTPackage:
    """Build fixed-resolution loader-compatible engines from an already-fused graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MODEL after all desired LoRA/model patches have been applied."}),
                "clip": ("CLIP", {"tooltip": "Matching CLIP after all desired CLIP LoRA patches have been applied."}),
                "vae": ("VAE", {"tooltip": "Matching VAE, saved so Load TensorRT Engine remains a normal checkpoint loader."}),
                "package_name": ("STRING", {"default": "sdxl_tensorrt", "multiline": False}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1536, "min": 64, "max": 16384, "step": 8}),
                "maximum_context_length": ("INT", {"default": 462, "min": 77, "max": 2048, "step": 1}),
                "optimization_context_length": ("INT", {"default": 308, "min": 77, "max": 2048, "step": 1}),
                "workspace_mib": ("INT", {"default": 1024, "min": 64, "max": 32768, "step": 64}),
                "builder_optimization_level": ("INT", {"default": 3, "min": 0, "max": 5, "step": 1,
                                                       "tooltip": "NVIDIA TensorRT tactic-search level. Level 3 is the TensorRT default."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT/conversion"
    DESCRIPTION = "Builds one fixed-resolution FP16 TensorRT engine from already-fused MODEL, CLIP, and VAE inputs."

    def convert(
        self, model, clip, vae, package_name: str, width: int, height: int,
        maximum_context_length: int, optimization_context_length: int, workspace_mib: int, builder_optimization_level: int,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT package export requires CUDA.")
        if width % 8 or height % 8:
            raise RuntimeError("Width and height must be divisible by 8.")
        context_profile = {"min": _CONTEXT_MIN, "opt": optimization_context_length, "max": maximum_context_length}
        if not _CONTEXT_MIN <= optimization_context_length <= maximum_context_length:
            raise RuntimeError("Context lengths must satisfy 77 <= optimization_context_length <= maximum_context_length.")
        latent_shape = (1, 4, height // 8, width // 8)
        package = _safe_package_path(package_name)
        existing_manifest: dict[str, Any] | None = None
        new_package = not package.exists()
        if package.exists():
            if not package.is_dir() or not (package / "manifest.json").is_file():
                raise RuntimeError(f"Existing package is incomplete: {package}. Choose a different package_name.")
            try:
                existing_manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Existing package manifest is invalid: {package / 'manifest.json'}") from exc
            if existing_manifest.get("format_version") != 1:
                raise RuntimeError(
                    "Existing package uses the removed dynamic-size format. "
                    "Choose a new package_name or remove that package before rebuilding."
                )
        else:
            (package / "engines").mkdir(parents=True)
            (package / "metadata").mkdir()

        # Clones preserve the graph inputs while allowing this side-effect node to
        # materialize their patches for checkpoint serialization and ONNX export.
        fused_model, fused_clip = model.clone(), clip.clone()
        checkpoint = package / "merged_checkpoint.safetensors"
        stem = f"{width}x{height}_ctxdyn{maximum_context_length}_b1"
        staging = package / f".build_{stem}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        incoming_checkpoint = staging / "merged_checkpoint.safetensors"
        candidate_engine = staging / f"model_{stem}.engine"
        candidate_report = staging / f"tensorrt_{stem}.json"
        report: dict[str, Any] | None = None
        try:
            unet_fingerprint = _materialize_and_write_checkpoint(fused_model, fused_clip, vae, incoming_checkpoint)
            if existing_manifest is None:
                incoming_checkpoint.replace(checkpoint)
            elif unet_fingerprint != _checkpoint_unet_fingerprint(checkpoint):
                raise RuntimeError(
                    "The supplied fused UNet differs from the existing package. "
                    "Use a new package_name; mixing engines from different models is unsafe."
                )
            else:
                incoming_checkpoint.unlink()
            onnx = staging / f"model_{stem}.onnx"
            _export_materialized_checkpoint(checkpoint, onnx, width, height, maximum_context_length)
            report = _build_engine(
                onnx, candidate_engine, candidate_report, workspace_mib, context_profile, latent_shape,
                builder_optimization_level,
            )
            if existing_manifest is not None:
                existing_major = existing_manifest.get("requirements", {}).get("tensorrt_major")
                current_major = int(str(report["tensorrt_version"]).split(".", 1)[0])
                if existing_major != current_major:
                    raise RuntimeError("Existing package was built with a different TensorRT major version.")
        except Exception:
            if new_package:
                # Avoid leaving an unusable package name that would block a retry.
                shutil.rmtree(package)
            # Existing packages retain their manifest and prior engines; their staging
            # directory keeps TensorRT parser/build diagnostics for inspection.
            raise
        finally:
            try:
                fused_model.unpatch_model(device_to=torch.device("cpu"))
            except Exception:
                pass
            try:
                fused_clip.patcher.unpatch_model(device_to=torch.device("cpu"))
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        assert report is not None
        engine = package / "engines" / candidate_engine.name
        current_engine = {
            "path": engine.relative_to(package).as_posix(), "sha256": _sha256(candidate_engine),
            "size_bytes": candidate_engine.stat().st_size, "precision": report["precision"],
            "image_size": [width, height], "latent_shape": list(latent_shape),
            "context_profile": context_profile, "bindings": report["bindings"],
            "tensorrt_version": report["tensorrt_version"],
        }
        prior_engines = [] if existing_manifest is None else existing_manifest.get("engines", [])
        previous_engine_paths = [
            item.get("path", "") for item in prior_engines
            if tuple(item.get("image_size", ())) == (width, height)
        ]
        engines = [
            item for item in prior_engines
            if tuple(item.get("image_size", ())) != (width, height)
        ]
        engines.append(current_engine)
        manifest = existing_manifest or {
            "format_version": 1, "model_family": "SDXL / Illustrious",
            "checkpoint": {"path": checkpoint.name, "sha256": _sha256(checkpoint), "size_bytes": checkpoint.stat().st_size},
            "requirements": {
                "cuda": "runtime CUDA required", "tensorrt_major": int(str(report["tensorrt_version"]).split(".", 1)[0]),
                "compute_capability_min": list(torch.cuda.get_device_capability()), "engine_batch_sizes": [1],
                "runtime_batching": "ComfyUI loader may split independent batches into B1 engine calls.",
            },
            "precision": {"fp16": True, "fp8": False},
            "failure_policy": "Unsupported latent size, token count, TensorRT major version, GPU capability, or model feature is an error.",
        }
        manifest["format_version"] = 1
        manifest["engines"] = engines
        result = {
            "status": "pass", "package": str(package), "engines": [entry["path"] for entry in engines],
            "context_profile": context_profile, "input_policy": "MODEL, CLIP, and VAE were materialized from the input graph; no LoRA files were read.",
        }
        # Keep all externally visible package updates until the full TensorRT build,
        # deserialization check, and manifest preparation have succeeded.  A failed
        # build therefore leaves an existing package's manifest and metadata intact.
        staged_manifest = staging / "manifest.json"
        staged_export_report = staging / "export_report.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged_export_report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        candidate_engine.replace(engine)
        candidate_report.replace(package / "metadata" / candidate_report.name)
        staged_export_report.replace(package / "metadata" / "export_report.json")
        staged_manifest.replace(package / "manifest.json")
        for relative_path in previous_engine_paths:
            previous = (package / relative_path).resolve()
            if previous != engine.resolve() and previous.parent == (package / "engines").resolve() and previous.is_file():
                previous.unlink()
        shutil.rmtree(staging)
        return {"ui": {"text": [f"TensorRT package created: {package}"]}}


NODE_CLASS_MAPPINGS = {"ExportTensorRTPackage": ExportTensorRTPackage}
NODE_DISPLAY_NAME_MAPPINGS = {"ExportTensorRTPackage": "Export TensorRT Package (Fused Model)"}
