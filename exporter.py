"""Export the materialized SDXL UNet; keep source reconstruction at the node boundary."""
from __future__ import annotations

from collections import Counter
import gc
import json
from pathlib import Path
import tempfile
import traceback

import folder_paths
import torch
import comfy.model_management
import comfy.sd
import comfy.supported_models
from comfy.ldm.modules.attention import attention_pytorch

from .manifest import TEMP_ROOT, PRECISIONS, ManifestError, engine_name, load_manifest, package_lock, package_name as default_package_name, publish_engine, source_name
from .provenance import recover_chain, resolve_source
from .quantization import export_context, finish_export, load_calibration, modelopt_modules, quantize_linears
from .runtime import torch_dtype, validate_interface


_CONTEXT_MIN = 77
_PACKAGE_ROOT = Path(folder_paths.models_dir) / "tensorrt"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float16, "nvfp4": torch.float16}


def validate_precision(precision):
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ManifestError("Install TensorRT 11 Python bindings to export engines.") from exc
    if precision not in PRECISIONS:
        raise ManifestError(f"Unknown precision {precision!r}.")
    if int(trt.__version__.split(".")[0]) != 11:
        raise ManifestError(f"TensorRT 11.x is required; installed version is {trt.__version__}.")
    if not torch.cuda.is_available():
        raise ManifestError("TensorRT export requires CUDA.")
    capability = torch.cuda.get_device_capability()
    minimum = {"fp32": (7, 5), "fp16": (7, 5), "bf16": (8, 0), "fp8": (8, 9), "nvfp4": (10, 0)}[precision]
    if capability < minimum:
        raise ManifestError(f"{precision} requires SM {minimum}; GPU {torch.cuda.get_device_name()} is SM {capability}.")
    return trt


def _export_attention(_selected_backend, *args, **kwargs):
    return attention_pytorch(*args, **kwargs)


class _ExportUNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent, timestep, context, adm):
        return self.unet(latent, timestep, context=context, y=adm,
                         transformer_options={"optimized_attention_override": _export_attention})


def materialize_unet(model, precision):
    if type(model.model.model_config) is not comfy.supported_models.SDXL:
        raise ManifestError("TensorRT export currently supports base SDXL UNets only.")
    comfy.model_management.load_models_gpu([model])
    state = model.model_state_dict_for_saving(model.model.diffusion_model, "diffusion_model.")
    state = {name: value.to(device="cpu").detach().contiguous() for name, value in state.items()}
    comfy.model_management.unload_model_and_clones(model)
    exported = comfy.sd.load_diffusion_model_state_dict(
        state, model_options={"dtype": DTYPES[precision]}, disable_dynamic=True,
    )
    if exported is None:
        raise ManifestError("Could not reconstruct the materialized SDXL UNet for export.")
    exported.model.diffusion_model.to(dtype=DTYPES[precision])
    return exported


def _export_onnx(unet, output, width, height, context_length, precision, calibration=None, verification_path=None):
    dtype = DTYPES[precision]
    device = next(unet.parameters()).device
    wrapper = _ExportUNet(unet).eval()
    latent = torch.randn((1, 4, height // 8, width // 8), device=device, dtype=dtype)
    timestep = torch.full((1,), 500.0, device=device, dtype=torch.float32)
    context = torch.randn((1, context_length, 2048), device=device, dtype=dtype)
    adm = torch.randn((1, 2816), device=device, dtype=dtype)
    if verification_path is not None:
        expected = wrapper(latent, timestep, context, adm).detach().cpu()
        torch.save({"inputs": {key: value.cpu() for key, value in zip(("latent", "timestep", "context", "adm"), (latent, timestep, context, adm))},
                    "expected": expected}, verification_path)
    if precision in {"fp8", "nvfp4"}:
        if calibration is None:
            raise ManifestError("Quantized export requires representative denoising calibration data.")
        quantize_linears(wrapper, precision, calibration)
    with export_context(wrapper, precision):
        torch.onnx.export(
            wrapper, (latent, timestep, context, adm), str(output),
            input_names=["latent", "timestep", "context", "adm"], output_names=["noise_pred"],
            opset_version=20 if precision in {"fp8", "nvfp4"} else 18, dynamo=False, external_data=True,
            dynamic_axes={"context": {1: "context_length"}}, do_constant_folding=True,
        )
    return finish_export(output, precision)


def _build_engine(onnx, engine, report, workspace_mib, context_profile, latent_shape, builder_optimization_level, precision, verification_path=None):
    trt = validate_precision(precision)

    class Logger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.messages = []

        def log(self, severity, message):
            self.messages.append(f"{severity}: {message}")

    logger = Logger()
    builder = network = parser = config = profile = serialized = runtime = built = context = inspector = None
    try:
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ManifestError(f"TensorRT {precision} ONNX parser failed:\n{errors}")
        counts = Counter(str(network.get_layer(i).get_output(j).dtype)
                         for i in range(network.num_layers) for j in range(network.get_layer(i).num_outputs)
                         if network.get_layer(i).get_output(j) is not None)
        requested = {"fp32": trt.DataType.FLOAT, "fp16": trt.DataType.HALF, "bf16": trt.DataType.BF16,
                     "fp8": trt.DataType.FP8, "nvfp4": trt.DataType.FP4}[precision]
        if not counts[str(requested)]:
            raise ManifestError(f"ONNX graph has no {precision} tensors; actual types: {dict(counts)}")
        config = builder.create_builder_config()
        config.builder_optimization_level = builder_optimization_level
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        # TensorRT 11 infers precision from the graph. Preserve default TF32 behavior.
        profile = builder.create_optimization_profile()
        profile.set_shape("context", *[(1, context_profile[key], 2048) for key in ("min", "opt", "max")])
        if config.add_optimization_profile(profile) < 0:
            raise ManifestError("TensorRT rejected the context optimization profile.")
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise ManifestError(f"TensorRT {precision} compilation failed:\n" + "\n".join(logger.messages[-30:]))
        runtime = trt.Runtime(logger)
        built = runtime.deserialize_cuda_engine(serialized)
        if built is None:
            raise ManifestError("TensorRT engine failed immediate deserialization.")
        actual_profile = validate_interface(built, trt, latent_shape)
        if actual_profile != tuple(context_profile[key] for key in ("min", "opt", "max")):
            raise ManifestError("Built TensorRT context profile does not match the request.")
        context = built.create_execution_context()
        if context is None:
            raise ManifestError("Built TensorRT engine cannot create an execution context.")
        shapes = {"latent": latent_shape, "timestep": (1,), "context": (1, context_profile["opt"], 2048), "adm": (1, 2816)}
        verification = torch.load(verification_path, map_location="cpu", weights_only=True) if verification_path is not None else None
        tensors = {}
        for name, shape in shapes.items():
            dtype = torch_dtype(trt, built.get_tensor_dtype(name))
            if verification is not None:
                shape = tuple(verification["inputs"][name].shape)
            tensors[name] = (verification["inputs"][name].to(device="cuda", dtype=dtype) if verification is not None
                             else torch.randn(shape, device="cuda", dtype=dtype)).contiguous()
            if not context.set_input_shape(name, shape):
                raise ManifestError(f"Built TensorRT engine rejected {name} shape {shape}.")
        tensors["noise_pred"] = torch.empty(latent_shape, device="cuda", dtype=torch_dtype(trt, built.get_tensor_dtype("noise_pred")))
        for name, tensor in tensors.items():
            if not context.set_tensor_address(name, tensor.data_ptr()):
                raise ManifestError(f"Built TensorRT engine rejected {name} binding.")
        stream = torch.cuda.current_stream()
        if not context.execute_async_v3(stream.cuda_stream):
            raise ManifestError("Built TensorRT engine failed validation inference.")
        stream.synchronize()
        output = tensors["noise_pred"].float().cpu()
        if not torch.isfinite(output).all():
            raise ManifestError("Built TensorRT engine produced nonfinite validation output.")
        accuracy = None
        if verification is not None:
            expected = verification["expected"].float()
            difference = output - expected
            accuracy = {"max_abs": difference.abs().max().item(), "rmse": difference.square().mean().sqrt().item()}
            if precision in {"fp32", "fp16", "bf16"}:
                torch.testing.assert_close(output, expected, rtol=0.15 if precision == "bf16" else 0.03,
                                           atol=0.1 if precision == "bf16" else 0.02)
        inspector = built.create_engine_inspector()
        inspection = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        layer_types = Counter()
        for layer in inspection["Layers"]:
            for tensor in layer.get("Inputs", []) + layer.get("Outputs", []):
                layer_types[tensor["Datatype"]] += 1
        expected_type = {"fp32": "Float", "fp16": "Half", "bf16": "BFloat16", "fp8": "FP8", "nvfp4": "FP4E2M1"}[precision]
        if not layer_types[expected_type]:
            raise ManifestError(f"Engine inspector reports no {precision} execution tensors: {dict(layer_types)}")
        # Inspect tensor formats after TensorRT fusion/tactic selection.
        result = {"precision": precision, "network_tensor_types": dict(counts), "engine_tensor_types": dict(layer_types), "engine_layers": inspection, "validation_accuracy": accuracy,
                  "requirements": {"tensorrt_version": trt.__version__, "gpu_name": torch.cuda.get_device_name(),
                                   "compute_capability": list(torch.cuda.get_device_capability())}}
        engine.write_bytes(bytes(serialized))
        report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    except Exception:
        report.write_text("\n".join(logger.messages) + "\n" + traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        # Tracebacks retain locals on failure; close TensorRT owners before staging cleanup.
        inspector = context = built = runtime = None
        parser = network = config = profile = builder = serialized = None


class ExportTensorRTPackage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",),
            "package_name": ("STRING", {"default": "", "tooltip": "Package folder; blank uses the checkpoint name. Use a separate folder for each LoRA chain."}),
            "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
            "height": ("INT", {"default": 1536, "min": 64, "max": 16384, "step": 8}),
            "maximum_context_length": ("INT", {"default": 462, "min": 77, "max": 2048}),
            "optimization_context_length": ("INT", {"default": 308, "min": 77, "max": 2048}),
            "workspace_mib": ("INT", {"default": 1024, "min": 64, "max": 32768, "step": 64}),
            "builder_optimization_level": ("INT", {"default": 3, "min": 0, "max": 5}),
        }, "optional": {"precision": (list(PRECISIONS), {"default": "fp16"}),
                         "calibration_data": ("STRING", {"default": "", "tooltip": "FP8/NVFP4: .npz denoising inputs under ComfyUI/dev_temp. See README for layout."})},
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT/conversion"
    DESCRIPTION = "Builds a TensorRT 11 SDXL engine from native checkpoint and ordered LoRA loaders."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def convert(self, model, clip, vae, width, height, maximum_context_length, optimization_context_length,
                workspace_mib, builder_optimization_level, precision="fp16", dynprompt=None, unique_id=None, package_name="", calibration_data=""):
        validate_precision(precision)
        if width <= 0 or height <= 0 or width % 8 or height % 8:
            raise ManifestError("Width and height must be positive multiples of 8.")
        if not _CONTEXT_MIN <= optimization_context_length <= maximum_context_length:
            raise ManifestError("Context lengths must satisfy 77 <= optimization <= maximum.")
        if not 0 <= builder_optimization_level <= 5 or workspace_mib <= 0:
            raise ManifestError("Builder optimization must be 0-5 and workspace must be positive.")
        calibration = None
        if precision in {"fp8", "nvfp4"}:
            calibration = load_calibration(calibration_data, width, height)
            modelopt_modules()
        chain = recover_chain(dynprompt, unique_id)
        expected = Path(resolve_source("checkpoints", chain["base_model"])).resolve()
        for patcher in (model, clip.patcher, vae.patcher):
            factory = patcher.cached_patcher_init
            if factory is None or Path(factory[1][0]).resolve() != expected:
                raise ManifestError("Input patcher checkpoint does not match the recovered source graph.")
        root = _PACKAGE_ROOT.resolve()
        selected_name = package_name or default_package_name(chain)
        source_name(selected_name)
        package = (root / selected_name).resolve()
        if package.parent != root:
            raise ManifestError("Package path escapes the TensorRT model directory.")
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with package_lock(str(package) + ".build"):
            if package.exists() and load_manifest(package).chain != chain:
                raise ManifestError("Existing package belongs to a different model chain. Select a different package_name for this checkpoint/LoRA configuration.")
            with tempfile.TemporaryDirectory(prefix="build-", dir=TEMP_ROOT) as temporary:
                staging = Path(temporary)
                candidate = staging / engine_name(chain, width, height)
                report = staging / "build.json"
                exported = None
                try:
                    exported = materialize_unet(model, precision)
                    comfy.model_management.load_models_gpu([exported], force_patch_weights=True, force_full_load=True)
                    onnx = staging / "model.onnx"
                    verification_path = staging / "verification.pt"
                    onnx = _export_onnx(exported.model.diffusion_model, onnx, width, height, optimization_context_length, precision, calibration, verification_path)
                    comfy.model_management.unload_model_and_clones(exported)
                    exported = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    profile = {"min": _CONTEXT_MIN, "opt": optimization_context_length, "max": maximum_context_length}
                    result = _build_engine(onnx, candidate, report, workspace_mib, profile,
                                           (1, 4, height // 8, width // 8), builder_optimization_level, precision, verification_path)
                    publish_engine(package, chain, candidate, (width, height), precision, result["requirements"], staging)
                except Exception:
                    diagnostic = TEMP_ROOT / (staging.name + "-failure.txt")
                    diagnostic.write_text((report.read_text() if report.exists() else "") + "\n" + traceback.format_exc(), encoding="utf-8")
                    raise ManifestError(f"TensorRT {precision} export failed. Diagnostics: {diagnostic}\n{traceback.format_exc().splitlines()[-1]}")
                finally:
                    if exported is not None:
                        comfy.model_management.unload_model_and_clones(exported)
                    gc.collect()
                    torch.cuda.empty_cache()
        types = result["engine_tensor_types"]
        return {"ui": {"text": [f"TensorRT package: {package}\nRequested {precision}; engine tensor types after fusion: {types}\n"
                                f"Validation deviation from export model: {result['validation_accuracy']}\n"
                                "Validation checks execution; quantized image quality depends on representative calibration."]}}


NODE_CLASS_MAPPINGS = {"ExportTensorRTPackage": ExportTensorRTPackage}
NODE_DISPLAY_NAME_MAPPINGS = {"ExportTensorRTPackage": "Export TensorRT Package (Fused Model)"}
