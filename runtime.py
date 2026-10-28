from __future__ import annotations

from dataclasses import dataclass
import threading

import torch
import comfy.model_patcher

from .manifest import EngineSpec, ManifestError, PackageManifest, load_manifest, package_lock


IO_NAMES = {"latent", "timestep", "context", "adm", "noise_pred"}


def torch_dtype(trt, dtype):
    types = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
             trt.DataType.BF16: torch.bfloat16, trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64}
    if dtype not in types:
        raise ManifestError(f"Unsupported TensorRT I/O dtype: {dtype}. Use FP32, FP16, or BF16 external bindings.")
    return types[dtype]


def validate_interface(engine, trt, latent_shape):
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    if names != IO_NAMES or engine.num_optimization_profiles != 1:
        raise ManifestError(f"Expected one SDXL profile and I/O {sorted(IO_NAMES)}, got {sorted(names)}.")
    shapes = {"latent": latent_shape, "timestep": (1,), "context": (1, -1, 2048), "adm": (1, 2816), "noise_pred": latent_shape}
    for name, shape in shapes.items():
        mode = trt.TensorIOMode.OUTPUT if name == "noise_pred" else trt.TensorIOMode.INPUT
        if tuple(engine.get_tensor_shape(name)) != shape or engine.get_tensor_mode(name) != mode:
            raise ManifestError(f"Invalid TensorRT interface for {name}: {engine.get_tensor_shape(name)}.")
        if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
            raise ManifestError(f"TensorRT {name} must use linear CUDA memory.")
        if torch_dtype(trt, engine.get_tensor_dtype(name)) not in (torch.float32, torch.float16, torch.bfloat16):
            raise ManifestError(f"TensorRT {name} must expose floating-point I/O.")
    profile = engine.get_tensor_profile_shape("context", 0)
    if len(profile) != 3 or any(tuple(v)[::2] != (1, 2048) or v[1] <= 0 for v in profile):
        raise ManifestError("Invalid TensorRT dynamic context profile.")
    return tuple(int(v[1]) for v in profile)


@dataclass
class _LoadedEngine:
    engine: object
    context: object
    profile: tuple[int, int, int]


class TensorRTUNet(torch.nn.Module):
    """Fixed-resolution SDXL engines with native ComfyUI sampling outside the plan."""

    dtype = torch.float32

    def __init__(self, package: PackageManifest):
        super().__init__()
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ManifestError("TensorRT 11 Python bindings are required.") from exc
        if not torch.cuda.is_available():
            raise ManifestError("TensorRT requires CUDA-enabled PyTorch.")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._trt = trt
        self._package = package
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = None
        self._loaded = None
        self._active_key = None
        self._lock = threading.RLock()
        self._manifest_stamp = None
        self._completion = None
        self._stream = None

    @property
    def package(self):
        return self._package

    def resident_memory(self):
        if self._loaded is None:
            return 0
        # Plans include weights; TensorRT separately reports execution-context memory.
        return self._active_key[3] + self._loaded.engine.device_memory_size_v2

    def _check_compatibility(self, spec):
        if self._trt.__version__ != spec.tensorrt_version:
            raise ManifestError(f"TensorRT version mismatch: engine requires {spec.tensorrt_version}, runtime is {self._trt.__version__}.")
        if torch.cuda.get_device_capability(self.device) != spec.compute_capability or torch.cuda.get_device_name(self.device) != spec.gpu_name:
            raise ManifestError(f"Engine requires {spec.gpu_name} (SM {spec.compute_capability}); rebuild for this GPU.")

    def _spec_for(self, latent):
        stamp = (self._package.root / "manifest.json").stat()
        stamp = (stamp.st_ino, stamp.st_mtime_ns, stamp.st_size)
        if stamp != self._manifest_stamp:
            package = load_manifest(self._package.root)
            if package.chain != self._package.chain:
                raise ManifestError("TensorRT package source changed; reload the MODEL.")
            self._release_active_engine()
            self._package = package
            self._manifest_stamp = stamp
        shape = (1, *tuple(latent.shape[1:]))
        for spec in self._package.engines:
            if spec.latent_shape == shape:
                return spec
        supported = ", ".join(f"{v.image_size[0]}x{v.image_size[1]}" for v in self._package.engines)
        raise RuntimeError(f"Unsupported TensorRT latent shape {tuple(latent.shape)}. Available image resolutions: {supported}.")

    def _engine_for(self, spec: EngineSpec):
        self._check_compatibility(spec)
        stat = spec.path.stat()
        key = (str(spec.path), stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if self._active_key == key and self._loaded is not None:
            return self._loaded
        self._release_active_engine()
        if self._runtime is None:
            self._runtime = self._trt.Runtime(self._logger)
        with package_lock(self._package.root):
            engine = self._runtime.deserialize_cuda_engine(spec.path.read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize engine: {spec.path.name}")
        context = None
        try:
            profile = validate_interface(engine, self._trt, spec.latent_shape)
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError(f"TensorRT failed to create an execution context: {spec.path.name}")
            completion = torch.cuda.Event()
            self._loaded = _LoadedEngine(engine, context, profile)
            self._completion = completion
        finally:
            context = engine = None
            if self._loaded is None:
                self._runtime = None
        self._active_key = key
        return self._loaded

    def _release_active_engine(self):
        if self._loaded is not None:
            if self._completion is not None:
                self._completion.synchronize()
            self._loaded.context = None
            self._loaded.engine = None
            self._loaded = None
            self._active_key = None
            self._completion = None
            self._stream = None

    def release(self):
        with self._lock, torch.cuda.device(self.device):
            self._release_active_engine()
            self._runtime = None

    def _execute(self, spec, x, timesteps, context, y):
        loaded = self._engine_for(spec)
        stream = torch.cuda.current_stream(x.device)
        if self._stream is not None and self._stream != stream.cuda_stream:
            self._completion.synchronize()
        if not loaded.profile[0] <= context.shape[1] <= loaded.profile[2]:
            raise RuntimeError(f"TensorRT context has {context.shape[1]} tokens; {spec.path.name} supports {loaded.profile[0]}-{loaded.profile[2]} (opt {loaded.profile[1]}).")
        tensors = {"latent": x, "timestep": timesteps, "context": context, "adm": y}
        for name, tensor in tensors.items():
            tensor = tensor.to(device=x.device, dtype=torch_dtype(self._trt, loaded.engine.get_tensor_dtype(name))).contiguous()
            tensors[name] = tensor
            if not loaded.context.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(f"TensorRT rejected {name} shape {tuple(tensor.shape)}.")
        output_shape = tuple(loaded.context.get_tensor_shape("noise_pred"))
        tensors["noise_pred"] = torch.empty(output_shape, device=x.device, dtype=torch_dtype(self._trt, loaded.engine.get_tensor_dtype("noise_pred")))
        for name, tensor in tensors.items():
            if not loaded.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"TensorRT could not bind {name}.")
            tensor.record_stream(stream)
        if not loaded.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed.")
        self._completion.record(stream)
        self._stream = stream.cuda_stream
        return tensors["noise_pred"].to(dtype=x.dtype)

    def forward(self, x, timesteps, context=None, y=None, control=None, **_kwargs):
        if control is not None:
            raise RuntimeError("TensorRT SDXL engines do not support ControlNet.")
        if x.device != self.device or context is None or y is None:
            raise RuntimeError(f"TensorRT requires latent on {self.device}, context, and SDXL ADM conditioning.")
        with self._lock, torch.cuda.device(self.device):
            spec = self._spec_for(x)
            if x.shape[0] == 1:
                return self._execute(spec, x, timesteps, context, y)
            return torch.cat([
                self._execute(spec, x[i:i + 1], timesteps[i:i + 1], context[i:i + 1], y[i:i + 1])
                for i in range(x.shape[0])
            ], dim=0)


class TensorRTModelPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False, fast_disk=False):
        super().__init__(model, load_device, offload_device, size, weight_inplace_update, fast_disk)
        self.engine_unet = model.diffusion_model

    def model_size(self):
        return max(self.loaded_size(), max(spec.path.stat().st_size for spec in self.engine_unet.package.engines))

    def loaded_size(self):
        return super().loaded_size() + self.engine_unet.resident_memory()

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        before = self.engine_unet.resident_memory()
        self.engine_unet.release()
        return before + super().partially_unload(device_to, max(0, memory_to_free - before), force_patch_weights)
