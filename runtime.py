from __future__ import annotations

from dataclasses import dataclass
import gc

import torch

from .manifest import EngineSpec, ManifestError, PackageManifest


@dataclass
class _LoadedEngine:
    engine: object
    context: object


class TensorRTUNet(torch.nn.Module):
    """Raw SDXL UNet replacement used by a real ComfyUI ModelPatcher."""

    dtype = torch.float16
    device = torch.device("cuda")

    def __init__(self, package: PackageManifest):
        super().__init__()
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ManifestError("TensorRT Python bindings are required by TensorRTLoader.") from exc
        if not torch.cuda.is_available():
            raise ManifestError("TensorRTLoader requires a CUDA-enabled PyTorch installation.")
        if int(trt.__version__.split(".", 1)[0]) != package.tensorrt_major:
            raise ManifestError(
                f"TensorRT major version mismatch: package requires {package.tensorrt_major}, runtime is {trt.__version__}."
            )
        capability = torch.cuda.get_device_capability()
        if capability < package.compute_capability_min:
            required = ".".join(map(str, package.compute_capability_min))
            available = ".".join(map(str, capability))
            raise ManifestError(f"GPU compute capability {available} is below package requirement {required}.")
        self._trt = trt
        self._package = package
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self._loaded: _LoadedEngine | None = None
        self._active_key: str | None = None

    def _spec_for(self, latent: torch.Tensor, context: torch.Tensor) -> EngineSpec:
        shape = (1, *tuple(latent.shape[1:]))
        matches = [
            spec for spec in self._package.engines
            if spec.latent_shape == shape and spec.context_min <= context.shape[1] <= spec.context_max
        ]
        if matches:
            return matches[0]
        supported = "\n".join(
            f"image {item.image_size[0]}x{item.image_size[1]}, "
            f"{item.context_min}-{item.context_max} tokens (optimized for {item.context_opt} tokens)"
            for item in self._package.engines
        )
        raise RuntimeError(
            "TensorRT engine does not support the requested latent size or conditioning length. "
            f"Got latent shape {tuple(latent.shape)} and {context.shape[1]} tokens. Supported combinations:\n{supported}"
        )

    def _engine_for(self, spec: EngineSpec) -> _LoadedEngine:
        key = str(spec.path)
        if self._active_key == key and self._loaded is not None:
            return self._loaded
        self._release_active_engine()
        engine = self._runtime.deserialize_cuda_engine(spec.path.read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize engine: {spec.path.name}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"TensorRT failed to create an execution context: {spec.path.name}")
        expected = {item["name"] for item in spec.bindings}
        actual = {engine.get_tensor_name(index) for index in range(engine.num_io_tensors)}
        if actual != expected:
            raise RuntimeError(f"TensorRT binding mismatch for {spec.path.name}; package and engine do not match.")
        loaded = _LoadedEngine(engine, context)
        self._loaded = loaded
        self._active_key = key
        return loaded

    def _release_active_engine(self) -> None:
        if self._loaded is None:
            return
        # TensorRT owns allocations outside PyTorch's allocator. Synchronizing is
        # necessary before an orientation switch destroys the execution context.
        torch.cuda.synchronize()
        self._loaded.context = None
        self._loaded.engine = None
        self._loaded = None
        self._active_key = None
        gc.collect()
        torch.cuda.empty_cache()

    def release(self) -> None:
        """Release the TensorRT plan and context until this model is used again."""
        self._release_active_engine()

    @property
    def checkpoint(self):
        return self._package.checkpoint

    def _execute(self, spec: EngineSpec, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loaded = self._engine_for(spec)
        if not loaded.context.set_input_shape("latent", tuple(x.shape)):
            raise RuntimeError(f"TensorRT could not select the latent shape {tuple(x.shape)}.")
        if not loaded.context.set_input_shape("context", tuple(context.shape)):
            raise RuntimeError(f"TensorRT could not select the {context.shape[1]}-token context shape.")
        output_shape = tuple(loaded.context.get_tensor_shape("noise_pred"))
        tensors = {"latent": x, "timestep": timesteps, "context": context, "adm": y, "noise_pred": torch.empty(output_shape, device=x.device, dtype=x.dtype)}
        for name, tensor in tensors.items():
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise RuntimeError(f"TensorRT input {name} must be a contiguous CUDA tensor.")
            if tuple(tensor.shape) != tuple(loaded.context.get_tensor_shape(name)):
                raise RuntimeError(f"TensorRT input {name} shape mismatch: got {tuple(tensor.shape)}.")
            expected_dtype = loaded.engine.get_tensor_dtype(name)
            actual_dtype = {torch.float16: self._trt.DataType.HALF, torch.float32: self._trt.DataType.FLOAT}.get(tensor.dtype)
            if actual_dtype != expected_dtype:
                raise RuntimeError(f"TensorRT input {name} dtype mismatch: got {tensor.dtype}, expected {expected_dtype}.")
            if not loaded.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"TensorRT could not bind {name}.")
        stream = torch.cuda.current_stream(x.device)
        if not loaded.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed.")
        return tensors["noise_pred"]

    def forward(self, x, timesteps, context=None, y=None, control=None, **_kwargs):
        if control is not None:
            raise RuntimeError("TensorRTLoader package does not include ControlNet execution.")
        if x.device.type != "cuda" or context is None or y is None:
            raise RuntimeError("TensorRTLoader requires CUDA latent, context, and SDXL ADM conditioning tensors.")
        spec = self._spec_for(x, context)
        if x.shape[0] == 1:
            return self._execute(spec, x, timesteps, context, y)
        # ComfyUI batches unconditional and conditional CFG passes. The engines are
        # explicitly B1, so slice the independent samples on CUDA rather than feed an
        # invalid binding shape or change the requested resolution.
        return torch.cat([
            self._execute(spec, x[index:index + 1], timesteps[index:index + 1], context[index:index + 1], y[index:index + 1])
            for index in range(x.shape[0])
        ], dim=0)
