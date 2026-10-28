"""Calibrated linear quantization of the independent export model."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import onnx
import torch

from .manifest import TEMP_ROOT, ManifestError


def load_calibration(filename: str, width: int, height: int) -> dict[str, np.ndarray]:
    if not filename:
        raise ManifestError("FP8/NVFP4 require representative denoising calibration data (.npz) under ComfyUI/dev_temp. Supply calibration_data; random calibration is not used.")
    path = Path(filename).resolve()
    if not path.is_relative_to(TEMP_ROOT.parent.resolve()) or path.suffix != ".npz" or not path.is_file():
        raise ManifestError("calibration_data must be an existing .npz file under ComfyUI/dev_temp.")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"latent", "timestep", "context", "adm"}:
            raise ManifestError("Calibration requires latent, timestep, context, and adm arrays from real denoising calls.")
        data = {key: archive[key] for key in archive.files}
    if data["latent"].ndim != 5:
        raise ManifestError("Calibration latent must have shape [samples, 1, 4, height/8, width/8].")
    count = len(data["latent"])
    shapes = {"latent": (1, 4, height // 8, width // 8), "timestep": (1,), "adm": (1, 2816)}
    for key, value in data.items():
        shape = value.shape[1:]
        if key == "context":
            valid_shape = len(shape) == 3 and shape[0] == 1 and shape[1] > 0 and shape[2] == 2048
        else:
            valid_shape = shape == shapes[key]
        if not count or not value.ndim or len(value) != count or not valid_shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ManifestError(f"Invalid calibration array {key}: {value.shape}, {value.dtype}.")
    return data


def modelopt_modules():
    try:
        import modelopt.torch.quantization as mtq
        from modelopt.onnx.export import FP8QuantExporter, NVFP4QuantExporter
        from modelopt.torch.quantization.export_onnx import configure_linear_module_onnx_quantizers
    except ImportError as exc:
        raise ManifestError("Calibrated FP8/NVFP4 export requires nvidia-modelopt[onnx]==0.47.0 in the ComfyUI environment.") from exc
    return mtq, {"fp8": FP8QuantExporter, "nvfp4": NVFP4QuantExporter}, configure_linear_module_onnx_quantizers


def quantize_linears(wrapper: torch.nn.Module, precision: str, calibration: dict[str, np.ndarray]) -> None:
    mtq, _, _ = modelopt_modules()
    # ComfyUI cast/offload operations are unnecessary on this fully materialized copy.
    # Standard Linear modules let ModelOpt use its supported quantization/export path.
    count = 0
    for module in list(wrapper.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear):
                with torch.device("meta"):
                    linear = torch.nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                linear.weight, linear.bias = child.weight, child.bias
                setattr(module, name, linear)
                count += 1
    if not count:
        raise ManifestError("Model has no supported Linear operations to quantize.")
    device = next(wrapper.parameters()).device
    dtype = next(wrapper.parameters()).dtype

    def calibrate(model):
        for index in range(len(calibration["latent"])):
            inputs = [torch.from_numpy(calibration[key][index]).to(device=device, dtype=torch.float32 if key == "timestep" else dtype)
                      for key in ("latent", "timestep", "context", "adm")]
            model(*inputs)

    config = mtq.FP8_DEFAULT_CFG if precision == "fp8" else mtq.NVFP4_DEFAULT_CFG
    mtq.quantize(wrapper, config, forward_loop=calibrate)


def export_context(wrapper: torch.nn.Module, precision: str):
    if precision != "nvfp4":
        return nullcontext()
    _, _, configure = modelopt_modules()
    return configure(wrapper)


def validate_quantized_graph(graph: onnx.GraphProto, precision: str) -> None:
    target = onnx.TensorProto.FLOAT8E4M3FN if precision == "fp8" else onnx.TensorProto.FLOAT4E2M1
    weights = {item.name for item in graph.initializer if item.data_type == target}
    weights.update(output for node in graph.node if node.op_type == "Constant"
                   for attr in node.attribute if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == target
                   for output in node.output)
    producers = {output: node for node in graph.node for output in node.output}

    def dequantized(value):
        node = producers.get(value)
        while node is not None and node.op_type in {"Cast", "Transpose", "Reshape"}:
            node = producers.get(node.input[0])
        return node.input[0] if node is not None and node.op_type == "DequantizeLinear" else None

    activation_op = "QuantizeLinear" if precision == "fp8" else "TRT_FP4DynamicQuantize"
    for node in graph.node:
        if node.op_type in {"MatMul", "Gemm", "Conv"}:
            inputs = [dequantized(value) for value in node.input[:2]]
            if any(value in weights for value in inputs) and any(
                value in producers and producers[value].op_type == activation_op for value in inputs
            ):
                return
    raise ManifestError(f"ModelOpt did not produce {precision} weight-and-activation quantized matrix operations.")


def finish_export(path: Path, precision: str) -> Path:
    if precision in {"fp8", "nvfp4"}:
        _, exporters, _ = modelopt_modules()
        if precision == "fp8":
            # ModelOpt's Softmax rewrite chooses its scale dtype from value_info.
            # Populate it before conversion so strongly typed attention keeps FP16 I/O.
            inferred = path.with_name(path.stem + ".inferred.onnx")
            onnx.shape_inference.infer_shapes_path(str(path), str(inferred))
            path = inferred
        model = exporters[precision].process_model(onnx.load(str(path)))
        converted = path.with_name(path.stem + ".quantized.onnx")
        onnx.save_model(model, str(converted), save_as_external_data=True, all_tensors_to_one_file=True,
                        location=converted.with_suffix(".data").name, size_threshold=1024)
        path = converted
    if precision in {"fp8", "nvfp4"}:
        graph = onnx.load(str(path), load_external_data=False).graph
        validate_quantized_graph(graph, precision)
    return path
