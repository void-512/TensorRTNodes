"""Standalone diagnostics; all generated artifacts belong in ComfyUI/dev_temp.

Run with ComfyUI's Python and an explicit CUDA_VISIBLE_DEVICES selection.
Production nodes do not import this module.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback
import types

import numpy as np
import onnx
import torch
import tensorrt as trt


COMFY_ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = COMFY_ROOT / "dev_temp"
GPU_FIELDS = ("uuid", "name", "driver_version", "utilization.gpu", "utilization.memory", "memory.free",
              "clocks.sm", "clocks.mem", "power.draw", "temperature.gpu", "clocks_event_reasons.active",
              "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_thermal_slowdown",
              "clocks_event_reasons.hw_power_brake_slowdown")
DTYPES = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16, trt.DataType.BF16: torch.bfloat16}
PRECISIONS = ("fp32", "fp16", "bf16")


def artifact_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(DEV_ROOT.resolve()):
        raise ValueError(f"Experimental artifacts must stay under {DEV_ROOT}: {path}")
    return path


def write_json(path, data):
    path = artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    return {"count": len(values), "median": statistics.median(values), "p95": values[math.ceil(.95 * len(values)) - 1],
            "mean": mean, "stddev": std, "cv": std / mean if mean else 0,
            "min": values[0], "max": values[-1]}


def contention(samples):
    """Only idle-window samples can estimate background activity, not our own kernels."""
    values = [sample["utilization.gpu"] for sample in samples if isinstance(sample.get("utilization.gpu"), (int, float))]
    if not values:
        return "unknown"
    p95 = distribution(values)["p95"]
    return "relatively_idle" if p95 <= 5 else "moderately_contended" if p95 <= 30 else "heavily_contended"


def gpu_snapshot(gpu):
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=" + ",".join(GPU_FIELDS),
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
    values = next(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    data = {"time_ns": time.time_ns()}
    for field, value in zip(GPU_FIELDS, values):
        try:
            data[field] = float(value)
        except ValueError:
            data[field] = value
    return data


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


class Telemetry:
    def __init__(self, gpu):
        self.gpu = gpu
        self.samples = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            try:
                self.samples.append(gpu_snapshot(self.gpu))
            except (OSError, subprocess.SubprocessError) as exc:
                self.errors.append(str(exc))
            self.stop.wait(.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


def idle_window(gpu, seconds):
    samples = []
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        samples.append(gpu_snapshot(gpu))
        time.sleep(min(.5, max(0, until - time.monotonic())))
    return samples


def environment(gpu):
    snapshot = gpu_snapshot(gpu)
    actual_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid).removeprefix("GPU-")
    if actual_uuid != snapshot["uuid"].removeprefix("GPU-"):
        raise ValueError("Telemetry GPU differs from CUDA's current device; check --gpu and CUDA_VISIBLE_DEVICES")
    return {"gpu": snapshot, "processes": gpu_processes(), "torch": torch.__version__,
            "tensorrt": trt.__version__, "onnx": onnx.__version__, "cuda": torch.version.cuda,
            "python": sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY")}


def project_modules():
    # ComfyUI parses its own command line at import. CPU policy keeps export/reference
    # loading independent of whichever GPU this standalone diagnostic measures.
    sys.path.insert(0, str(COMFY_ROOT))
    original_argv = sys.argv
    sys.argv = [sys.argv[0], "--cpu"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        package = types.ModuleType("trt_benchmark_nodes")
        package.__path__ = [str(Path(__file__).resolve().parent)]
        sys.modules.setdefault(package.__name__, package)
        return {name: importlib.import_module(f"{package.__name__}.{name}")
                for name in ("exporter", "runtime", "manifest", "provenance")}
    finally:
        sys.argv = original_argv


def graph_summary(path):
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    tensors = {item.name: onnx.TensorProto.DataType.Name(item.data_type) for item in graph.initializer}
    casts = Counter()
    for node in graph.node:
        if node.op_type == "Cast":
            target = next(attr.i for attr in node.attribute if attr.name == "to")
            casts[f"{tensors.get(node.input[0], 'activation')}->{onnx.TensorProto.DataType.Name(target)}"] += 1
    def shape(item):
        return [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in item.type.tensor_type.shape.dim]
    return {"nodes": len(graph.node), "operators": dict(Counter(node.op_type for node in graph.node)),
            "initializer_count": len(initializers), "initializer_elements": sum(math.prod(item.dims) for item in initializers.values()),
            "initializer_types": dict(Counter(tensors.values())), "casts": dict(casts),
            "parameter_shapes": {name: list(item.dims) for name, item in initializers.items()},
            "inputs": {item.name: shape(item) for item in graph.input}, "outputs": {item.name: shape(item) for item in graph.output}}


def graph_file_bytes(path):
    path = Path(path).resolve()
    graph = onnx.load(str(path), load_external_data=False).graph
    files = {path}
    for tensor in graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            external = (path.parent / dict((item.key, item.value) for item in tensor.external_data)["location"]).resolve()
            if not external.is_relative_to(path.parent):
                raise ValueError("ONNX external data escapes the prepared graph directory")
            files.add(external)
    return sum(file.stat().st_size for file in files)


def tensor_error(output, reference):
    if output.shape != reference.shape:
        raise ValueError("Output/reference shapes differ")
    output, reference = output.float().cpu(), reference.float().cpu()
    if not torch.isfinite(output).all():
        raise ValueError("Nonfinite inference output")
    difference = output - reference
    rmse = difference.square().mean().sqrt().item()
    norm = reference.square().mean().sqrt().item()
    return {"rmse": rmse, "relative_rmse": rmse / norm if norm else None,
            "max_abs": difference.abs().max().item(), "shape": list(output.shape), "finite": True}


def input_signature(inputs):
    digest = hashlib.sha256()
    for name, tensor in sorted(inputs.items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prepare(args):
    modules = project_modules()
    source = json.loads(Path(args.source).read_text())
    source = source.get("source", source)
    root = artifact_path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    print("Loading source MODEL", flush=True)
    model, _, _ = modules["provenance"].reconstruct(source, output_clip=False, output_vae=False, disable_dynamic=True)
    print("Materializing patched weights", flush=True)
    exported = modules["exporter"].materialize_unet(model, args.precision)
    unet = exported.model.diffusion_model.to(args.export_device)
    generator = torch.Generator().manual_seed(args.seed)
    inputs = {"latent": torch.randn(1, 4, args.height // 8, args.width // 8, generator=generator),
              "timestep": torch.tensor([500.]), "context": torch.randn(1, args.context, 2048, generator=generator),
              "adm": torch.randn(1, 2816, generator=generator)}
    dtype = modules["exporter"].DTYPES[args.precision]
    converted = [value.to(device=args.export_device, dtype=torch.float32 if key == "timestep" else dtype)
                 for key, value in inputs.items()]
    print("Computing native reference", flush=True)
    expected = modules["exporter"]._ExportUNet(unet)(*converted).detach().cpu()
    torch.save({"inputs": inputs, "expected": expected}, root / "verification.pt")
    started = time.perf_counter()
    print("Exporting ONNX", flush=True)
    graph = modules["exporter"]._export_onnx(unet, root / "model.onnx", args.width, args.height, args.context,
                                               args.precision)
    metadata = {"source": source, "precision": args.precision, "width": args.width, "height": args.height,
                "context": args.context, "seed": args.seed, "export_device": args.export_device,
                "onnx": str(graph), "verification": str(root / "verification.pt"),
                "export_seconds": time.perf_counter() - started, "graph": graph_summary(graph)}
    metadata["software"] = {"torch": torch.__version__, "onnx": onnx.__version__}
    write_json(root / "prepared.json", metadata)
    print(json.dumps({key: metadata[key] for key in ("precision", "export_seconds", "onnx")}), flush=True)


class LayerProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.times = defaultdict(list)

    def report_layer_time(self, name, milliseconds):
        self.times[name].append(milliseconds)


class EngineSession:
    def __init__(self, engine_path, data, context_length=None):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        started = time.perf_counter()
        serialized = Path(engine_path).read_bytes()
        self.read_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        self.deserialize_seconds = time.perf_counter() - started
        if self.engine is None:
            raise RuntimeError("Engine deserialization failed")
        del serialized
        started = time.perf_counter()
        self.context = self.engine.create_execution_context()
        self.context_seconds = time.perf_counter() - started
        if self.context is None:
            raise RuntimeError("Execution-context creation failed")
        self.profiler = None
        self.stream = torch.cuda.current_stream()
        self.inputs = {name: value.cuda() for name, value in data["inputs"].items()}
        if context_length is not None:
            self.inputs["context"] = self.inputs["context"][:, :1].expand(1, context_length, 2048).contiguous()
        self.bound = {name: value.to(dtype=DTYPES[self.engine.get_tensor_dtype(name)]).contiguous() for name, value in self.inputs.items()}
        for name, value in self.bound.items():
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise ValueError(f"Engine rejected {name} shape {tuple(value.shape)}")
        self.bound["noise_pred"] = torch.empty(tuple(self.context.get_tensor_shape("noise_pred")), device="cuda",
                                               dtype=DTYPES[self.engine.get_tensor_dtype("noise_pred")])
        for name, value in self.bound.items():
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise ValueError(f"Engine rejected {name} binding")
        torch.cuda.synchronize()

    def execute(self):
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self.bound["noise_pred"]

    def close(self):
        self.stream.synchronize()
        self.context = self.engine = self.runtime = None
        self.profiler = None
        self.bound.clear()
        self.inputs.clear()


def inspection(engine):
    inspector = engine.create_engine_inspector()
    return json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))


def build(args):
    prepared = json.loads(Path(args.prepared).read_text())
    if prepared["precision"] not in PRECISIONS:
        raise ValueError(f"Unsupported engine precision: {prepared['precision']}")
    output = artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "build.json").exists():
        raise ValueError("Use a new output directory for each independent engine build")
    env = environment(args.gpu)
    free, _ = torch.cuda.mem_get_info()
    weight_bytes = graph_file_bytes(prepared["onnx"])
    if free < weight_bytes + args.workspace * 2**20 + args.reserve_gib * 2**30:
        raise RuntimeError(f"Insufficient build headroom: {free / 2**30:.2f} GiB free; graph artifacts {weight_bytes / 2**30:.2f} GiB, workspace {args.workspace} MiB, reserve {args.reserve_gib} GiB")
    logger = trt.Logger(trt.Logger.INFO)
    config_key = {"gpu_uuid": env["gpu"]["uuid"], "driver": env["gpu"]["driver_version"], "tensorrt": trt.__version__,
                  "cuda": torch.version.cuda, "precision": prepared["precision"], "source": prepared["source"],
                  "width": prepared["width"], "height": prepared["height"], "context": [args.context_min, prepared["context"], args.context_max],
                  "optimization_level": args.level, "workspace_mib": args.workspace, "avg_timing_iterations": args.avg_timing_iterations,
                  "editable_cache": args.editable_cache, "graph_operators": prepared["graph"]["operators"]}
    config_key["export_device"] = prepared["export_device"]
    before = idle_window(args.gpu, args.idle_seconds)
    started = time.perf_counter()
    with Telemetry(args.gpu) as telemetry:
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(prepared["onnx"]):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.builder_optimization_level = args.level
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace * 2**20)
        if args.avg_timing_iterations is not None:
            config.avg_timing_iterations = args.avg_timing_iterations
        if args.editable_cache:
            config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
        profile = builder.create_optimization_profile()
        profile.set_shape("context", (1, args.context_min, 2048), (1, prepared["context"], 2048), (1, args.context_max, 2048))
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError("Invalid context optimization profile")
        cache_path = artifact_path(args.timing_cache) if args.timing_cache else output / "timing.cache"
        cache_metadata = cache_path.with_suffix(".json")
        cache = b""
        if cache_path.exists():
            if json.loads(cache_metadata.read_text()) != config_key:
                raise ValueError("Timing cache environment/configuration mismatch")
            cache = cache_path.read_bytes()
        timing_cache = config.create_timing_cache(cache)
        if not config.set_timing_cache(timing_cache, False):
            raise RuntimeError("TensorRT rejected timing cache")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build failed; see build log")
        modules = project_modules()
        engine_path = output / modules["manifest"].engine_name(prepared["source"], prepared["width"], prepared["height"])
        engine_path.write_bytes(bytes(serialized))
        timing_entries = len(config.get_timing_cache().queryKeys())
        builder_flags = {name: config.get_flag(flag) for name, flag in trt.BuilderFlag.__members__.items()}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
        write_json(cache_metadata, config_key)
        serialized = parser = network = config = timing_cache = profile = builder = None
        gc.collect()
    result = {"status": "built", "prepared": str(Path(args.prepared).resolve()), "source": prepared["source"],
              "precision": prepared["precision"], "engine": str(engine_path), "engine_bytes": engine_path.stat().st_size,
              "build_seconds": time.perf_counter() - started, "config": config_key, "environment": env,
              "telemetry": telemetry.samples, "telemetry_errors": telemetry.errors, "pre_idle": before,
              "contention": contention(before), "independent_builds": 1, "cache_input_bytes": len(cache)}
    result["timing_cache_entries"] = timing_entries
    result["builder_flags"] = builder_flags
    data = torch.load(prepared["verification"], weights_only=True, map_location="cpu")
    session = EngineSession(engine_path, data)
    try:
        write_json(output / "inspection.json", inspection(session.engine))
        result["context_memory_bytes"] = session.engine.device_memory_size_v2
        write_json(output / "build.json", result)
        output_tensor = session.execute()
        torch.cuda.synchronize()
        result["accuracy"] = tensor_error(output_tensor, data["expected"])
        result["status"] = "pass"
    finally:
        session.close()
    write_json(output / "build.json", result)
    print(json.dumps({key: result[key] for key in ("status", "engine", "build_seconds", "accuracy")}), flush=True)


def timed_gpu(function, repetitions):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    gpu, host = [], []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        begin.record()
        function()
        end.record()
        end.synchronize()
        host.append((time.perf_counter_ns() - started) / 1e6)
        gpu.append(begin.elapsed_time(end))
    return {"cuda_ms": distribution(gpu), "host_ms": distribution(host), "cuda_samples_ms": gpu, "host_samples_ms": host}


def operation_group(layer):
    metadata = layer.get("Metadata", "").lower()
    matrix = layer.get("LayerType") == "gemm" or (layer.get("LayerType") == "fusion" and
              any(word in metadata for word in ("/matmul", "/gemm")))
    if "attn" in metadata and "softmax" in metadata:
        return "attention_fused" if "gemm_mha" in layer.get("TacticName", "") else "attention_softmax"
    if layer.get("LayerType") == "correlation" or (layer.get("LayerType") == "fusion" and "/conv" in metadata):
        return "convolution"
    if matrix:
        return "attention_matrix" if "/attn" in metadata and "/to_" not in metadata else "linear"
    if any(word in metadata for word in ("normalization", "layernorm", "instancenorm")):
        return "normalization/fused_pointwise"
    if layer.get("LayerType") in {"cast", "reshape"} or "reformat" in metadata:
        return "layout/cast"
    return "pointwise/other"


def profile_layers(session, repetitions):
    profiler = LayerProfiler()
    layers = inspection(session.engine)["Layers"]
    lookup = {layer["Name"]: layer for layer in layers}
    session.profiler = profiler
    session.context.profiler = profiler
    session.context.enqueue_emits_profile = False
    for _ in range(repetitions):
        session.execute()
        session.stream.synchronize()
        if not session.context.report_to_profiler():
            raise RuntimeError("TensorRT profiling report failed")
    rows, groups = [], defaultdict(float)
    for name, times in profiler.times.items():
        layer = lookup.get(name, {})
        group = operation_group(layer)
        milliseconds = statistics.mean(times)
        rows.append({"name": name, "group": group, "mean_ms": milliseconds, "samples": len(times),
                     "tactic": layer.get("TacticName"), "layer_type": layer.get("LayerType"),
                     "input_types": [item.get("Datatype") for item in layer.get("Inputs", [])],
                     "output_types": [item.get("Datatype") for item in layer.get("Outputs", [])]})
        groups[group] += milliseconds
    return {"layers": sorted(rows, key=lambda row: row["mean_ms"], reverse=True), "groups_ms": dict(groups),
            "sum_layer_ms": sum(groups.values()), "iterations": repetitions}


def wrapper_runtime(build_info, prepared):
    modules = project_modules()
    root = artifact_path(Path(build_info["engine"]).parent / "wrapper-package")
    root.mkdir(exist_ok=True)
    (root / "engines").mkdir(exist_ok=True)
    source = Path(build_info["engine"])
    destination = root / "engines" / source.name
    if destination.exists() and destination.stat().st_ino != source.stat().st_ino:
        destination.unlink()
    if not destination.exists():
        os.link(source, destination)
    requirements = {"tensorrt_version": trt.__version__, "gpu_name": torch.cuda.get_device_name(),
                    "compute_capability": list(torch.cuda.get_device_capability())}
    write_json(root / "manifest.json", {"format_version": 3, "source": prepared["source"], "engines": [{
        "path": "engines/" + source.name, "image_size": [prepared["width"], prepared["height"]],
        "precision": prepared["precision"], "requirements": requirements}]})
    return modules["runtime"].TensorRTUNet(modules["manifest"].load_manifest(root))


def run(args):
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    info = json.loads(Path(args.build).read_text())
    prepared = json.loads(Path(info["prepared"]).read_text())
    data = torch.load(prepared["verification"], map_location="cpu", weights_only=True)
    result = {"build": str(Path(args.build).resolve()), "config": info["config"], "source": prepared["source"],
              "precision": prepared["precision"], "environment": environment(args.gpu), "warmups": args.warmups,
              "repetitions": args.repetitions, "independent_builds": 1, "engine_bytes": info["engine_bytes"], "stream": args.stream}
    result["input_signature"] = input_signature(data["inputs"]) + f":context_override={args.context}"
    result["pre_idle"] = idle_window(args.gpu, args.idle_seconds)
    with Telemetry(args.gpu) as telemetry:
        free_before = torch.cuda.mem_get_info()[0]
        session = EngineSession(info["engine"], data, args.context)
        result["startup_seconds"] = {"file_read": session.read_seconds, "deserialization": session.deserialize_seconds,
                                     "execution_context": session.context_seconds}
        try:
            for _ in range(args.warmups):
                session.execute()
            torch.cuda.synchronize()
            result["raw"] = timed_gpu(session.execute, args.repetitions)
            result["observed_resident_bytes"] = free_before - torch.cuda.mem_get_info()[0]
            if args.profile_iterations:
                write_json(Path(info["engine"]).parent / "inspection.json", inspection(session.engine))
            if args.context is None:
                result["accuracy_same_precision"] = tensor_error(session.bound["noise_pred"], data["expected"])
                if args.reference:
                    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
                    if input_signature(reference["inputs"]) != input_signature(data["inputs"]):
                        raise ValueError("Reference input values differ")
                    result["accuracy_reference"] = tensor_error(session.bound["noise_pred"], reference["expected"])
            if args.save_output:
                torch.save(session.bound["noise_pred"].float().cpu(), artifact_path(args.save_output))
            if args.profile_iterations:
                result["layer_profile"] = profile_layers(session, args.profile_iterations)
            if args.components:
                result["components"] = {
                    "input_conversion": timed_gpu(lambda: {name: value.to(dtype=DTYPES[session.engine.get_tensor_dtype(name)]).contiguous()
                                                          for name, value in session.inputs.items()}, args.repetitions),
                    "output_conversion": timed_gpu(lambda: session.bound["noise_pred"].float(), args.repetitions),
                    "allocation": timed_gpu(lambda: torch.empty_like(session.bound["noise_pred"]), args.repetitions),
                    "shape_setup": timed_gpu(lambda: [session.context.set_input_shape(name, tuple(value.shape))
                                                       for name, value in session.inputs.items()], args.repetitions),
                    "address_setup": timed_gpu(lambda: [session.context.set_tensor_address(name, value.data_ptr())
                                                         for name, value in session.bound.items()], args.repetitions)}
        finally:
            session.close()
        del session
        gc.collect()
        torch.cuda.empty_cache()
        if args.wrapper:
            wrapper = wrapper_runtime(info, prepared)
            inputs = {name: value.cuda() for name, value in data["inputs"].items()}
            if args.context is not None:
                inputs["context"] = inputs["context"][:, :1].expand(1, args.context, 2048).contiguous()
            def execute_wrapper():
                return wrapper(inputs["latent"], inputs["timestep"], context=inputs["context"], y=inputs["adm"])
            try:
                for _ in range(args.warmups):
                    execute_wrapper()
                torch.cuda.synchronize()
                result["wrapper"] = timed_gpu(execute_wrapper, args.repetitions)
            finally:
                wrapper.release()
    result["post_settling"] = idle_window(args.gpu, 2)
    result["post_idle"] = idle_window(args.gpu, args.idle_seconds)
    result["contention"] = contention(result["pre_idle"] + result["post_idle"])
    result["telemetry"] = telemetry.samples
    result["telemetry_errors"] = telemetry.errors
    result["validity"] = {"background_estimate": "pre/post windows only; execution utilization includes this benchmark",
                          "stable_raw_cv_below_10_percent": result["raw"]["cuda_ms"]["cv"] < .1,
                          "not_heavily_contended": result["contention"] not in {"heavily_contended", "unknown"}}
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("precision", "contention", "startup_seconds")}
                     | {"raw_cuda_ms": result["raw"]["cuda_ms"]}), flush=True)


def compare(args):
    results = [json.loads(Path(path).read_text()) for path in args.results]
    def signature(result):
        return {key: result["config"][key] for key in ("gpu_uuid", "driver", "tensorrt", "cuda", "source", "width", "height", "context")} | {
            "torch": result["environment"]["torch"], "inputs": result["input_signature"]}
    signatures = [signature(result) for result in results]
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError("Benchmark source, environment, shapes, or inputs differ")
    grouped = defaultdict(list)
    for result in results:
        key = json.dumps({"config": result["config"], "stream": result["stream"]}, sort_keys=True)
        grouped[key].append(result)
    rows = []
    for group in grouped.values():
        samples = [value for result in group for value in result["raw"]["cuda_samples_ms"]]
        rows.append({"config": group[0]["config"], "stream": group[0]["stream"], "cuda_ms": distribution(samples), "processes": len(group),
                     "independent_builds": len({result["build"] for result in group}),
                     "process_medians_ms": [result["raw"]["cuda_ms"]["median"] for result in group],
                     "contention": dict(Counter(result["contention"] for result in group))})
    report = {"environment_and_workload": signatures[0], "results": rows}
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        if signature(baseline) != signatures[0]:
            raise ValueError("Baseline environment/workload mismatch")
        base_latency = baseline["raw"]["cuda_ms"]["median"]
        checks = []
        for row in rows:
            valid = (set(row["contention"]) == {"relatively_idle"} and row["cuda_ms"]["cv"] <= args.max_cv
                     and baseline["contention"] == "relatively_idle" and baseline["raw"]["cuda_ms"]["cv"] <= args.max_cv)
            ratio = row["cuda_ms"]["median"] / base_latency
            checks.append({"speedup": 1 / ratio, "valid_for_regression_check": valid,
                           "regression": ratio > 1 + args.max_regression if valid else None})
        report["baseline"] = args.baseline
        report["regression_checks"] = checks
        report["regression_threshold"] = args.max_regression
        report["max_cv"] = args.max_cv
    write_json(args.output, report)
    print(json.dumps(rows, indent=2), flush=True)


def native(args):
    modules = project_modules()
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    prepared = json.loads(Path(args.prepared).read_text())
    env = environment(args.gpu)
    data = torch.load(prepared["verification"], map_location="cpu", weights_only=True)
    model, _, _ = modules["provenance"].reconstruct(prepared["source"], output_clip=False, output_vae=False, disable_dynamic=True)
    exported = modules["exporter"].materialize_unet(model, args.precision)
    unet = exported.model.diffusion_model
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in unet.parameters())
    free = torch.cuda.mem_get_info()[0]
    if free < weight_bytes + args.reserve_gib * 2**30:
        raise RuntimeError(f"Insufficient native inference headroom: {free / 2**30:.2f} GiB free, weights {weight_bytes / 2**30:.2f} GiB, reserve {args.reserve_gib} GiB")
    before = idle_window(args.gpu, args.idle_seconds)
    dtype = modules["exporter"].DTYPES[args.precision]
    started = time.perf_counter()
    unet.to("cuda")
    inputs = [value.to(device="cuda", dtype=torch.float32 if key == "timestep" else dtype)
              for key, value in data["inputs"].items()]
    wrapper = modules["exporter"]._ExportUNet(unet)
    torch.cuda.synchronize()
    startup = time.perf_counter() - started
    def execute():
        return wrapper(*inputs)
    with Telemetry(args.gpu) as telemetry:
        for _ in range(args.warmups):
            execute()
        torch.cuda.synchronize()
        result = {"precision": args.precision, "implementation": "native_pytorch_export_attention",
                  "environment": env, "source": prepared["source"], "prepared": str(Path(args.prepared).resolve()),
                  "startup_seconds": startup, "raw": timed_gpu(execute, args.repetitions),
                  "accuracy": tensor_error(execute(), data["expected"])}
    result["pre_idle"] = before
    result["post_settling"] = idle_window(args.gpu, 2)
    result["post_idle"] = idle_window(args.gpu, args.idle_seconds)
    result["contention"] = contention(before + result["post_idle"])
    result["telemetry"] = telemetry.samples
    result["telemetry_errors"] = telemetry.errors
    write_json(args.output, result)
    print(json.dumps({"precision": args.precision, "cuda_ms": result["raw"]["cuda_ms"], "accuracy": result["accuracy"]}), flush=True)


def evaluate(args):
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    with np.load(args.dataset, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("latent", "timestep", "context", "adm")}
    indices = list(range(len(arrays["latent"]))) if not args.indices else [int(value) for value in args.indices.split(",")]
    inputs = [{name: torch.from_numpy(value[index]) for name, value in arrays.items()} for index in indices]
    outputs = []
    if args.build:
        info = json.loads(Path(args.build).read_text())
        session = EngineSession(info["engine"], {"inputs": inputs[0]})
        try:
            for values in inputs:
                for name, tensor in values.items():
                    session.bound[name].copy_(tensor.to(device="cuda", dtype=session.bound[name].dtype))
                outputs.append(session.execute().float().cpu())
        finally:
            session.close()
        source = info["source"]
        precision = info["precision"]
    else:
        modules = project_modules()
        prepared = json.loads(Path(args.prepared).read_text())
        source, precision = prepared["source"], args.precision
        model, _, _ = modules["provenance"].reconstruct(source, output_clip=False, output_vae=False, disable_dynamic=True)
        exported = modules["exporter"].materialize_unet(model, precision)
        wrapper = modules["exporter"]._ExportUNet(exported.model.diffusion_model.to(args.device))
        dtype = modules["exporter"].DTYPES[precision]
        for index, values in zip(indices, inputs):
            converted = [tensor.to(device=args.device, dtype=torch.float32 if name == "timestep" else dtype)
                         for name, tensor in values.items()]
            outputs.append(wrapper(*converted).detach().float().cpu())
            print("Evaluated native input", index, flush=True)
    stacked = torch.stack(outputs)
    torch.save({"outputs": stacked, "indices": indices, "source": source, "precision": precision}, artifact_path(args.save_output))
    result = {"dataset": str(Path(args.dataset).resolve()), "indices": indices, "source": source, "precision": precision,
              "finite": bool(torch.isfinite(stacked).all()), "shape": list(stacked.shape)}
    if not result["finite"]:
        raise ValueError("Nonfinite denoising outputs")
    if args.reference:
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
        if reference["indices"] != indices or reference["source"] != source:
            raise ValueError("Reference dataset indices/source differ")
        result["aggregate_error"] = tensor_error(stacked, reference["outputs"])
        result["per_input_error"] = [tensor_error(output, expected) for output, expected in zip(outputs, reference["outputs"])]
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Materialize the source and export one graph")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--precision", choices=PRECISIONS, required=True)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--context", type=int, default=77)
    p.add_argument("--seed", type=int, default=812)
    p.add_argument("--export-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(function=prepare)
    p = commands.add_parser("build", help="Compile an isolated diagnostic engine and timing cache")
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--level", type=int, choices=range(6), default=3)
    p.add_argument("--workspace", type=int, default=1024)
    p.add_argument("--context-min", type=int, default=77)
    p.add_argument("--context-max", type=int, default=154)
    p.add_argument("--timing-cache")
    p.add_argument("--editable-cache", action="store_true")
    p.add_argument("--avg-timing-iterations", type=int)
    p.add_argument("--reserve-gib", type=float, default=2)
    p.add_argument("--gpu", default="0", help="Physical nvidia-smi GPU index or UUID for telemetry")
    p.add_argument("--idle-seconds", type=float, default=3)
    p.set_defaults(function=build)
    p = commands.add_parser("run", help="Measure uninstrumented execution, then optional separate profiles")
    p.add_argument("--build", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--profile-iterations", type=int, default=0)
    p.add_argument("--components", action="store_true")
    p.add_argument("--stream", choices=("default", "nondefault"), default="default")
    p.add_argument("--wrapper", action="store_true")
    p.add_argument("--reference")
    p.add_argument("--save-output")
    p.add_argument("--context", type=int)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--gpu", default="0")
    p.add_argument("--idle-seconds", type=float, default=3)
    p.set_defaults(function=run)
    p = commands.add_parser("compare", help="Pool matched fresh-process runs by build configuration")
    p.add_argument("results", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--baseline")
    p.add_argument("--max-regression", type=float, default=.1)
    p.add_argument("--max-cv", type=float, default=.1)
    p.set_defaults(function=compare)
    p = commands.add_parser("native", help="Measure native PyTorch with the export attention contract")
    p.add_argument("--prepared", required=True)
    p.add_argument("--precision", choices=PRECISIONS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--gpu", default="0")
    p.add_argument("--idle-seconds", type=float, default=3)
    p.add_argument("--reserve-gib", type=float, default=4)
    p.set_defaults(function=native)
    p = commands.add_parser("evaluate", help="Compare recorded real denoising inputs with native or engine outputs")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--build")
    group.add_argument("--prepared")
    p.add_argument("--dataset", required=True)
    p.add_argument("--indices", help="Comma-separated sample indices; defaults to every recorded sample")
    p.add_argument("--precision", choices=PRECISIONS, default="fp32")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--reference")
    p.add_argument("--save-output", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(function=evaluate)
    return result


if __name__ == "__main__":
    for variable, directory in (("CUDA_CACHE_PATH", "cuda"), ("TORCH_EXTENSIONS_DIR", "extensions"),
                                ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")):
        os.environ.setdefault(variable, str(DEV_ROOT / "tensorrt" / "performance" / "cache" / directory))
    arguments = parser().parse_args()
    try:
        if arguments.command == "run" and arguments.stream == "nondefault":
            with torch.cuda.stream(torch.cuda.Stream()):
                arguments.function(arguments)
        else:
            arguments.function(arguments)
    except Exception as error:
        destination = artifact_path(arguments.output)
        if arguments.command in {"build", "prepare"}:
            destination = destination / "failure.json"
        write_json(destination, {"status": "fail", "command": arguments.command,
                                 "error": str(error), "traceback": traceback.format_exc()})
        raise
