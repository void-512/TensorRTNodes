# TensorRT Nodes

TensorRT 11.x, CUDA-enabled PyTorch, and ONNX are required. Supports base SDXL UNets, including Illustrious checkpoints, fixed image resolutions, dynamic context lengths, and independent B1 execution of larger batches. ControlNet and model merges are unsupported.

Connect native **Load Checkpoint** (`CheckpointLoaderSimple`) and sequential **Load LoRA** nodes to **Export TensorRT Package (Fused Model)**. Supply matching MODEL/CLIP outputs and the checkpoint's VAE. Model-only LoRA loaders and zero, negative, fractional, repeated, and asymmetric MODEL/CLIP strengths are supported.

The exporter reads its actual ComfyUI dynamic prompt, follows native loader connections, records filenames and both LoRA strengths in order, and checks the input patchers' checkpoint reload factories. Filenames and strengths must be literal loader settings. Independent MODEL/CLIP branches are combined in a deterministic order, with zero effective strength for an unused component. Unknown transformations, different base checkpoints, and linked source settings fail explicitly; no global loader hooks or manually duplicated source configuration are used.

Version 3 records the actual MODEL, CLIP, and VAE loading dtypes in `source.load_dtypes`. Native reconstruction uses those dtypes even when launch flags change. These describe the supplied source objects; changing engine precision leaves them unchanged. A loading-dtype change requires a separate source package.

| Precision | Graph preparation |
| --- | --- |
| `fp32` | FP32 weights and computation graph; preserves TensorRT's default TF32 policy. TF32 execution is not guaranteed for every operation. |
| `fp16` | FP16 graph with existing explicit higher precision operations preserved. |
| `bf16` | BF16 graph with existing explicit higher precision operations preserved; requires SM80 or newer. |

The exporter uses TensorRT 11's [strongly typed precision rules](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-patterns.html). Each build must deserialize, produce finite output, match the export model within its precision tolerance, and expose the requested precision in engine inspection before publication.

Patched weights are materialized through ComfyUI's state-dict API into a separate UNet in memory before ONNX tracing, resolving deferred patches without tracing the shared input model. No merged checkpoint is written.

`intermediate_directory` selects temporary disk storage, defaulting to `models/tensorrt/tmp`. Absolute paths and `~` are supported; relative paths use the ComfyUI directory. The field is locked and its value is ignored when RAM mode is enabled.

`keep_intermediates_in_ram` defaults to false and is automatically locked off when the server has no writable `/dev/shm`. Enable it to put all build intermediates there, including ONNX graphs, exported weights, verification tensors, build reports and the candidate engine. RAM mode never creates or uses `intermediate_directory`. On Windows or other systems without `/dev/shm`, a manually mounted RAM disk can still be selected as `intermediate_directory` with RAM mode disabled.

Temporary contents are removed after success, failure or cancellation in both modes. No intermediate checkpoints, build reports or failure diagnostic files are retained; errors appear in the normal execution error. Failed-export tracebacks release private model references as well. Final engine publication and rollback use a short-lived transaction on the output filesystem for atomic replacement. Package locks live beside packages under `models/tensorrt/.locks`. Normal exports do not use `dev_temp`; it is reserved for tests and diagnostics.

`package_name` selects a folder under `models/tensorrt`; blank uses the checkpoint stem. Choose a separate folder for each different checkpoint/LoRA chain. Existing sources are compared directly, including LoRA order, both strengths, and source loading dtypes; a mismatch is rejected. Precision, resolution, and context profile do not change source identity.

```text
models/tensorrt/selected_model/
  manifest.json
  engines/
    selected_model_1024x1024.engine
    selected_model_1024x1536.engine
```

Engine filenames use the sanitized checkpoint stem and image dimensions only. Rebuilding a resolution replaces its engine and manifest entry regardless of precision or context limits. Different resolutions coexist. Publication uses package locks, staged replacement, and rollback on replacement failure. An interrupted process can leave a lock; the error identifies its path. Remove it only after that process has exited.

Example manifest:

```json
{
  "format_version": 3,
  "source": {
    "base_model": "selected_model.safetensors",
    "load_dtypes": {"model": "float16", "clip": "float32", "vae": "float32"},
    "loras": [
      {"name": "style.safetensors", "strength_model": 0.8, "strength_clip": 0.6}
    ]
  },
  "engines": [
    {
      "path": "engines/selected_model_1024x1024.engine",
      "image_size": [1024, 1024],
      "precision": "fp16",
      "requirements": {
        "tensorrt_version": "11.4.0.106",
        "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "compute_capability": [12, 0]
      }
    }
  ]
}
```

Context profiles and tensor shapes/types are read from the engine itself. Plans require the recorded TensorRT version and GPU model/compute capability. Older manifests must be rebuilt into a new package folder; their source loading dtypes cannot be recovered reliably. Manifests contain no weight hashes, fingerprints, merged checkpoint paths, ONNX references, or build history. Keep source files available and unchanged; changing their contents under the same filenames requires rebuilding every resolution in that package.

**Load TensorRT Engine** takes `package` and returns MODEL, CLIP, VAE. It constructs a TensorRT-backed MODEL without loading a duplicate PyTorch UNet into VRAM, loads source CLIP/VAE, and reapplies CLIP LoRAs. Engine I/O is bound using actual FP32/FP16/BF16 tensor types. Unsupported resolutions and context lengths fail explicitly. Cached wrappers detect engine replacement. ComfyUI offloading releases execution contexts and engine allocations.

**Model Switcher** takes required `trigger: ANY` and `engine_model: MODEL`, and returns only MODEL. Its non-lazy trigger is a normal execution dependency. It releases TensorRT contexts and engines, unloads the managed model, and reconstructs a conventional model from the original checkpoint and ordered LoRAs using native ComfyUI patching. Future TensorRT execution can deserialize the engine again. Missing checkpoint/LoRA files produce explicit errors and configured model search directories are respected.

Run tests with ComfyUI's Python from the ComfyUI root:

```sh
python custom_nodes/TensorRTNodes/tests/test_regressions.py
python custom_nodes/TensorRTNodes/tests/test_benchmark.py
TENSORRT_TEST_GPU=1 python custom_nodes/TensorRTNodes/tests/test_gpu.py
```

The frontend control-locking tests use Node's built-in test runner: `node --test custom_nodes/TensorRTNodes/tests/test_storage_widgets.mjs`.

GPU tests build small synthetic engines and test precision inspection, dynamic contexts, batching, replacement, output types, and repeated release/reload. They are not SDXL benchmarks. The GPU suite covers FP32, FP16 and BF16; `TENSORRT_TEST_PRECISIONS` can select a subset. Test artifacts stay under `dev_temp/tensorrt`.

## Validation and measured performance

The [performance investigation](PERFORMANCE.md) documents the supported precisions and the standalone `benchmark.py` diagnostics. Matched 1024×1024, B1 measurements on an RTX PRO 6000 Blackwell Server Edition with TensorRT 11.4.0.106 found:

| Precision | Level 0 median | Level 3 median |
| --- | ---: | ---: |
| FP16 | 100.16 ms | 25.52 ms |
| BF16 | 33.12 ms | 25.90 ms |

These were observed-idle shared-GPU measurements, not exclusive hardware performance. Layer and kernel profiling attributed the level-0 gap to convolution and GEMM tactic selection. An independent cached FP16 rebuild reproduced the tactics and measured 25.54 ms. Larger workspace and optimization level 5 gave no material improvement; the UI defaults remain level 3 and 1024 MiB workspace.

Validation covers source reconstruction and loading dtypes, package replacement/rollback, temporary-storage cleanup on success/failure/cancellation, RAM-control locking, dynamic context lengths, batching and engine release/reload. Standalone benchmark artifacts and tests use `ComfyUI/dev_temp`; normal generation uses the selected intermediate storage.
