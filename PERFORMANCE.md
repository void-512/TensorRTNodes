# TensorRT 11 SDXL performance

Measured on 2026-10-28/29 using an RTX PRO 6000 Blackwell Server Edition (SM 12.0, 96 GiB), driver 615.71.10, TensorRT 11.4.0.106, PyTorch 2.15.0+cu132 and ONNX 1.23.2. The benchmark CLI supports FP32, FP16 and BF16. It is separate from normal node execution and stores experimental artifacts under `ComfyUI/dev_temp`.

## Workload and measurement conditions

The source was `cocoIllustriousNoobai_v55.safetensors`, followed by `add_contrast_XL` (MODEL 0.8, CLIP 0.6), `add_brightness_XL` (-0.3, 0.4), repeated `add_contrast_XL` (0.2, -0.1), and `add_saturation_XL` (0, 0). Loading dtypes were MODEL FP32, CLIP FP16 and VAE FP32. Engine precision does not change source provenance or Model Switcher reconstruction.

Inputs use seed 812: latent `[1,4,128,128]`, timestep `[500]`, context `[1,77,2048]`, ADM `[1,2816]`; image 1024×1024, B1, context profile 77/77/154. Every engine receives the same values rounded at its bindings.

GPU 0 was compute idle but inaccessible because another user's MPS server held its Exclusive_Process context. Measurements used GPU 1 while compute idle; vLLM still held approximately 84,800 MiB, leaving about 12,483 MiB before our context. These are observed-idle shared-GPU measurements, not exclusive peak throughput. No workloads, clocks, power settings or MPS services were changed.

The harness records utilization, available VRAM, clocks, power, temperature, throttle reasons and competing processes. Background classification uses p95 utilization during three-second pre/post windows: ≤5% relatively idle, >5–30% moderately contended, >30% heavily contended. A two-second settling window excludes our own trailing utilization. This cannot exclude a competing burst during inference.

Timing uses 20 warmups and 100 synchronized CUDA-event measurements per process. Layer profiling runs afterward. Reports include raw samples, median, nearest-rank p95, mean, standard deviation, coefficient of variation (CV), extrema, process count and independent build count. Input signatures reject comparisons with different values, sources, shapes or environments.

## Builder results

All rows use 256 MiB workspace unless specified and relatively idle background windows.

| Configuration | Median / p95 (ms) | Mean / stddev (ms) | Min / max (ms) | CV |
| --- | ---: | ---: | ---: | ---: |
| FP16 level 0 | 100.156 / 100.255 | 100.171 / 0.125 | 100.045 / 101.288 | 0.125% |
| BF16 level 0 | 33.116 / 33.141 | 33.120 / 0.031 | 33.092 / 33.394 | 0.094% |
| FP16 level 3 | 25.516 / 25.555 | 25.519 / 0.025 | 25.467 / 25.616 | 0.098% |
| FP16 level 3, fresh process | 25.527 / 25.555 | 25.529 / 0.014 | 25.503 / 25.571 | 0.053% |
| FP16 level 3, cached independent build | 25.535 / 25.646 | 25.549 / 0.041 | 25.490 / 25.676 | 0.162% |
| BF16 level 3 | 25.901 / 26.016 | 25.916 / 0.044 | 25.849 / 26.070 | 0.171% |
| BF16 level 3, fresh process | 25.894 / 25.917 | 25.905 / 0.099 | 25.872 / 26.881 | 0.383% |
| FP16 level 3, 1024 MiB | 25.557 / 25.636 | 25.570 / 0.032 | 25.521 / 25.674 | 0.125% |
| FP16 level 5 | 25.549 / 25.598 | 25.563 / 0.102 | 25.505 / 26.547 | 0.400% |

Level 3 improves FP16 by 3.925× and closes the approximately threefold level-0 FP16/BF16 gap. FP16 process medians differ by less than 0.1%. A practical diagnostic regression threshold is 3% with CV ≤3% on matched relatively idle runs; it is not a guarantee under contention.

FP16 compilation took 72.37 s at level 0 and 129.47 s at level 3. Reusing the compatible timing cache reduced an independent level-3 build to 49.96 s, preserving all 1,233 named layer tactics and bit-identical outputs on six real denoising inputs. The ordinary cache contained 83 queryable entries. Cache metadata checks GPU UUID, driver, TensorRT/CUDA, model source, precision, profile and builder settings; mismatched caches are rejected.

The [TensorRT builder documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_builder_config.html) explains that level 0 picks the first successful tactic and restricts optimization. It should not be used to infer hardware precision rankings. Level 3 and 1024 MiB remain the node defaults. Level 5 and larger workspace had no material benefit here, so a 4096 MiB sweep was not justified. Individual editable-cache tactic pinning was not established by these experiments.

## Graph and kernel evidence

FP16 and BF16 have identical ONNX connectivity, 13,833 nodes, 1,680 initializers, 2,567,463,684 initializer elements, shapes and cast counts; low precision tensor types differ as intended. Each level-0 engine contains 1,237 execution layers and 140 fused attention kernels.

Mean IProfiler times include instrumentation and must not be pooled with raw latency:

| Operator group | FP16 level 0 | BF16 level 0 | FP16 level 3 | BF16 level 3 |
| --- | ---: | ---: | ---: | ---: |
| Convolution | 47.145 ms | 7.640 ms | 5.165 ms | 5.346 ms |
| Linear/GEMM | 48.263 ms | 20.984 ms | 16.608 ms | 16.864 ms |
| Fused attention | 3.846 ms | 3.717 ms | 3.825 ms | 3.732 ms |
| Normalization/fused pointwise | 2.261 ms | 2.244 ms | 2.114 ms | 2.119 ms |
| Other | 1.656 ms | 1.600 ms | 0.988 ms | 0.993 ms |

Convolution explains about 59% of the level-0 gap and GEMM about 41%. FP16 selects indexed implicit-GEMM convolution (`sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16...`) and predominantly `128x256x32` GEMMs. BF16 selects different convolution families and `128x128x64` GEMMs with FP32 accumulation indicated in tactic names. Level 3 selects faster FP16 convolution algorithms and varied GEMM tiles, eliminating the costly indexed convolution family.

Nsight Systems captures independently confirm 1,236 kernels totaling 101.67 ms for FP16 level 0 versus 33.49 ms for BF16. FP16 level 3 launches 1,292 kernels totaling 25.09 ms: launch count alone does not explain the improvement. These are instrumented single captures. CUDA launch time overlaps GPU work, and synchronization wait must not be added again as independent overhead.

Nsight Compute counter access failed with `ERR_NVGPUCTRPERM`; CPU sampling was unavailable as well. Occupancy, measured bandwidth efficiency and actual Tensor Core utilization therefore remain unverified. Tactic names identify kernel families, not measured hardware utilization.

## Accuracy, startup and integration

Six real positive/negative denoising inputs at early, middle and late timesteps were compared with native FP32 on CPU:

| Engine | RMSE | Relative RMSE | Maximum absolute error |
| --- | ---: | ---: | ---: |
| FP16 level 0 | 0.003994 | 0.4516% | 0.05460 |
| FP16 level 3 | 0.004053 | 0.4583% | 0.06649 |
| BF16 level 0 | 0.007803 | 0.8823% | 0.12533 |
| BF16 level 3 | 0.007914 | 0.8948% | 0.14253 |

All outputs were finite. Independently cached FP16 builds matched bit for bit. Eight-step full-generation tests also completed for two seeds, including VAE decoding and Model Switcher reconstruction. FP16 level-3 final-latent relative RMSE was 8.50% / 8.25%, and decoded-image RMSE 0.02210 / 0.02169 on floating-point [0,1] pixels. BF16 was 12.85% / 11.09% and 0.03009 / 0.02709. These limited samples demonstrate functional regression checks, not exact generation equivalence or broad image-quality certification. Short and 154-token long-prompt workflows passed.

FP16 level-3 raw execution measured 25.52 ms versus 25.77 ms through TensorRTUNet. Conversion, allocation, shape setup and address setup each cost roughly 0.007–0.029 ms. A nondefault caller stream gave no material improvement. Existing context reuse remains; persistent output buffers and CUDA Graph capture were not justified by about 1% wrapper overhead.

Representative FP16 startup was 0.619 s file read, 0.286 s deserialization and 0.648 s context creation. The engine occupied 4.829 GiB and observed resident memory about 5.15 GiB; BF16 was similar. Cold source loading, CLIP initialization and engine reload after Model Switcher caused substantial workflow variation. Warm B2 CFG UNet calls were about 51 ms, while complete second executions including eight steps, VAE decode and reconstruction took about 3.33 s. Cold workflow observations are not steady-state throughput claims.

A new full-size FP32 GPU comparison was rejected by the memory headroom check: 11.65 GiB free versus 9.57 GiB graph artifacts + 256 MiB workspace + 2 GiB reserve. Native GPU FP32 also lacked headroom. Historical FP32/TF32 measured 303.039 ms on a more heavily shared GPU and is not a matched new baseline. Its inspector showed mixed TF32 GEMM and FP32 convolution families. The normal TF32 policy remains enabled; no smaller model was substituted to manufacture a new result.

## Reproduce

Select a usable GPU with `CUDA_VISIBLE_DEVICES` and pass its physical nvidia-smi index or UUID to `--gpu`. The harness verifies UUID agreement. Source JSON can be a version-3 package manifest or its source object. Use fresh output folders for independent builds.

```sh
python custom_nodes/TensorRTNodes/benchmark.py prepare \
  --source models/tensorrt/selected_model/manifest.json \
  --precision fp16 --export-device cuda --output dev_temp/experiment/prepared
python custom_nodes/TensorRTNodes/benchmark.py build \
  --prepared dev_temp/experiment/prepared/prepared.json \
  --output dev_temp/experiment/build-a --gpu 1 --level 3 --workspace 1024
python custom_nodes/TensorRTNodes/benchmark.py run \
  --build dev_temp/experiment/build-a/build.json --gpu 1 \
  --output dev_temp/experiment/run-a.json --wrapper --components --profile-iterations 10
python custom_nodes/TensorRTNodes/benchmark.py build \
  --prepared dev_temp/experiment/prepared/prepared.json \
  --output dev_temp/experiment/build-b --gpu 1 --level 3 --workspace 1024 \
  --timing-cache dev_temp/experiment/build-a/timing.cache
python custom_nodes/TensorRTNodes/benchmark.py run \
  --build dev_temp/experiment/build-b/build.json --gpu 1 \
  --output dev_temp/experiment/run-b.json
python custom_nodes/TensorRTNodes/benchmark.py compare \
  dev_temp/experiment/run-a.json dev_temp/experiment/run-b.json \
  --baseline dev_temp/experiment/run-a.json --max-regression .03 --max-cv .03 \
  --output dev_temp/experiment/comparison.json
```

Repeat runs in fresh processes. `native` measures the native export-attention contract; `evaluate` compares recorded real inputs against native or engine references. Diagnostic files remain local under `dev_temp`, and benchmark instrumentation is absent from normal generation. Node-generated intermediates follow the RAM/disk cleanup policy documented in the README.
