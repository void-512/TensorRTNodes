# TensorRT 11 SDXL performance investigation

Measured 2026-10-28/29 on an RTX PRO 6000 Blackwell Server Edition (SM 12.0, 96 GiB), driver 615.71.10, TensorRT 11.4.0.106, PyTorch 2.15.0+cu132, and ONNX 1.23.2. This report separates measured implementation effects from hardware claims and unsuccessful experiments. Raw JSON, inspector output, logs, calibration records, and experiment scripts are in `ComfyUI/dev_temp/tensorrt/performance`.

## Findings

The FP16/BF16 anomaly is primarily a **builder tactic problem**. At optimization level 0, matched FP16 execution took 100.16 ms and BF16 took 33.12 ms. Level 3 reduced them to 25.52 and 25.90 ms. The graphs have identical topology; FP16's level-0 convolution and matrix multiplication tactics account for almost the entire gap. BF16 is not intrinsically three times faster on this GPU.

An independently compiled FP16 engine using the compatible timing cache chose the same 1,233 layer tactics and measured 25.54 ms. Increasing workspace from 256 MiB to 1 GiB or optimization from level 3 to 5 did not materially improve this workload. The production UI already defaults to level 3 and 1 GiB; retain those defaults. A separate validated FP8 graph normalization restores Half attention around calibrated FP8 linears, reducing the matched level-0 latency from 101.00 to 84.13 ms (1.20×). FP8 level 3 still fails numerical validation; it is not presented as a usable optimized engine.

## Root-cause assessment

| Historical observation | Evidence and interpretation | Engineering status / uncertainty |
| --- | --- | --- |
| BF16 ≈3× faster than FP16 | Equivalent graphs; 47.1 vs 7.6 ms convolution and 48.3 vs 21.0 ms GEMM at level 0; level 3 closes the gap | Fixable tactic-search restriction, confirmed. |
| FP8 approximately equals FP16 | FP8 linears improve, but Half convolution remains 47 ms and loss of fused attention adds unfused matrix/Softmax work | Mixed precision plus a graph-fusion defect. Restoring Half attention gives 101.00→84.13 ms at level 0; the separate level-3 numerical failure remains unresolved. |
| NVFP4 slower than BF16 | Level-0 Half convolution dominates despite genuinely faster FP4 linears | Level 3 removes this bottleneck: 22.31 vs 25.90 ms. Scaling/quantization and mixed operators still limit gains; quality remains separately qualified. |
| FP16 only 1.44× faster than FP32 | Historical shared-GPU comparison used poor level-0 FP16 tactics; FP32 had mixed TF32/FP32 kernels and unfused work | Historical ratio is not a hardware ratio. New full-size FP32 comparison is constrained by free VRAM. |
| Smaller engine is not always faster | Inspector and timings expose unquantized operators, fusion differences and dynamic quantization costs | Workload/kernel limitations remain after builder tuning; storage size alone does not determine latency. |
| Historical absolute latencies were much higher | New observed-idle level-0 FP16/BF16/NVFP4 latencies are about half the old shared-GPU numbers | Contention/environment artifact is material; its exact historical contribution cannot be reconstructed. |

## Baseline validity and hypotheses

The source was `cocoIllustriousNoobai_v55.safetensors`, followed, in order, by `add_contrast_XL` (MODEL 0.8, CLIP 0.6), `add_brightness_XL` (-0.3, 0.4), repeated `add_contrast_XL` (0.2, -0.1), and `add_saturation_XL` (0, 0). Recorded loading dtypes were MODEL FP32, CLIP FP16, VAE FP32. Engine precision never changes this source identity. No manifest, filename, package replacement, or switcher contract was changed.

The first baseline uses deterministic seed 812: latent `[1,4,128,128]`, timestep `[500]`, context `[1,77,2048]`, ADM `[1,2816]`; image 1024×1024, batch 1, context profile 77/77/154. Every precision receives the same FP32 input values rounded only at its bindings. A second comparison uses exactly the same recorded held-out denoising input across precisions. Real calibration contains 96 inputs from six prompts, eight Euler steps, CFG 3, positive and negative conditioning; 16 inputs from a seventh prompt are held out.

| Hypothesis | Required evidence | Result |
| --- | --- | --- |
| FP16 export has extra computation/casts | Compare graph connectivity, shapes, initializers and casts | Rejected for FP16/BF16: same 13,833 nodes, 1,680 initializers, 2,567,463,684 initializer elements, parameter/input shapes, operator and cast counts. Only low precision tensor types differ. |
| BF16 uses better attention | Inspector tactics and attention timings | Rejected as main cause: both use 140 fused MHA kernels and approximately 3.7–3.8 ms profiled attention. |
| Level 0 chooses poor FP16 tactics | Corresponding layer timings and level-3 rebuild | Confirmed: convolution and GEMM dominate the gap; level 3 eliminates it. |
| Workspace is the main restriction | Change workspace alone | Not supported: 256→1024 MiB at level 3 gave 25.52→25.56 ms. |
| Contention explains the ranking | New observed-idle windows, clocks and repeated processes | It affects absolute historical numbers, but the level-0 ranking persists. |
| Python, allocation or stream overhead dominates | Raw bindings vs wrapper, component timings, nondefault stream | Rejected for this workload; wrapper overhead is about 1% at level 3. |
| Low precision means low precision end to end | Actual optimized tensor types, tactics and operator timings | Rejected: quantized linears coexist with higher precision convolution, normalization and attention. |
| A newer ModelOpt fixes removed Triton block pointers | Isolated export/build/run on the actual installed stack | Default 0.48rc1 still fails; stable 0.47's SDK CUDA backend passes both mechanics and full UNet export/build/run without replacing Triton. |

GPU 0 appeared idle but was in Exclusive_Process mode, held by another user's MPS server. Direct access failed with CUDA error 46; connecting to its MPS pipe queued and timed out. No permissions, servers, workloads, clocks or power settings were changed. GPU 1 was usable and usually compute idle, with approximately 84,800 MiB resident in vLLM and 12,483 MiB free before this process. All reported new timings use GPU 1. They are **observed-idle shared-GPU measurements**, not exclusive peak hardware measurements.

Representative telemetry spans SM clocks 2295–2415 MHz, fixed memory clock 12481 MHz and temperatures 37–58°C. Sampled thermal, power-cap and power-brake flags were inactive in the primary comparisons (`telemetry-summary.json`); short unsampled events cannot be excluded. Clock changes of this scale do not explain a threefold gap.

The harness samples GPU/memory utilization, free VRAM, SM/memory clocks, power, temperature and throttle reasons every 0.5 seconds, and records process snapshots. Background classification uses the p95 GPU utilization in three-second windows before and after the measurement: ≤5% relatively idle, >5–30% moderately contended, >30% heavily contended; unavailable telemetry is unknown. A two-second settling window excludes trailing utilization caused by our own work. These windows cannot rule out a competing burst during inference. Resident allocations are recorded even when utilization is zero.

Unprofiled latency uses 20 warmups and 100 synchronized CUDA-event measurements. Layer profiling occurs afterward and is never pooled into raw timing. Statistics include median, nearest-rank p95, mean, population standard deviation, coefficient of variation, extrema and count. Raw samples remain in each JSON. An initial FP16 run falsely classified its own trailing utilization as contention; the primary level-0 latency comes from the corrected second process. No raw evidence was silently overwritten.

## Matched nonquantized results

All rows below use the deterministic input, 256 MiB workspace unless stated, and relatively idle background windows.

| Precision/configuration | Median / p95 (ms) | CV | Executions | Fresh processes / builds |
| --- | ---: | ---: | ---: | ---: |
| FP16 level 0 | 100.156 / 100.255 | 0.125% | 100 | 1 / 1 |
| BF16 level 0 | 33.116 / 33.141 | 0.094% | 100 | 1 / 1 |
| FP16 level 3, first process | 25.516 / 25.555 | 0.098% | 100 | 1 / 1 |
| FP16 level 3, fresh process | 25.527 / 25.555 | 0.053% | 100 | 1 / same build |
| FP16 level 3, cached independent build | 25.535 / 25.646 | 0.162% | 100 | 1 / second build |
| BF16 level 3, first process | 25.901 / 26.016 | 0.171% | 100 | 1 / 1 |
| BF16 level 3, fresh process | 25.894 / 25.917 | 0.383% | 100 | 1 / same build |
| FP16 level 3, workspace 1024 MiB | 25.557 / 25.636 | 0.125% | 100 | 1 / 1 |
| FP16 level 5, workspace 256 MiB | 25.549 / 25.598 | 0.400% | 100 | 1 / 1 |

Thus the controlled FP16 improvement is **3.925×**, and BF16 improves approximately 1.279×. Fresh FP16 process medians span less than 0.1%; BF16 medians differ by about 0.03%. A practical regression threshold is 3% on matched relatively idle runs with CV ≤3%, leaving margin above the observed variation. It is a diagnostic threshold, not a guarantee under contention.

## Historical reference and matched before/after

The new results in this table all use the **same real held-out input**, not a different workload for each precision. Historical values are supplied for context only: they used a heavily shared GPU and different calibration. The improvement column compares the newly measured matched pair shown in parentheses, never the historical figure. Every matched row has 100 executions in each process and relatively idle background windows.

| Precision | Historical median (ms) | New median (ms) | Matched improvement / baseline→new (ms) | New engine (GiB) | Main bottleneck |
| --- | ---: | ---: | --- | ---: | --- |
| FP32/TF32 | 303.039 | Not safely feasible | No new ratio | 9.573 historical | Mixed TF32/FP32; new profiling limited by VRAM |
| FP16 | 210.019 | 25.533 | 3.925× (100.230→25.533) | 4.829 | Linear/GEMM |
| BF16 | 68.903 | 25.898 | 1.280× (33.158→25.898) | 4.831 | Linear/GEMM |
| FP8 | 211.135 | 84.130 | 1.200× (100.996→84.130) | 2.729 | Half convolution; level 3 fails validation |
| NVFP4 | 134.404 | 22.335 | 2.929× (65.413→22.335) | 1.830 | Mixed compute and dynamic scaling |

FP16/BF16/NVFP4 pairs change level 0→3 with 256 MiB workspace. The FP8 pair changes only Softmax Q/DQ normalization at level 0; FP8 level 3 is excluded because it produces nonfinite output. Smaller engines are not automatically faster: even the improved FP8 path remains slower than optimized FP16/BF16/NVFP4.

### Complete steady-state statistics

`matched-comparison.json` retains pooled statistics, process/build counts and raw-report provenance; `fp16-regression-check.json` passes the 3% regression/CV criteria. The following deterministic-input rows supplement the earlier median/p95 table. Values are milliseconds; each row contains 100 executions.

| Configuration | Mean | Standard deviation | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| FP16 L0 | 100.1714 | 0.1253 | 100.0451 | 101.2883 |
| BF16 L0 | 33.1200 | 0.0310 | 33.0918 | 33.3939 |
| FP16 L3 A | 25.5193 | 0.0250 | 25.4670 | 25.6158 |
| FP16 L3 A fresh | 25.5291 | 0.0135 | 25.5027 | 25.5707 |
| FP16 L3 B cached | 25.5487 | 0.0414 | 25.4896 | 25.6757 |
| BF16 L3 A | 25.9164 | 0.0444 | 25.8494 | 26.0696 |
| BF16 L3 A fresh | 25.9047 | 0.0991 | 25.8723 | 26.8810 |
| FP16 L3 1024 MiB | 25.5697 | 0.0319 | 25.5214 | 25.6743 |
| FP16 L5 256 MiB | 25.5635 | 0.1022 | 25.5052 | 26.5473 |
| FP8 L0 original | 101.1768 | 0.7817 | 100.9480 | 108.9103 |
| NVFP4 L0 | 65.4210 | 0.0868 | 65.2187 | 65.6968 |
| NVFP4 L3 | 22.3089 | 0.0215 | 22.2769 | 22.4723 |

Full real-input, stream, profiler, native and workflow results remain in their named JSON files; invalid runs contain errors instead of misleading successful statistics. Input-byte signatures prevent pooling different deterministic/recorded values; older reports were copied into `audited-results` with signatures reconstructed from retained verification inputs, leaving originals unchanged.

## Layer and tactic evidence

Mean IProfiler times below include instrumentation and therefore exceed raw execution. Groups are reconstructed from inspector metadata, including fused GEMM and convolution layers; generic `fusion` layers are not all counted as pointwise operations.

| Group | FP16 level 0 (ms) | BF16 level 0 (ms) | FP16 level 3 (ms) | BF16 level 3 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Convolution | 47.145 | 7.640 | 5.165 | 5.346 |
| Linear/GEMM | 48.263 | 20.984 | 16.608 | 16.864 |
| Fused attention | 3.846 | 3.717 | 3.825 | 3.732 |
| Normalization/fused pointwise | 2.261 | 2.244 | 2.114 | 2.119 |
| Other | 1.656 | 1.600 | 0.988 | 0.993 |

Convolution explains about 59% of the profiled level-0 FP16/BF16 difference; linear algebra explains about 41%. Normalization, casts and attention are essentially unchanged. Both level-0 engines have 1,237 execution layers: 448 GEMM, 51 convolution/correlation, 737 generated kernels and one shape layer.

* FP16 level 0 selects 443 GEMMs in the `sm80_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x256x32_stage3_warpsize2x4x1_tensor16x8x16_aligna2_alignc2` family. Thirty-six convolutions use `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x16_aligna2_alignc2`; the slowest are about 2.89 ms each.
* BF16 level 0 chooses `sm80_xmma_gemm_bf16bf16_bf16f32_f32_nn_n_tilesize128x128x64_stage3_warpsize2x2x1_tensor16x8x16` and SM86 implicit-GEMM convolution families with FP32 accumulation indicated by tactic names. Comparable slow convolutions are about 0.53 ms.
* FP16 level 3 uses 1,233 execution layers, including 294 fused layers, 227 GEMM and 48 correlation layers. It selects different convolution algorithms, diverse matrix tiles and FP32-accumulating variants where useful. The indexed convolution family responsible for the large level-0 cost disappears.

Nsight Systems 2026.3.2 CUDA traces independently confirm the kernel attribution. A captured warm execution launches 1,236 kernels for each level-0 engine, with summed GPU kernel duration 101.67 ms (FP16) versus 33.49 ms (BF16). The FP16 slow GEMM family consumes 47.85 ms and its indexed convolution family 46.39 ms. Level-3 FP16 launches 1,292 kernels but totals only 25.09 ms: fewer inspector layers do not necessarily mean fewer kernel launches, and launch count alone does not explain the speedup. These are instrumented single captures, not the statistical baseline. CUDA API launch durations overlap GPU work; `cudaDeviceSynchronize` measures waiting for it and must not be added as independent inference overhead.

Nsight Compute 2026.3.1 was actually attempted and failed with `ERR_NVGPUCTRPERM`. The driver restricts profiling counters to administrators; no sudo or system changes were used. CPU sampling was also unavailable (`perf_event_paranoid=3`), so Systems capture used `--sample=none --cpuctxsw=none`. No occupancy, memory-bandwidth efficiency or precise Tensor Core utilization is claimed.

Tactic names expose implementation families and some accumulation/layout details; they are not hardware counter measurements. A small two-matrix gated-FFN mechanics test also reproduces the level-0 FP16/BF16 tactic difference, but does not replace full-model evidence.

## Builder configuration and reproducibility

[TensorRT's builder documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_builder_config.html) specifies that level 0 picks the first successful tactic, disables dynamic kernel generation and ignores timing-cache optimization. Level 3 is the normal default. A precision-dependent first tactic can therefore create a large ranking difference without a graph defect.

FP16 build time was 72.37 s at level 0 and 129.47 s at level 3. BF16 level 3 took 109.71 s. FP16 level 3 with 1 GiB workspace took 127.38 s; level 5 with 256 MiB took 123.20 s. Timing varies with compilation/cache state; build time is not inferred from inference speed. The independently serialized level-3 cached rebuild took 49.96 s versus 129.47 s initially, consumed a 10,105,344-byte cache and reproduced all 1,233 named tactics exactly. Engine outputs and held-out errors are checked separately.

The harness uses the documented [TensorRT 11 timing-cache API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/TimingCache.html), using `queryKeys()` and optionally `EDITABLE_TIMING_CACHE`. The ordinary cache contained 83 queryable entries even though a TensorRT log said “Serialized 0 timing cache entries”; the public query is the evidence. Cache metadata binds GPU UUID, driver, TensorRT/CUDA, source, precision, profile, export device, workspace, level and timing options. Mismatched caches are rejected; `ignore_mismatch=False`. The full-model replay used an ordinary compatible cache; editing or pinning individual entries was not established by that experiment. No obsolete algorithm selector is used. This is experimental harness behavior, not an extra production package/cache format.

Recommendation: retain production **level 3 / 1024 MiB**. Level 0 is useful for fast diagnostic compilation but unsuitable for performance comparisons. Level 5 and persistent runtime buffers add no demonstrated benefit here. A 4096 MiB build was not justified after 1024 MiB showed no gain, especially with another workload's VRAM resident.

## FP8 and NVFP4 investigation

The original optimized FP8 engine genuinely executes FP8 activation-and-weight GEMMs; it is not merely smaller stored weights. At level 0 it takes 101.09 ms versus FP16's 100.16 ms. Profiling gives 47.11 ms convolution, 30.73 ms quantized linears, 14.60 ms attention matrix work, 6.82 ms attention Softmax, 2.84 ms quantization-labelled kernels and approximately 3.44 ms remaining work. Q/DQ fused into GEMM is charged to that GEMM, not counted again as conversion overhead.

ModelOpt inserts FP8 Q/DQ after 140 Softmax nodes to facilitate fully quantized MHA. This integration calibrates Linear inputs and weights, while their outputs remain Half. Consequently TensorRT does not recognize the same fused Half MHA as in FP16/BF16/NVFP4: FP8 has 140 unfused Softmax kernels, additional QK/AV matrix operations and 1,952 execution layers in the historical inspector. FP8 convolution remains Half and selects the same slow level-0 convolution tactics. Even perfect zero-cost quantized linears leave a roughly 47 ms convolution floor, already slower than the complete 33 ms BF16 level-0 engine; an all-linear speedup cannot overcome that floor.

The retained production change bypasses only ModelOpt-added Softmax Q/DQ pairs feeding MatMul, leaving every calibrated FP8 Linear weight, activation pair and scale unchanged. It restores all 140 fused MHA layers and reduces execution layers from 1,952 to 1,602. On identical real inputs at level 0, latency improves **100.996→84.130 ms**. A second process measured 84.139 ms and an independently built engine 84.138 ms; all 1,602 named tactics and all six held-out outputs match exactly across those builds. The production normalizer produces a byte-identical ONNX graph to the separately measured variant (`production-rewrite-equivalence.json`).

FP8 profiling changes attention from 14.56 ms matrix work + 6.82 ms Softmax to 3.80 ms fused MHA. Quantized linears remain about 30.7 ms and convolution about 47.2 ms. Nsight traces show 2,452→2,102 launches and 100.50→83.35 ms summed kernel duration. Thus this is an actual fusion improvement, not a precision relabel or a change to calibrated linear weights.

Both original and attention-restored **level-3** FP8 builds produce nonfinite output on the Gaussian test and real held-out inputs. Removing Softmax quantization does not resolve that separate failure. The exact first failing fusion/tactic has not been isolated, so it remains an engine/graph integration issue rather than a claimed hardware limitation. Finite-output validation continues to reject these builds; optimization level 3 is not recommended for this FP8 path until resolved.

The SDK also compresses the 51 convolution weight tensors into FP8 D/Q during ONNX postprocessing even though this integration does not calibrate convolution activations. Inspector convolution constants and activations are Half, with `f16f16` tactics: these D/Q nodes are not evidence of FP8 convolution execution. The change preserves this pre-existing SDK behavior.

Holding every non-linear group fixed, removing the approximately 31.4 ms linear contribution from the original 105.6 ms profiled total gives only about **1.42×** conditional Amdahl speedup; convolution alone contributes a 47 ms floor. This explains why optimizing linears alone cannot make this level-0 graph beat BF16. It is not a bound after changing convolution tactics or quantization scope.

Extending FP8 to convolution activations is technically supported by [TensorRT 11.4 convolution types](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/_static/operators/Convolution.html) and [ModelOpt's supported examples](https://github.com/NVIDIA/Model-Optimizer/blob/0.47.0/examples/torch_onnx/torch_quant_to_onnx.py). It would require adapting ComfyUI's Conv2d operations, calibration and accuracy checks. After level-3 tuning, convolution is only about 5 ms, and current quantization already has substantial quality drift; broadening it is not justified here without further accuracy work. The documented convolution types do not include FP4; no unsupported NVFP4 convolution conversion was added.

NVFP4 inspection confirms block-scaled FP4 kernels in the `cutlass3x_sm120_bstensorop...block_scaled_ue4m3xe2m1...stream_k` family, FP8 block scales and dynamic activation quantization. The new matched NVFP4 measurements are 65.42 ms at level 0 and 22.31 ms at level 3 (100 executions, p95 65.60/22.33 ms, CV 0.133%/0.097%, relatively idle windows). Level 3 is approximately 2.93× faster than its matched level-0 baseline, and about 1.16× faster than BF16 level 3 for this input. Profiled convolution drops from 47.27 to 5.22 ms, while quantized linears change only 11.31→10.81 ms. Dynamic quantization-labelled fused work remains about 3.7 ms, attention about 3.86 ms. These fusion groups are inclusive, so the quantization-labelled total is not an isolated measurement of conversion alone.

Nsight records 1,957 launches and 21.94 ms summed kernel duration for level-3 NVFP4, including the expected SM120 block-scaled FP4 GEMM families. Kernel/operation timing shows a mixed workload with material scaling/quantization work, not a storage-bandwidth-only speedup. Without counters it cannot be classified conclusively as compute-, bandwidth-, latency- or occupancy-bound.

Higher precision convolutions and fused MHA remain. Lower storage precision does not imply the entire denoising step runs in FP4, nor that each activation conversion or scale computation is free.

ModelOpt 0.47 with Triton 3.9 fails because its NVFP4 fake-quantization kernel calls removed `tl.make_block_ptr`. An isolated ModelOpt 0.48rc1 experiment on the unchanged PyTorch 2.15/Triton 3.9 stack fails the same way. Explicitly selecting the SDK's CUDA extension using its `DISABLE_TRITON_KERNEL` setting passes a small NVFP4 quantize→ONNX→TensorRT→inference test with Half, FP8 and FP4E2M1 tensors; RMSE is 0.000309. Ninja and a CUDA toolkit must be on PATH. This is a process-local diagnostic alternative, not a global monkeypatch or a reason to replace the user's Triton installation. Stable ModelOpt **0.47 + Triton 3.9 + PyTorch 2.15** then passed full 96-input calibration/export, a level-3 TensorRT build, 100 executions (median **22.259 ms**, p95 22.346 ms, CV 0.186%, relatively idle), six held-out comparisons (relative RMSE 9.323%) and a real two-execution sampling/decode/switcher workflow. This resolves the export compatibility mechanics through NVIDIA's existing CUDA implementation, without downgrading the working stack or substituting another precision. The backend setting remains explicit in the standalone diagnostic process; production nodes do not mutate SDK-global state in a running ComfyUI server. The older isolated Triton 3.6 environment remains an experimental reference only.

## Numerical validation

Six held-out inputs (indices 0, 1, 8, 9, 14, 15) cover positive/negative conditioning at early, middle and late denoising stages of an unseen prompt. The reference is the same native FP32 source evaluated on CPU. These are not training/calibration examples. Aggregate relative RMSE is normalized by the reference RMS; maximum error is over all six outputs.

| Engine | RMSE | Relative RMSE | Maximum absolute error | Finite |
| --- | ---: | ---: | ---: | --- |
| FP16 level 0 | 0.003994 | 0.4516% | 0.05460 | Yes |
| FP16 level 3 | 0.004053 | 0.4583% | 0.06649 | Yes |
| BF16 level 0 | 0.007803 | 0.8823% | 0.12533 | Yes |
| BF16 level 3 | 0.007914 | 0.8948% | 0.14253 | Yes |
| FP8 level 0 | 0.036292 | 4.1037% | 0.69441 | Yes |
| FP8 level 0, restored attention | 0.036454 | 4.1220% | 0.70271 | Yes |
| FP8 level 3, original / attention variant | — | — | — | No; rejected |
| NVFP4 level 0 | 0.081833 | 9.2532% | 1.43318 | Yes |
| NVFP4 level 3 | 0.082081 | 9.2813% | 1.25032 | Yes |

Independent FP16 cached builds and independent restored-attention FP8 builds produce bit-identical outputs on all six inputs. FP16 level 0→3 changes denoiser output by 0.558% relative RMSE; level 3 at 256→1024 MiB changes it by 0.263%, and level 3→5 is bit-identical on these inputs. These are small compared with the measured quantization drift, but tactic tuning is not promised to be bit-exact across configurations.

The level-3 speed changes do not materially change aggregate error within a precision. Quantization error is larger on real held-out inputs than on the Gaussian test: NVFP4 is approximately 9.3% here versus 3.25% on the deterministic Gaussian input. Calibration with six prompts is a stronger diagnostic than the original eight-input smoke dataset, but is insufficient to certify broad image quality. FP8/NVFP4 remain experimental; this report does not call their output equivalent to FP32 or recommend them solely because they are faster/smaller. Full sampler and decoded-image comparisons below assess accumulated denoising effects separately.

## FP32/TF32 and hardware limits

The historical FP32/TF32 run took 303.039 ms with a 9.573 GiB engine. Its TF32 builder policy was left at TensorRT's default. Strong graph types and default TF32 permission do not guarantee every FP32 operator uses TF32 Tensor Cores. FP32 graph tensors and tactic names must be distinguished from actual hardware instruction utilization.

The retained FP32 level-0 inspector has 1,623 layers: 728 GEMM, 51 correlation, 843 generated kernels and one shape layer. It includes 373 `sm86_xmma_gemm_f32f32_tf32f32_f32...128x128x32` GEMMs and 140 `sm80_xmma_gemm_f32f32_tf32f32_f32...128x64x16` variants. Thirty convolutions select indexed TF32 implicit GEMM, while eleven select an FP32 `ffma` family. Thus TF32 permission was effective for many kernels, with mixed execution; it was not silently disabled. These are historical tactics, not new uncontended layer timings.

A fresh full-size build is subject to the harness's weight-artifact + workspace + 2 GiB headroom check; native FP32 requires weights + 4 GiB headroom. The build attempt measured 11.65 GiB free against 9.57 GiB graph artifacts + 0.25 GiB workspace + 2 GiB reserve and was rejected before compilation. Native GPU FP32 similarly failed its 9.56 GiB weights + 4 GiB reserve check. The resident vLLM allocation limits what can be tested safely. No full-size FP32 result is fabricated by substituting a smaller resolution or a different source. The native FP32 CPU output supplies the numerical reference even where a GPU throughput comparison cannot fit.

[NVIDIA's Server Edition specifications](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/?country=international) give equal nominal FP16/BF16 throughput, higher FP8/FP4 ceilings and 1,597 GB/s memory bandwidth. These are ceilings with their documented qualifications, not predictions of SDXL latency. At batch 1, small matrices, convolution, launch costs, scaling and fusion determine practical utilization. Engine size divided by latency is not measured memory bandwidth: caching, rereads, activations and compression invalidate that inference. Occupancy, actual Tensor Core utilization and compute/bandwidth-bound claims require counters and are not inferred merely from kernel names.

## Runtime and workflow ownership

`runtime.py` already retains the engine/context until explicit offload or replacement. It allocates per-call conversion/output buffers, binds them on the caller stream and serializes use with a lock. This avoids persistent output aliasing and stale addresses across calls. Whole-model release clears the context/runtime; the loader and Model Switcher remain responsible for source reconstruction.

The native FP16 export-attention comparison measured median 57.30 ms, p95 76.08 ms and CV 11.01% over 100 calls. It reproduces its saved native reference exactly, but is too variable for a confident throughput ratio; CPU launch scheduling on the shared host remains a limitation. A previous native run was also noisy.

FP16 level-3 raw execution is 25.52 ms and the real wrapper is 25.77 ms. Measured input conversion, output conversion, allocation, shape setup and address setup are individually about 0.007–0.029 ms. The default-stream diagnostic measured 25.516 ms, and the caller's nondefault stream measured 25.532 ms; there is no demonstrated throughput benefit from adding another stream policy. The TensorRT default-stream warning alone is not evidence that it is the dominant cost. No persistent tensor buffer cache or CUDA Graph capture was added for a roughly 1% overhead.

Cold file reading, engine deserialization and context construction are recorded separately from warm execution. Full workflow tests use native CLIP encoding, KSampler, VAE decode and Model Switcher through a real `PromptExecutor`, including a second execution after release. Cold loading and decode can dominate an eight-step workflow even when the engine is fast. Workflow timings must not be compared directly with a single B1 UNet call; CFG and batching execute additional engine calls.

## Cold startup, complete workflows and generation quality

Representative real-input startup measurements separate file read / deserialization / context creation: FP16 L3 **0.619 / 0.286 / 0.648 s**, BF16 L3 **0.664 / 0.271 / 0.755 s**, original FP8 L0 **0.417 / 0.160 / 0.217 s**, NVFP4 L3 **0.203 / 0.115 / 0.340 s**. These are individual startup observations, sensitive to filesystem and process caches, not statistical cold-start guarantees. Observed resident deltas were approximately 5.15, 5.16, 3.59 and 2.13 GiB respectively. Context allocation and persistent engine storage are different from the builder workspace limit.

The following real PromptExecutor workflows use the held-out still-life prompt, seeds 914/915, 1024×1024, eight Euler steps, CFG 3, VAE decoding and Model Switcher. The second execution reuses source/CLIP objects but must load the engine again after release. The warm UNet median is for a CFG call executing two B1 engine invocations; first calls include engine startup and are visible as large p95 values in the raw workflow JSON.

| Engine | First / second complete execution (s) | First / second sampler (s) | Median CFG UNet call (ms) |
| --- | ---: | ---: | ---: |
| FP16 L0 | 11.543 / 3.714 | 2.892 / 2.647 | 200.280 |
| FP16 L3 | 5.489 / 3.328 | 2.067 / 1.773 | 51.030 |
| BF16 L3 | 5.513 / 3.328 | 2.189 / 1.849 | 51.788 |
| FP8 original L0 | 5.515 / 3.293 | 2.559 / 2.211 | 202.165 |
| FP8 restored attention L0 | 13.533 / 3.227 | 9.886 / 1.967 | 168.157 |
| NVFP4 L3 | 4.443 / 2.259 | 1.228 / 1.012 | 44.530 |

Source/CLIP load, encode, decode and reconstruction phases are recorded separately. First-execution CLIP initialization ranged from about 0.5 to 6.2 s; an FP8 restored-attention first sampler incurred a 9.89 s startup. These cold outliers prevent treating a handful of complete workflows as controlled throughput comparisons. The second sampler and warm UNet numbers are more informative, but include intentional reload after switching. No repeated-execution latency guarantee is made for full cold workflows.

Long-prompt FP16 and NVFP4 workflows also passed; the explicit token-recording check confirms **154 context tokens** for the FP16 long case. Both short and long tests decoded finite 1024×1024 images, checked all three source loading dtypes, reconstructed 722 MODEL patches and confirmed the engine context was released after each Model Switcher execution. The stable-ModelOpt CUDA-backend NVFP4 workflow passed with 77 tokens and warm CFG call median 44.42 ms.

Native FP32 CPU sampling uses exactly the saved positive/negative conditioning and initial-noise seeds. It supplies a quality reference, not a GPU throughput comparison. Each seed required approximately four minutes of sampling on eight CPU threads. Decoded pixels are compared in their floating-point [0,1] range; display PNGs are quantized only for viewing.

| Engine | Latent relative RMSE, seed 914 / 915 | Decoded-image RMSE, seed 914 / 915 |
| --- | ---: | ---: |
| FP16 L0 | 7.79% / 8.83% | 0.02329 / 0.02343 |
| FP16 L3 | 8.50% / 8.25% | 0.02210 / 0.02169 |
| BF16 L3 | 12.85% / 11.09% | 0.03009 / 0.02709 |
| FP8 original L0 | 39.94% / 20.04% | 0.07891 / 0.04761 |
| FP8 restored attention L0 | 39.19% / 19.05% | 0.07633 / 0.04694 |
| NVFP4 L3 | 43.90% / 47.38% | 0.09013 / 0.11398 |

The FP8 normalization improves decoded-image RMSE slightly for both seeds, without claiming bitwise equality to the old attention implementation. Quantized outputs have material composition/detail changes compared with FP32; restored attention does not remove the underlying quantization drift. FP16/BF16 also accumulate numerical differences over eight steps. These two seeds of one unseen prompt support functional regression checking, not broad perceptual-quality certification. `generation-comparison.json` includes RMSE, relative RMSE, maximum error, shape and finite status; `comparison-914.png` and `comparison-915.png` preserve side-by-side actual outputs.

## Reproducing the investigation

Run from the ComfyUI root using the Python environment appropriate to the precision. Set `CUDA_VISIBLE_DEVICES` to a usable GPU and pass its **physical nvidia-smi index or UUID** to `--gpu`; the harness checks UUID agreement. On this server the unused GPU was inaccessible, so the runs used `CUDA_VISIBLE_DEVICES=1` and a private, nonexistent MPS pipe under `dev_temp`. Do not change another user's MPS server.

```sh
python custom_nodes/TensorRTNodes/benchmark.py prepare \
  --source dev_temp/source.json --precision fp16 --export-device cuda \
  --output dev_temp/experiment/prepared
python custom_nodes/TensorRTNodes/benchmark.py build \
  --prepared dev_temp/experiment/prepared/prepared.json \
  --output dev_temp/experiment/build-a --gpu 1 --level 3 --workspace 1024
python custom_nodes/TensorRTNodes/benchmark.py run \
  --build dev_temp/experiment/build-a/build.json --gpu 1 \
  --output dev_temp/experiment/run-a.json --wrapper --components \
  --profile-iterations 10
python custom_nodes/TensorRTNodes/benchmark.py build \
  --prepared dev_temp/experiment/prepared/prepared.json \
  --output dev_temp/experiment/build-b --gpu 1 --level 3 --workspace 1024 --timing-cache dev_temp/experiment/build-a/timing.cache
python custom_nodes/TensorRTNodes/benchmark.py compare \
  dev_temp/experiment/run-a.json dev_temp/experiment/run-b.json \
  --baseline dev_temp/experiment/run-a.json --max-regression .03 --max-cv .03 \
  --output dev_temp/experiment/comparison.json
```

Run the second engine with `run` to create `run-b.json`, and repeat each run in fresh processes. Source JSON can be a version-3 manifest or its `source` object. FP8/NVFP4 preparation requires `--calibration path.npz --export-device cuda`; the isolated CUDA-backend experiment additionally uses `--modelopt-cuda`. `evaluate` compares recorded real inputs against native FP32 or another engine; `native` measures the native export-attention contract. These do not silently publish production packages. Every output/cache path must resolve inside `ComfyUI/dev_temp`; tests cover symlink escape attempts. A failed validation retains diagnostic build provenance and inspector information without asserting a pass.

## Engineering feasibility and source changes

1. **Can the FP16/BF16 gap be eliminated?** Yes for this workload: 100.16/33.12 ms becomes 25.52/25.90 ms by choosing level 3. Graph, tactic, operation timings, fresh processes and an independent cached build support this result.
2. **Can FP8 become substantially faster?** Restoring higher precision fused attention gives a validated 1.20× level-0 improvement, reproduced across fresh processes and an independent build, with slightly improved image RMSE in both tested generations. It still loses to optimized FP16/BF16. Level-3 numerical failures remain unresolved and are rejected.
3. **Can NVFP4 beat BF16?** Yes in the matched deterministic workload: level 3 gives 22.31 versus 25.90 ms, approximately 1.16×. This is a modest measured speed benefit, with separate numerical/quality qualifications, rather than a nominal FP4 throughput multiplier. Mixed precision and dynamic quantization impose a workload-specific floor.
4. **Can rebuilds be consistent?** Yes in the tested FP16 cache experiment: identical tactics and <0.1% median spread. This is not a guarantee across drivers, GPUs, TensorRT versions or arbitrary contention.
5. **Can repeated executions avoid large differences?** Warm, matched observed-idle runs are stable. Explicit context reuse already handles the principal startup cost; source switching deliberately releases resources and incurs cold loading again.
6. **What is fundamental?** Batch-1 SDXL contains mixed operators and shapes with different kernel efficiency; precision alone cannot predict end-to-end ranking. Counter restrictions limit stronger hardware bottleneck conclusions. The FP16 level-0 anomaly is removable, not a fundamental BF16 advantage.
7. **What should be production default?** Retain level 3 / 1 GiB for nonquantized and NVFP4 builds, normal TF32 policy, strongly typed precision, finite-output validation and runtime ownership. Preserve Half attention for FP8 Linear exports. FP8 level 3 remains invalid in this test; level 0 is a slower experimental workaround, not an automatic fallback. FP16 remains the practical default given the quantized quality drift. The diagnostic builder defaults to level 3 / 1 GiB.
8. **What remains experimental?** Editable timing-cache pinning, process-level SDK backend overrides, and quantized calibration quality. The measured FP8 attention restoration is retained, but does not certify the broader FP8 path or fix its level-3 failure. Level 5, larger workspace, persistent buffers and CUDA Graphs have no measured justification here.

Code commits:

* `67ac5f8` — Add reproducible TensorRT performance diagnostics: `benchmark.py` and `tests/test_benchmark.py`.
* `66a4857` — Preserve fused attention in FP8 linear exports: the small graph normalization in `quantization.py` and three focused cases in `tests/test_quantization.py`.

`README.md` links this report and distinguishes historical measurements from new results; `PERFORMANCE.md` contains the analysis. No core ComfyUI files, dependencies, node APIs, manifests, naming rules, source reconstruction, package isolation or Model Switcher behavior changed. Benchmark telemetry/profiling is opt-in and absent from normal inference.

Resource cleanup removed approximately **82.98 GiB** of redundant external weights and plans, leaving an approximately 11 GiB experiment workspace. Three validated engine folders remain: `fp16-l3-w256-a`, `fp8-half-attention-l0-w256-a`, and `nvfp4-cuda-l3-w256-a`, including their isolated wrapper packages. Graph metadata, calibration/held-out inputs, verification tensors, raw measurements, inspector reports, timing caches, images and scripts remain. External graph weights and other engine plans require re-export/rebuild before reuse; `cleanup.json` lists removals. Production packages and source checkpoints/LoRAs were not overwritten. Final GPU 1 free memory returned to 12,483 MiB with 0% utilization.

Validation passed **48 regression tests, 13 calibration/quantization tests and 12 benchmark-validity tests**. The two-test CUDA suite passed across FP32/FP16/BF16/FP8/NVFP4, then passed again for FP8 after the normalization. Full SDXL graphs, two independent FP8 builds, cached FP16 rebuilds, six held-out native FP32 comparisons, short/long generations and real PromptExecutor workflows supplied the performance/numerical evidence above. Standalone scripts still emit pre-existing ComfyUI patcher-destructor warnings during interpreter shutdown; results and resource-release assertions pass before shutdown.

Numerical measurements distinguish native FP32, representational rounding, engine/tactic differences and full denoising divergence; finite images alone do not establish quality equivalence.
