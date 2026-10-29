"""CPU checks for benchmark accounting and artifact containment."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        benchmark.DEV_ROOT.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=benchmark.DEV_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_precision_choices_match_supported_engines(self):
        self.assertEqual(benchmark.PRECISIONS, ("fp32", "fp16", "bf16"))
        arguments = benchmark.parser().parse_args(["prepare", "--source", "source.json", "--output", "prepared", "--precision", "bf16"])
        self.assertEqual(arguments.precision, "bf16")

    def test_build_rejects_unsupported_precision_before_loading_cuda(self):
        prepared = self.root / "prepared.json"
        prepared.write_text(json.dumps({"precision": "int8"}))
        with self.assertRaisesRegex(ValueError, "Unsupported engine precision"):
            benchmark.build(types.SimpleNamespace(prepared=str(prepared)))

    def test_statistics_include_tail_and_variability(self):
        result = benchmark.distribution(list(range(1, 101)))
        self.assertEqual(result["count"], 100)
        self.assertEqual(result["median"], 50.5)
        self.assertEqual(result["p95"], 95)
        self.assertEqual(result["min"], 1)
        self.assertEqual(result["max"], 100)
        self.assertGreater(result["cv"], .5)
        self.assertEqual(benchmark.distribution([2, 2, 2])["stddev"], 0)
        self.assertEqual(benchmark.distribution([]), {"count": 0})

    def test_background_activity_classification(self):
        for utilization, expected in [(0, "relatively_idle"), (5, "relatively_idle"),
                                      (6, "moderately_contended"), (30, "moderately_contended"),
                                      (31, "heavily_contended")]:
            self.assertEqual(benchmark.contention([{"utilization.gpu": utilization}]), expected)
        self.assertEqual(benchmark.contention([{"utilization.gpu": "N/A"}]), "unknown")

    def test_input_signature_distinguishes_values_and_shapes(self):
        values = {"latent": torch.tensor([[1., 2.]])}
        first = benchmark.input_signature(values)
        self.assertEqual(first, benchmark.input_signature({"latent": values["latent"].clone()}))
        self.assertNotEqual(first, benchmark.input_signature({"latent": torch.tensor([[1., 3.]])}))
        self.assertNotEqual(first, benchmark.input_signature({"latent": values["latent"].reshape(2, 1)}))

    def test_accuracy_does_not_broadcast_or_accept_nonfinite_outputs(self):
        with self.assertRaisesRegex(ValueError, "shapes"):
            benchmark.tensor_error(torch.ones(1, 2), torch.ones(2))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            benchmark.tensor_error(torch.tensor([float("nan")]), torch.ones(1))

    def test_fused_compute_is_not_counted_as_generic_pointwise(self):
        for metadata, expected in [("[ONNX Layer: /time_embed/Gemm]", "linear"),
                                   ("[ONNX Layer: /block/Conv]", "convolution"),
                                   ("[ONNX Layer: /ff/MatMul]", "linear")]:
            self.assertEqual(benchmark.operation_group({"LayerType": "fusion", "Metadata": metadata}), expected)

    def test_artifacts_cannot_escape_through_symlinks(self):
        self.assertEqual(benchmark.artifact_path(self.root / "result.json"), self.root / "result.json")
        with self.assertRaises(ValueError):
            benchmark.artifact_path(Path(__file__))
        link = self.root / "outside"
        link.symlink_to(Path(__file__).parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            benchmark.artifact_path(link / "result.json")

    def test_external_graph_storage_is_counted_once(self):
        weights = self.root / "weights.bin"
        weights.write_bytes(bytes(16))
        tensors = []
        for index in range(2):
            tensor = onnx.TensorProto(name=f"w{index}", data_type=onnx.TensorProto.FLOAT,
                                      dims=[2], data_location=onnx.TensorProto.EXTERNAL)
            tensor.external_data.add(key="location", value="weights.bin")
            tensor.external_data.add(key="offset", value=str(index * 8))
            tensor.external_data.add(key="length", value="8")
            tensors.append(tensor)
        graph = onnx.helper.make_graph([], "external", [], [], tensors)
        model = onnx.helper.make_model(graph)
        path = self.root / "model.onnx"
        path.write_bytes(model.SerializeToString())
        self.assertEqual(benchmark.graph_file_bytes(path), path.stat().st_size + 16)

    def result(self, build, gpu="gpu-one"):
        return {"config": {"gpu_uuid": gpu, "driver": "driver", "tensorrt": "11", "cuda": "13",
                           "source": {"base_model": "exact.safetensors"}, "width": 1024, "height": 1024,
                           "context": [77, 77, 154], "precision": "fp16"},
                "stream": "default", "build": build, "contention": "relatively_idle", "environment": {"torch": "version"},
                "input_signature": "identical-test-inputs",
                "raw": {"cuda_samples_ms": [10, 11, 12], "cuda_ms": {"median": 11}}}

    def test_process_count_is_separate_from_independent_build_count(self):
        files = []
        for index, build in enumerate(("build-a", "build-a", "build-b")):
            path = self.root / f"run{index}.json"
            path.write_text(json.dumps(self.result(build)))
            files.append(str(path))
        output = self.root / "comparison.json"
        benchmark.compare(types.SimpleNamespace(results=files, output=str(output), baseline=None))
        row = json.loads(output.read_text())["results"][0]
        self.assertEqual(row["processes"], 3)
        self.assertEqual(row["independent_builds"], 2)
        self.assertEqual(row["cuda_ms"]["count"], 9)

    def test_mismatched_gpu_comparisons_rejected(self):
        files = []
        for gpu in ("gpu-one", "gpu-two"):
            path = self.root / f"{gpu}.json"
            path.write_text(json.dumps(self.result("build", gpu)))
            files.append(str(path))
        with self.assertRaisesRegex(ValueError, "environment"):
            benchmark.compare(types.SimpleNamespace(results=files, output=str(self.root / "bad.json")))

    def test_different_input_values_cannot_be_pooled(self):
        files = []
        for index in range(2):
            result = self.result("build")
            result["input_signature"] = f"input-values-{index}"
            path = self.root / f"input{index}.json"
            path.write_text(json.dumps(result))
            files.append(str(path))
        with self.assertRaisesRegex(ValueError, "inputs"):
            benchmark.compare(types.SimpleNamespace(results=files, output=str(self.root / "bad.json")))

    def test_contended_runs_cannot_pass_regression_checks(self):
        baseline = self.result("baseline")
        baseline["raw"]["cuda_ms"]["cv"] = .01
        candidate = self.result("candidate")
        candidate["raw"]["cuda_samples_ms"] = [11, 11, 11]
        candidate["contention"] = "heavily_contended"
        base_path, candidate_path = self.root / "base.json", self.root / "candidate.json"
        base_path.write_text(json.dumps(baseline))
        candidate_path.write_text(json.dumps(candidate))
        output = self.root / "check.json"
        benchmark.compare(types.SimpleNamespace(results=[str(candidate_path)], output=str(output),
                                                baseline=str(base_path), max_regression=.05, max_cv=.03))
        check = json.loads(output.read_text())["regression_checks"][0]
        self.assertFalse(check["valid_for_regression_check"])
        self.assertIsNone(check["regression"])

    def test_stable_regression_is_detected(self):
        baseline, candidate = self.result("base"), self.result("candidate")
        baseline["raw"]["cuda_ms"] = {"median": 10, "cv": 0}
        candidate["raw"]["cuda_samples_ms"] = [12, 12, 12]
        base_path, candidate_path = self.root / "base.json", self.root / "candidate.json"
        base_path.write_text(json.dumps(baseline))
        candidate_path.write_text(json.dumps(candidate))
        output = self.root / "check.json"
        benchmark.compare(types.SimpleNamespace(results=[str(candidate_path)], output=str(output),
                                                baseline=str(base_path), max_regression=.05, max_cv=.03))
        check = json.loads(output.read_text())["regression_checks"][0]
        self.assertTrue(check["valid_for_regression_check"])
        self.assertTrue(check["regression"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
