"""Calibration boundary tests; GPU precision integration is tested separately."""
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import onnx

package = types.ModuleType("trtnodes")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules["trtnodes"] = package
quantization = importlib.import_module("trtnodes.quantization")
manifest = importlib.import_module("trtnodes.manifest")


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        manifest.TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=manifest.TEMP_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "calibration.npz"
        self.data = {"latent": np.zeros((2, 1, 4, 8, 8), dtype=np.float16),
                     "timestep": np.ones((2, 1), dtype=np.float32),
                     "context": np.zeros((2, 1, 77, 2048), dtype=np.float16),
                     "adm": np.zeros((2, 1, 2816), dtype=np.float16)}

    def load(self):
        np.savez(self.path, **self.data)
        return quantization.load_calibration(str(self.path), 64, 64)

    def test_round_trip(self):
        loaded = self.load()
        for key in self.data:
            np.testing.assert_array_equal(loaded[key], self.data[key])

    def test_missing_calibration(self):
        with self.assertRaisesRegex(manifest.ManifestError, "representative"):
            quantization.load_calibration("", 64, 64)

    def test_outside_development_directory(self):
        with self.assertRaisesRegex(manifest.ManifestError, "under ComfyUI/dev_temp"):
            quantization.load_calibration("/tmp/calibration.npz", 64, 64)

    def test_missing_inputs(self):
        del self.data["adm"]
        with self.assertRaisesRegex(manifest.ManifestError, "adm"):
            self.load()

    def test_nonfinite_inputs(self):
        self.data["context"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(manifest.ManifestError, "context"):
            self.load()

    def test_bad_shapes(self):
        for key, shape in (("latent", ()), ("latent", (2, 1, 4, 16, 8)), ("context", (2, 1, 0, 2048)),
                           ("timestep", (1, 1)), ("adm", (2, 2816))):
            original = self.data[key]
            self.data[key] = np.zeros(shape, dtype=np.float32)
            with self.subTest(key=key, shape=shape), self.assertRaises(manifest.ManifestError):
                self.load()
            self.data[key] = original


class QuantizedGraphTests(unittest.TestCase):
    def graph(self, precision, activations=True):
        dtype = onnx.TensorProto.FLOAT8E4M3FN if precision == "fp8" else onnx.TensorProto.FLOAT4E2M1
        weight = onnx.TensorProto(name="weight", data_type=dtype, dims=[16, 16], raw_data=bytes(256))
        nodes = [onnx.helper.make_node("DequantizeLinear", ["weight", "scale"], ["w"])]
        if activations:
            op = "QuantizeLinear" if precision == "fp8" else "TRT_FP4DynamicQuantize"
            nodes += [onnx.helper.make_node(op, ["input", "scale"], ["q"]),
                      onnx.helper.make_node("DequantizeLinear", ["q", "scale"], ["a"])]
        nodes += [onnx.helper.make_node("MatMul", ["a" if activations else "input", "w"], ["output"])]
        return onnx.helper.make_graph(nodes, "quantized", [], [], [weight])

    def test_weight_only_graphs_rejected(self):
        for precision in ("fp8", "nvfp4"):
            with self.subTest(precision=precision), self.assertRaisesRegex(manifest.ManifestError, "weight-and-activation"):
                quantization.validate_quantized_graph(self.graph(precision, False), precision)

    def test_paired_matrix_quantization(self):
        for precision in ("fp8", "nvfp4"):
            quantization.validate_quantized_graph(self.graph(precision), precision)

    def test_unrelated_target_weights_rejected(self):
        for precision in ("fp8", "nvfp4"):
            graph = self.graph(precision)
            graph.initializer[0].name = "unused_target_weight"
            with self.subTest(precision=precision), self.assertRaisesRegex(manifest.ManifestError, "weight-and-activation"):
                quantization.validate_quantized_graph(graph, precision)

    def test_unrelated_activation_quantization_rejected(self):
        for precision in ("fp8", "nvfp4"):
            graph = self.graph(precision)
            next(node for node in graph.node if node.output[0] == "a").input[0] = "unquantized_input"
            with self.subTest(precision=precision), self.assertRaisesRegex(manifest.ManifestError, "weight-and-activation"):
                quantization.validate_quantized_graph(graph, precision)


if __name__ == "__main__":
    unittest.main(verbosity=2)
