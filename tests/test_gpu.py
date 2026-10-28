"""Opt-in synthetic engine tests. These do not measure SDXL image quality.

TENSORRT_TEST_GPU=1 python custom_nodes/TensorRTNodes/tests/test_gpu.py
Set TENSORRT_TEST_PRECISIONS to a comma-separated subset when investigating a backend.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import test_regressions as support
import torch

from trtnodes.runtime import TensorRTUNet


class SmallUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2048, 2048)

    def forward(self, x, timestep, context, y, transformer_options):
        return (x + self.linear(context).mean((1, 2)).reshape(-1, 1, 1, 1)
                + y.mean(1).reshape(-1, 1, 1, 1) + timestep.to(x.dtype).reshape(-1, 1, 1, 1) * 0.001)


@unittest.skipUnless(os.environ.get("TENSORRT_TEST_GPU") == "1", "Set TENSORRT_TEST_GPU=1 to build CUDA engines")
class EngineTests(unittest.TestCase):
    def test_failed_parser_cleans_staging(self):
        support.manifest.TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=support.manifest.TEMP_ROOT) as temporary:
            root = Path(temporary)
            onnx = root / "invalid.onnx"
            onnx.write_bytes(b"invalid ONNX")
            with self.assertRaisesRegex(support.manifest.ManifestError, "parser failed"):
                support.exporter._build_engine(onnx, root / "candidate.engine", root / "failure.txt", 64,
                    {"min": 77, "opt": 77, "max": 154}, (1, 4, 8, 8), 0, "fp16")
            self.assertFalse((root / "candidate.engine").exists())
        self.assertFalse(root.exists())

    def test_precisions_replacement_batching_and_release(self):
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        torch.manual_seed(812)
        support.manifest.TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=support.manifest.TEMP_ROOT) as temporary:
            root = Path(temporary)
            state = SmallUNet().state_dict()
            x = torch.randn(2, 4, 8, 8, device="cuda")
            timestep = torch.tensor([500., 700.], device="cuda")
            context = torch.randn(2, 100, 2048, device="cuda")
            adm = torch.randn(2, 2816, device="cuda")
            calibration = {"latent": x[:1].cpu().numpy()[None], "timestep": timestep[:1].cpu().numpy()[None],
                           "context": context[:1, :77].cpu().numpy()[None], "adm": adm[:1].cpu().numpy()[None]}
            # Deterministic synthetic calibration tests Q/DQ mechanics, not model quality.
            calibration = {key: np.repeat(value, 2, axis=0) for key, value in calibration.items()}
            runtime = None
            try:
                for index, precision in enumerate(os.environ.get("TENSORRT_TEST_PRECISIONS", "fp32,fp16,bf16,fp8,nvfp4").split(",")):
                    with self.subTest(precision=precision):
                        model = SmallUNet().to(device="cuda", dtype=support.exporter.DTYPES[precision])
                        model.load_state_dict(copy.deepcopy(state))
                        candidate = root / "candidate.engine"
                        onnx = support.exporter._export_onnx(model, root / f"{precision}.onnx", 64, 64, 77, precision,
                                                             calibration if precision in {"fp8", "nvfp4"} else None,
                                                             root / "verification.pt")
                        maximum = 154 + index
                        profile = {"min": 77, "opt": 77 + index, "max": maximum}
                        report = support.exporter._build_engine(onnx, candidate, root / "build.json", 64,
                            profile, (1, 4, 8, 8), 0, precision, root / "verification.pt")
                        support.manifest.publish_engine(root / "package", support.CHAIN, candidate, (64, 64),
                                                        precision, report["requirements"], root)
                        (root / "previous.engine").unlink(missing_ok=True)
                        (root / "previous.json").unlink(missing_ok=True)
                        if runtime is None:
                            runtime = TensorRTUNet(support.manifest.load_manifest(root / "package"))
                        result = runtime(x, timestep, context=context, y=adm)
                        self.assertEqual(runtime._loaded.profile, (77, 77 + index, maximum))
                        dtype = support.exporter.DTYPES[precision]
                        expected = model(x.to(dtype), timestep, context.to(dtype), adm.to(dtype), {}).float()
                        self.assertEqual(result.shape, x.shape)
                        self.assertEqual(result.dtype, x.dtype)
                        torch.testing.assert_close(result, expected, rtol=0.04, atol=0.03)
                        for tokens in (77, maximum):
                            runtime(x[:1], timestep[:1], context=context[:1, :1].expand(1, tokens, 2048), y=adm[:1])
                        for tokens in (76, maximum + 1):
                            with self.assertRaisesRegex(RuntimeError, f"supports 77-{maximum}"):
                                runtime(x[:1], timestep[:1], context=context[:1, :1].expand(1, tokens, 2048), y=adm[:1])
                        for _ in range(3):
                            runtime.release()
                            self.assertEqual(runtime.resident_memory(), 0)
                            self.assertIsNone(runtime._runtime)
                            runtime(x[:1], timestep[:1], context=context[:1], y=adm[:1])
                model = SmallUNet().cuda().half()
                model.load_state_dict(state)
                onnx = support.exporter._export_onnx(model, root / "second.onnx", 128, 64, 77, "fp16")
                report = support.exporter._build_engine(onnx, candidate, root / "second.json", 64,
                    {"min": 77, "opt": 77, "max": 154}, (1, 4, 8, 16), 0, "fp16")
                support.manifest.publish_engine(root / "package", support.CHAIN, candidate, (128, 64),
                                                "fp16", report["requirements"], root)
                self.assertEqual(len(support.manifest.load_manifest(root / "package").engines), 2)
                wide = torch.randn(1, 4, 8, 16, device="cuda")
                self.assertEqual(runtime(wide, timestep[:1], context=context[:1], y=adm[:1]).shape, wide.shape)
                self.assertEqual(runtime(x[:1], timestep[:1], context=context[:1], y=adm[:1]).shape, x[:1].shape)
                runtime.release()
                data = support.manifest.read_manifest(root / "package")
                next(item for item in data["engines"] if item["image_size"] == [128, 64])["requirements"]["gpu_name"] = "Other GPU"
                (root / "package" / "manifest.json").write_text(json.dumps(data))
                runtime = TensorRTUNet(support.manifest.load_manifest(root / "package"))
                runtime(x[:1], timestep[:1], context=context[:1], y=adm[:1])
                with self.assertRaisesRegex(support.manifest.ManifestError, "Other GPU"):
                    runtime(wide, timestep[:1], context=context[:1], y=adm[:1])
            finally:
                if runtime is not None:
                    runtime.release()


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
