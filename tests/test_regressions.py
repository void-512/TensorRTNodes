"""Run with ComfyUI's Python: python custom_nodes/TensorRTNodes/tests/test_regressions.py."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
package = types.ModuleType("trtnodes")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules["trtnodes"] = package
manifest = importlib.import_module("trtnodes.manifest")
provenance = importlib.import_module("trtnodes.provenance")
loader = importlib.import_module("trtnodes.loader")
exporter = importlib.import_module("trtnodes.exporter")
import nodes
import torch
from comfy_execution.graph import DynamicPrompt, TopologicalSort
from comfy_execution.validation import validate_node_input

CHAIN = {"base_model": "base.safetensors", "loras": []}
REQ = {"tensorrt_version": "11.4.0.106", "gpu_name": "Test GPU", "compute_capability": [12, 0]}


def lora(name="a.safetensors", model=0.8, clip=0.6):
    return {"name": name, "strength_model": model, "strength_clip": clip}


def graph(items):
    prompt = {"base": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHAIN["base_model"]}}}
    model, clip = ["base", 0], ["base", 1]
    for i, item in enumerate(items):
        key = str(i)
        prompt[key] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip,
            "lora_name": item["name"], "strength_model": item["strength_model"], "strength_clip": item["strength_clip"]}}
        model, clip = [key, 0], [key, 1]
    prompt["export"] = {"class_type": "ExportTensorRTPackage", "inputs": {"model": model, "clip": clip, "vae": ["base", 2]}}
    return prompt


class SourceTests(unittest.TestCase):
    def recover(self, items):
        with patch.object(provenance, "resolve_source", return_value="source"):
            return provenance.recover_chain(DynamicPrompt(graph(items)), "export")

    def test_standard_chains(self):
        for items in ([], [lora()], [lora(), lora("b.safetensors", -0.5, 1.4)], [lora(), lora(), lora(model=0, clip=0)]):
            with self.subTest(items=items):
                self.assertEqual(self.recover(items), {**CHAIN, "loras": items})

    def test_chain_identity(self):
        a, b = lora(), lora("b.safetensors", -1, 0)
        chains = [CHAIN, {**CHAIN, "base_model": "nested/base.safetensors"},
                  {**CHAIN, "loras": [a]}, {**CHAIN, "loras": [b]},
                  {**CHAIN, "loras": [a, b]}, {**CHAIN, "loras": [b, a]},
                  {**CHAIN, "loras": [lora(model=0.9)]}, {**CHAIN, "loras": [lora(clip=0.9)]}]
        self.assertEqual(len({json.dumps(c, sort_keys=True) for c in chains}), len(chains))
        self.assertEqual(manifest.package_name(CHAIN), "base")
        self.assertEqual(manifest.package_name(CHAIN), manifest.package_name(copy.deepcopy(CHAIN)))
        self.assertEqual(manifest.package_name({**CHAIN, "loras": [lora(model=1)]}), manifest.package_name({**CHAIN, "loras": [lora(model=1.0)]}))

    def test_unknown_graph(self):
        prompt = graph([])
        prompt["base"]["class_type"] = "ModelMergeSimple"
        with self.assertRaisesRegex(manifest.ManifestError, "Cannot reconstruct MODEL"):
            provenance.recover_chain(DynamicPrompt(prompt), "export")

    def test_model_branch_without_clip_lora(self):
        prompt = graph([lora()])
        prompt["export"]["inputs"]["clip"] = ["base", 1]
        with patch.object(provenance, "resolve_source"):
            self.assertEqual(provenance.recover_chain(DynamicPrompt(prompt), "export")["loras"], [lora(clip=0)])

    def test_independent_intermediate_clip(self):
        prompt = graph([lora(), lora("b.safetensors")])
        prompt["1"]["inputs"]["clip"] = ["base", 1]
        with patch.object(provenance, "resolve_source"):
            self.assertEqual(provenance.recover_chain(DynamicPrompt(prompt), "export")["loras"], [lora(clip=0), lora("b.safetensors")])

    def test_clip_only_branch(self):
        prompt = graph([lora(model=2, clip=-0.4)])
        prompt["export"]["inputs"]["model"] = ["base", 0]
        with patch.object(provenance, "resolve_source"):
            self.assertEqual(provenance.recover_chain(DynamicPrompt(prompt), "export")["loras"], [lora(model=0, clip=-0.4)])

    def test_model_only_lora(self):
        prompt = graph([])
        prompt["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["base", 0], "lora_name": "a.safetensors", "strength_model": -1.0}}
        prompt["export"]["inputs"]["model"] = ["lora", 0]
        with patch.object(provenance, "resolve_source"):
            self.assertEqual(provenance.recover_chain(DynamicPrompt(prompt), "export")["loras"], [lora(model=-1, clip=0)])

    def test_linked_setting(self):
        prompt = graph([lora()])
        prompt["0"]["inputs"]["strength_model"] = ["value", 0]
        with self.assertRaisesRegex(manifest.ManifestError, "linked or missing"):
            provenance.recover_chain(DynamicPrompt(prompt), "export")

    def test_missing_sources(self):
        for kind in ("checkpoints", "loras"):
            with patch.object(provenance.folder_paths, "get_full_path", return_value=None):
                with self.assertRaisesRegex(manifest.ManifestError, "missing.safetensors"):
                    provenance.resolve_source(kind, "missing.safetensors")

    def test_clip_only_lora(self):
        items = [lora(model=0, clip=-0.4), lora("b.safetensors", model=1.2, clip=0)]
        self.assertEqual(self.recover(items)["loras"], items)

    def test_invalid_strengths(self):
        for value in (float("nan"), float("inf"), True, "1.0"):
            with self.subTest(value=value), self.assertRaises(manifest.ManifestError):
                manifest.validate_chain({**CHAIN, "loras": [lora(model=value)]})

    def test_unsafe_paths(self):
        for name in ("../bad", "/bad", "dir/../../bad", "C:\\bad", "dir\\bad", "a//bad", "bad\0name"):
            with self.subTest(name=name), self.assertRaises(manifest.ManifestError):
                manifest.source_name(name)

    def test_additional_model_directories(self):
        manifest.TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=manifest.TEMP_ROOT) as temporary:
            root = Path(temporary)
            primary, extra = root / "primary", root / "extra"
            primary.mkdir()
            (extra / "nested").mkdir(parents=True)
            source = extra / "nested" / "exact.safetensors"
            source.write_bytes(b"source")
            for kind in ("checkpoints", "loras"):
                with patch.dict(provenance.folder_paths.folder_names_and_paths,
                                {kind: ([str(primary), str(extra)], {".safetensors"})}):
                    self.assertEqual(Path(provenance.resolve_source(kind, "nested/exact.safetensors")), source)
                    with self.assertRaisesRegex(manifest.ManifestError, "Missing"):
                        provenance.resolve_source(kind, "exact.safetensors")

    def test_reconstruct_order_and_strengths(self):
        items = [lora(model=-1, clip=0.3), lora(model=0, clip=0), lora(model=2, clip=-0.7)]
        with patch.object(provenance, "resolve_source", side_effect=lambda kind, name: name), \
             patch.object(provenance.comfy.sd, "load_checkpoint_guess_config", return_value=("model", "clip", "vae", None)), \
             patch.object(provenance.comfy.utils, "load_torch_file", return_value=({"weights": 1}, {"meta": 1})), \
             patch.object(provenance.comfy.sd, "load_lora_for_models", side_effect=lambda m, c, *args, **kwargs: (m + "L", c + "L")) as apply:
            self.assertEqual(provenance.reconstruct({**CHAIN, "loras": items}), ("modelLL", "clipLL", "vae"))
            self.assertEqual([(call.args[3], call.args[4]) for call in apply.call_args_list], [(-1, 0.3), (2, -0.7)])

    def test_model_only_reconstruction_skips_clip_only_weights(self):
        with patch.object(provenance, "resolve_source", side_effect=lambda kind, name: name), \
             patch.object(provenance.comfy.sd, "load_checkpoint_guess_config", return_value=("model", None, None, None)), \
             patch.object(provenance.comfy.utils, "load_torch_file") as load:
            result = provenance.reconstruct({**CHAIN, "loras": [lora(model=0, clip=0.5)]}, output_clip=False, output_vae=False)
            self.assertEqual(result, ("model", None, None))
            load.assert_not_called()


class StorageTests(unittest.TestCase):
    def setUp(self):
        manifest.TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=manifest.TEMP_ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "package"
        self.staging = Path(self.tmp.name) / "staging"
        self.staging.mkdir()

    def publish(self, size=(1024, 1024), precision="fp16", value=b"engine", chain=CHAIN):
        candidate = self.staging / "candidate.engine"
        candidate.write_bytes(value)
        manifest.publish_engine(self.root, chain, candidate, size, precision, REQ, self.staging)
        # Backups are per-build scratch files.
        for name in ("previous.engine", "previous.json"):
            (self.staging / name).unlink(missing_ok=True)

    def test_first_multiple_and_replace(self):
        self.publish()
        self.publish((1024, 1536))
        self.publish(precision="bf16", value=b"new")
        loaded = manifest.load_manifest(self.root)
        self.assertEqual(len(loaded.engines), 2)
        spec = next(e for e in loaded.engines if e.image_size == (1024, 1024))
        self.assertEqual(spec.path.name, "base_1024x1024.engine")
        self.assertEqual(spec.precision, "bf16")
        self.assertEqual(spec.path.read_bytes(), b"new")
        self.assertFalse((self.root / "merged_checkpoint.safetensors").exists())
        self.assertEqual(set(p.name for p in self.root.iterdir()), {"manifest.json", "engines"})

    def test_rollback_manifest_failure(self):
        self.publish(value=b"old")
        old_manifest = (self.root / "manifest.json").read_bytes()
        replace = os.replace
        def fail_manifest(src, dst):
            if Path(src).name == "manifest.json":
                raise OSError("injected failure")
            return replace(src, dst)
        with patch("os.replace", side_effect=fail_manifest), self.assertRaisesRegex(OSError, "injected failure"):
            self.publish(precision="fp32", value=b"broken")
        self.assertEqual((self.root / "manifest.json").read_bytes(), old_manifest)
        self.assertEqual(manifest.load_manifest(self.root).engines[0].path.read_bytes(), b"old")

    def test_failed_first_publish(self):
        with patch("os.replace", side_effect=OSError("failure")), self.assertRaises(OSError):
            self.publish()
        self.assertFalse(self.root.exists())

    def test_rollback_after_manifest_replacement(self):
        self.publish(value=b"old")
        old_manifest = (self.root / "manifest.json").read_bytes()
        replace = os.replace
        def fail_after_manifest(src, dst):
            replace(src, dst)
            if Path(src).name == "manifest.json":
                raise OSError("failure after rename")
        with patch("os.replace", side_effect=fail_after_manifest), self.assertRaisesRegex(OSError, "after rename"):
            self.publish(precision="fp32", value=b"broken")
        self.assertEqual((self.root / "manifest.json").read_bytes(), old_manifest)
        self.assertEqual(manifest.load_manifest(self.root).engines[0].path.read_bytes(), b"old")

    def test_rollback_after_engine_replacement(self):
        self.publish(value=b"old")
        replace = os.replace
        def fail_after_engine(src, dst):
            replace(src, dst)
            if Path(src).name == "candidate.engine":
                raise OSError("failure after engine rename")
        with patch("os.replace", side_effect=fail_after_engine), self.assertRaisesRegex(OSError, "after engine"):
            self.publish(precision="bf16", value=b"broken")
        loaded = manifest.load_manifest(self.root)
        self.assertEqual(loaded.engines[0].precision, "fp16")
        self.assertEqual(loaded.engines[0].path.read_bytes(), b"old")

    def test_chain_collision(self):
        self.publish()
        with self.assertRaisesRegex(manifest.ManifestError, "collision"):
            self.publish(chain={**CHAIN, "loras": [lora()]})
        self.assertEqual(len(manifest.load_manifest(self.root).engines), 1)

    def test_concurrent_writer(self):
        with manifest.package_lock(self.root):
            with self.assertRaisesRegex(manifest.ManifestError, "busy"):
                self.publish()

    def test_duplicate_resolution(self):
        self.publish()
        data = manifest.read_manifest(self.root)
        data["engines"].append(data["engines"][0])
        (self.root / "manifest.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(manifest.ManifestError, "duplicate"):
            manifest.load_manifest(self.root)

    def test_path_escape(self):
        self.publish()
        data = manifest.read_manifest(self.root)
        data["engines"][0]["path"] = "../outside.engine"
        (self.root / "manifest.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(manifest.ManifestError, "unsafe"):
            manifest.load_manifest(self.root)

    def test_old_manifest_rejected(self):
        self.root.mkdir()
        (self.root / "manifest.json").write_text('{"format_version":1}')
        with self.assertRaisesRegex(manifest.ManifestError, "rebuild"):
            manifest.load_manifest(self.root)

    def test_invalid_manifest_structures(self):
        self.root.mkdir()
        for data in ([], None, {"format_version": 2, "source": CHAIN, "engines": None},
                     {"format_version": 2, "source": CHAIN, "engines": [None]}):
            with self.subTest(data=data):
                (self.root / "manifest.json").write_text(json.dumps(data))
                with self.assertRaises(manifest.ManifestError):
                    manifest.load_manifest(self.root)

    def test_missing_engine(self):
        self.publish()
        manifest.load_manifest(self.root).engines[0].path.unlink()
        with self.assertRaisesRegex(manifest.ManifestError, "Missing"):
            manifest.load_manifest(self.root)

    def test_engine_symlink_escape(self):
        self.publish()
        engine = manifest.load_manifest(self.root).engines[0].path
        outside = self.staging / "outside.engine"
        outside.write_bytes(b"outside")
        engine.unlink()
        engine.symlink_to(outside)
        with self.assertRaisesRegex(manifest.ManifestError, "unsafe"):
            manifest.load_manifest(self.root)

    def test_manifest_has_no_intermediate_artifacts(self):
        self.publish()
        data = manifest.read_manifest(self.root)
        self.assertEqual(set(data), {"format_version", "source", "engines"})
        self.assertEqual(set(data["engines"][0]), {"path", "image_size", "precision", "requirements"})
        for obsolete in ("sha256", "fingerprint", "merged_checkpoint", ".onnx", "size_bytes"):
            self.assertNotIn(obsolete, json.dumps(data))

    def test_loader_cache_changes_on_rebuild(self):
        self.publish()
        with patch.object(loader, "_resolve_package", return_value=self.root):
            before = loader.LoadTensorRTEngine.IS_CHANGED("package")
            self.publish(value=b"replaced", precision="fp32")
            self.assertNotEqual(before, loader.LoadTensorRTEngine.IS_CHANGED("package"))

    def test_failed_export_or_build_preserves_package_and_cleans_artifacts(self):
        self.publish(value=b"working")
        previous = (self.root / "manifest.json").read_bytes()
        scratch = self.staging / "builds"
        checkpoint = self.staging / "base.safetensors"
        patcher = Mock(cached_patcher_init=(None, (str(checkpoint),)))
        exported = Mock()

        for failure in ("export", "build"):
            def export_onnx(unet, output, *args):
                output.write_bytes(b"temporary graph")
                output.with_suffix(".data").write_bytes(b"temporary weights")
                if failure == "export":
                    raise RuntimeError("injected export failure")
                return output

            with self.subTest(failure=failure), \
                 patch.object(exporter, "TEMP_ROOT", scratch), \
                 patch.object(exporter, "_PACKAGE_ROOT", self.root.parent), \
                 patch.object(exporter, "validate_precision"), \
                 patch.object(exporter, "recover_chain", return_value=CHAIN), \
                 patch.object(exporter, "resolve_source", return_value=str(checkpoint)), \
                 patch.object(exporter, "materialize_unet", return_value=exported), \
                 patch.object(exporter, "_export_onnx", side_effect=export_onnx), \
                 patch.object(exporter, "_build_engine", side_effect=RuntimeError("injected build failure")), \
                 patch.object(exporter.comfy.model_management, "load_models_gpu"), \
                 patch.object(exporter.comfy.model_management, "unload_model_and_clones"), \
                 patch.object(torch.cuda, "empty_cache"), \
                 self.assertRaisesRegex(manifest.ManifestError, f"injected {failure} failure"):
                exporter.ExportTensorRTPackage().convert(
                    patcher, types.SimpleNamespace(patcher=patcher), types.SimpleNamespace(patcher=patcher),
                    1024, 1024, 154, 77, 256, 0, package_name=self.root.name,
                )
            self.assertEqual((self.root / "manifest.json").read_bytes(), previous)
            self.assertEqual(manifest.load_manifest(self.root).engines[0].path.read_bytes(), b"working")
            self.assertTrue(all(path.suffix == ".txt" for path in scratch.iterdir()))


class SwitcherTests(unittest.TestCase):
    def test_any_type_validation(self):
        for kind in ("IMAGE", "LATENT", "STRING", "INT", "FLOAT", "CUSTOM"):
            self.assertTrue(validate_node_input(kind, loader.ModelSwitcher.INPUT_TYPES()["required"]["trigger"][0]))
        self.assertEqual(tuple(loader.ModelSwitcher.RETURN_TYPES), ("MODEL",))

    def test_scheduler_dependency(self):
        class Source:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}
        class Sort(TopologicalSort):
            def is_cached(self, unique_id):
                return False
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {"ModelSwitcher": loader.ModelSwitcher, "Source": Source}):
            prompt = DynamicPrompt({"switch": {"class_type": "ModelSwitcher", "inputs": {"trigger": ["trigger", 0], "engine_model": ["model", 0]}},
                                    "trigger": {"class_type": "Source", "inputs": {}}, "model": {"class_type": "Source", "inputs": {}}})
            sort = Sort(prompt)
            sort.add_node("switch")
            self.assertEqual(sort.blockCount["switch"], 2)
            self.assertIn("switch", sort.blocking["trigger"])
            self.assertNotIn("lazy", loader.ModelSwitcher.INPUT_TYPES()["required"]["trigger"][1])

    def test_release_before_reconstruct_and_repeated(self):
        unet = Mock(spec=loader.TensorRTUNet)
        unet.package = types.SimpleNamespace(chain=CHAIN)
        engine_model = Mock()
        engine_model.get_model_object.return_value = unet
        events = []
        unet.release.side_effect = lambda: events.append("release")
        def rebuild(*args, **kwargs):
            events.append("rebuild")
            return "normal", None, None
        with patch.object(loader.comfy.model_management, "unload_model_and_clones", side_effect=lambda m: events.append("unload")), patch.object(loader, "reconstruct", side_effect=rebuild):
            for trigger in (torch.empty(1, 2, 2, 3), {"samples": torch.empty(1)}, "ready", 1, 1.5):
                result = loader.ModelSwitcher.execute(trigger, engine_model)
                self.assertEqual(result.result, ("normal",))
                self.assertEqual(events[-3:], ["release", "unload", "rebuild"])

    def test_wrong_model(self):
        model = Mock()
        model.get_model_object.return_value = object()
        with self.assertRaisesRegex(manifest.ManifestError, "Connect the MODEL"):
            loader.ModelSwitcher.execute("ready", model)


class PrecisionTests(unittest.TestCase):
    def test_choices(self):
        self.assertEqual(exporter.ExportTensorRTPackage.INPUT_TYPES()["optional"]["precision"][0], list(manifest.PRECISIONS))
        self.assertEqual(exporter.ExportTensorRTPackage.INPUT_TYPES()["optional"]["package_name"][1]["default"], "")

    def test_quantization_refusal(self):
        with self.assertRaisesRegex(manifest.ManifestError, "representative denoising"):
            exporter.load_calibration("", 64, 64)

    def test_unsupported_hardware(self):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)), patch.object(torch.cuda, "get_device_name", return_value="Turing"):
            with self.assertRaisesRegex(manifest.ManifestError, "requires SM"):
                exporter.validate_precision("bf16")


class MemoryTests(unittest.TestCase):
    def test_failed_context_does_not_retain_engine_in_traceback(self):
        runtime = importlib.import_module("trtnodes.runtime")
        references = []
        class Engine:
            def create_execution_context(self):
                return None
        class Runtime:
            def deserialize_cuda_engine(self, data):
                engine = Engine()
                references.append(weakref.ref(engine))
                return engine
        manifest.TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=manifest.TEMP_ROOT) as temporary:
            root = Path(temporary)
            engine = root / "candidate.engine"
            engine.write_bytes(b"engine")
            unet = runtime.TensorRTUNet.__new__(runtime.TensorRTUNet)
            torch.nn.Module.__init__(unet)
            unet._active_key = unet._loaded = None
            unet._runtime = Runtime()
            unet._package = types.SimpleNamespace(root=root)
            unet._trt = None
            unet._check_compatibility = lambda spec: None
            spec = types.SimpleNamespace(path=engine, latent_shape=(1, 4, 8, 8))
            with patch.object(runtime, "validate_interface", new=lambda engine, trt, shape: (77, 77, 154)):
                try:
                    unet._engine_for(spec)
                except RuntimeError as exc:
                    self.assertIn("execution context", str(exc))
                    self.assertIsNone(references[0]())
                    self.assertIsNone(unet._runtime)
                else:
                    self.fail("Expected context creation to fail")

    def test_source_offloaded_after_materialization_before_export_copy(self):
        model = Mock()
        model.model.model_config = object.__new__(exporter.comfy.supported_models.SDXL)
        events = []
        exported = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=torch.nn.Linear(2, 2)))

        def load_copy(state, **kwargs):
            self.assertEqual(state["weight"].device.type, "cpu")
            self.assertFalse(state["weight"].requires_grad)
            events.append("copy")
            return exported

        model.model_state_dict_for_saving.side_effect = lambda *args: (
            events.append("materialize") or {"weight": torch.ones(2, requires_grad=True)})
        with patch.object(exporter.comfy.model_management, "load_models_gpu", side_effect=lambda models: events.append("load")), \
             patch.object(exporter.comfy.model_management, "unload_model_and_clones", side_effect=lambda source: events.append("offload")) as offload, \
             patch.object(exporter.comfy.sd, "load_diffusion_model_state_dict", side_effect=load_copy):
            self.assertIs(exporter.materialize_unet(model, "fp16"), exported)
        offload.assert_called_once_with(model)
        self.assertEqual(events, ["load", "materialize", "offload", "copy"])
        self.assertEqual(exported.model.diffusion_model.weight.dtype, torch.float16)

    def test_patcher_accounts_for_engine_and_releases_it(self):
        class Engine(torch.nn.Module):
            def __init__(self, path):
                super().__init__()
                self.package = types.SimpleNamespace(engines=[types.SimpleNamespace(path=path)])
                self.resident = 512

            def resident_memory(self):
                return self.resident

            def release(self):
                self.resident = 0

        manifest.TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=manifest.TEMP_ROOT) as temporary:
            path = Path(temporary) / "model.engine"
            path.write_bytes(b"plan")
            base = torch.nn.Module()
            base.diffusion_model = Engine(path)
            patcher = loader.TensorRTModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
            self.assertEqual(patcher.loaded_size(), 512)
            self.assertGreaterEqual(patcher.model_size(), patcher.loaded_size())
            clone = patcher.clone()
            self.assertEqual(clone.loaded_size(), 512)
            self.assertEqual(patcher.partially_unload(torch.device("cpu"), 256), 512)
            self.assertEqual(clone.loaded_size(), 0)
            del clone, patcher


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
