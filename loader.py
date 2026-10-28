from __future__ import annotations

import os
from pathlib import Path

import torch
import folder_paths
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.sd
import comfy.supported_models
import comfy.utils
from comfy_api.latest import io

from .manifest import ManifestError, load_manifest
from .provenance import TORCH_DTYPES, apply_lora, reconstruct, resolve_source, source_vae
from .runtime import TensorRTUNet, TensorRTModelPatcher


PACKAGE_FOLDER = "tensorrt_packages"
folder_paths.add_model_folder_path(PACKAGE_FOLDER, os.path.join(folder_paths.models_dir, "tensorrt"), is_default=True)


def _packages():
    found = set()
    for root in folder_paths.get_folder_paths(PACKAGE_FOLDER):
        for current, _dirs, files in os.walk(root):
            if "manifest.json" in files:
                found.add(os.path.relpath(current, root).replace("\\", "/"))
    return sorted(found)


def _resolve_package(name):
    if not name or Path(name).is_absolute() or "\\" in name or any(p in {"", ".", ".."} for p in name.split("/")):
        raise ManifestError("Select a package under ComfyUI/models/tensorrt.")
    for root in folder_paths.get_folder_paths(PACKAGE_FOLDER):
        root = Path(root).resolve()
        candidate = (root / name).resolve()
        if candidate.is_relative_to(root) and (candidate / "manifest.json").is_file():
            return candidate
    raise ManifestError(f"TensorRT package not found: {name}")


class LoadTensorRTEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"package": (_packages(),)}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Loads source CLIP/VAE with ordered LoRAs and a fixed-resolution TensorRT SDXL MODEL."

    @classmethod
    def IS_CHANGED(cls, package):
        root = _resolve_package(package)
        paths = [root / "manifest.json"] + list((root / "engines").glob("*.engine"))
        return tuple((str(p), p.stat().st_ino, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(paths))

    def load(self, package):
        manifest = load_manifest(_resolve_package(package))
        unet = TensorRTUNet(manifest)
        checkpoint = resolve_source("checkpoints", manifest.chain["base_model"])
        state, metadata = comfy.utils.load_torch_file(checkpoint, return_metadata=True)
        prefix = comfy.model_detection.unet_prefix_from_state_dict(state)
        config = comfy.model_detection.model_config_from_unet(state, prefix, metadata=metadata)
        if type(config) is not comfy.supported_models.SDXL:
            raise ManifestError("TensorRTNodes currently supports base SDXL UNets only.")
        vae = source_vae(state, config, metadata, manifest.chain["load_dtypes"]["vae"])
        config.unet_config["disable_unet_model_creation"] = True
        config.set_inference_dtype(torch.float32, None)
        base = config.get_model(state, prefix, device=comfy.model_management.unet_offload_device())
        base.diffusion_model = unet
        model = TensorRTModelPatcher(base, load_device=unet.device, offload_device=comfy.model_management.unet_offload_device())
        model.add_callback(comfy.patcher_extension.CallbacksMP.ON_DETACH, lambda patcher, unpatch_all: unet.release())
        _, clip, _, _ = comfy.sd.load_state_dict_guess_config(
            state, output_model=False, output_clip=True, output_vae=False,
            embedding_directory=folder_paths.get_folder_paths("embeddings"), metadata=metadata,
            te_model_options={"dtype": TORCH_DTYPES[manifest.chain["load_dtypes"]["clip"]]},
        )
        if clip is None or vae is None:
            raise ManifestError("Source checkpoint must provide CLIP and VAE.")
        for item in manifest.chain["loras"]:
            path = resolve_source("loras", item["name"])
            if item["strength_clip"] != 0:
                lora, lora_metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
                _, clip = apply_lora(None, clip, lora, 0, item["strength_clip"], lora_metadata)
        return model, clip, vae


class ModelSwitcher(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelSwitcher", display_name="Model Switcher", category="TensorRT/loaders",
            description="Waits for the trigger, releases this TensorRT engine, and reconstructs its checkpoint and ordered LoRAs.",
            inputs=[io.AnyType.Input("trigger"), io.Model.Input("engine_model")],
            outputs=[io.Model.Output(display_name="MODEL")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Releasing GPU resources must happen on every execution, even with cached triggers.
        return float("nan")

    @classmethod
    def execute(cls, trigger, engine_model):
        unet = engine_model.get_model_object("diffusion_model")
        if not isinstance(unet, TensorRTUNet):
            raise ManifestError("Connect the MODEL from Load TensorRT Engine.")
        manifest = load_manifest(unet.package.root)
        if manifest.chain != unet.package.chain:
            raise ManifestError("TensorRT package source changed; reload Load TensorRT Engine before switching models.")
        unet.release()
        comfy.model_management.unload_model_and_clones(engine_model)
        model, _, _ = reconstruct(manifest.chain, output_clip=False, output_vae=False)
        if model is None:
            raise ManifestError("Source checkpoint did not provide a MODEL.")
        return io.NodeOutput(model)


NODE_CLASS_MAPPINGS = {"LoadTensorRTEngine": LoadTensorRTEngine, "ModelSwitcher": ModelSwitcher}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadTensorRTEngine": "Load TensorRT Engine", "ModelSwitcher": "Model Switcher"}
