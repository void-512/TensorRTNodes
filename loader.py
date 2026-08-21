from __future__ import annotations

import os
from pathlib import Path

import folder_paths
import comfy.model_management
import comfy.sd

from .manifest import ManifestError, load_manifest
from .runtime import TensorRTUNet


PACKAGE_FOLDER = "tensorrt_packages"
folder_paths.add_model_folder_path(PACKAGE_FOLDER, os.path.join(folder_paths.models_dir, "tensorrt"), is_default=True)


def _packages() -> list[str]:
    found = []
    for root in folder_paths.get_folder_paths(PACKAGE_FOLDER):
        if not os.path.isdir(root):
            continue
        for current, _dirs, files in os.walk(root):
            if "manifest.json" in files:
                found.append(os.path.relpath(current, root).replace("\\", "/"))
        for child in Path(root).iterdir():
            if child.is_dir() and (child / "manifest.json").is_file():
                found.append(child.name)
    return sorted(set(found))


def _resolve_package(name: str) -> Path:
    if not name or os.path.isabs(name):
        raise ManifestError("Select a TensorRT package stored under ComfyUI/models/tensorrt.")
    for root in folder_paths.get_folder_paths(PACKAGE_FOLDER):
        root = Path(os.path.abspath(root))
        candidate = Path(os.path.abspath(root / name))
        if candidate.parent == root or root in candidate.parents:
            if (candidate / "manifest.json").is_file():
                return candidate
    raise ManifestError(f"TensorRT package not found: {name}")


class LoadTensorRTEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"package": (_packages(), {"tooltip": "Package directory under ComfyUI/models/tensorrt."}),
                              "verify_integrity": ("BOOLEAN", {"default": False, "tooltip": "Hash the checkpoint and engines before loading; slow for large packages."})}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Loads merged SDXL components and dispatches their UNet to the matching fixed-resolution TensorRT engine."

    def load(self, package: str, verify_integrity: bool):
        manifest = load_manifest(_resolve_package(package), verify_integrity=verify_integrity)
        model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            str(manifest.checkpoint), output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        if model is None or clip is None or vae is None:
            raise RuntimeError("The TensorRT package checkpoint did not provide MODEL, CLIP, and VAE.")
        model.add_object_patch("diffusion_model", TensorRTUNet(manifest))
        return model, clip, vae


class LoadMergedModelForDetailer:
    """Load the normal merged model only after stage-one image selection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Selected image from the TensorRT generation stage."}),
                "engine_model": ("MODEL", {"tooltip": "MODEL output from Load TensorRT Engine."}),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("MERGED_MODEL", "IMAGE")
    FUNCTION = "load"
    CATEGORY = "TensorRT/loaders"
    DESCRIPTION = "Releases TensorRT, then loads the package's normal merged MODEL for a dynamic-size detailer stage."

    def load(self, image, engine_model):
        unet = engine_model.get_model_object("diffusion_model")
        if not isinstance(unet, TensorRTUNet):
            raise RuntimeError("Connect the TensorRT MODEL output from Load TensorRT Engine.")
        checkpoint = unet.checkpoint
        comfy.model_management.unload_model_and_clones(engine_model)
        unet.release()
        model, _clip, _vae, _ = comfy.sd.load_checkpoint_guess_config(
            str(checkpoint), output_vae=False, output_clip=False, embedding_directory=[]
        )
        if model is None:
            raise RuntimeError(f"The TensorRT package checkpoint did not provide a MODEL: {checkpoint}")
        return model, image


NODE_CLASS_MAPPINGS = {
    "LoadTensorRTEngine": LoadTensorRTEngine,
    "LoadMergedModelForDetailer": LoadMergedModelForDetailer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTensorRTEngine": "Load TensorRT Engine",
    "LoadMergedModelForDetailer": "Load Merged Model for Detailer",
}
