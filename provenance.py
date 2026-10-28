from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import folder_paths
import comfy.lora
import comfy.lora_convert
import comfy.model_detection
import comfy.sd
import comfy.utils
import nodes
import torch

from .manifest import ManifestError, source_name, validate_chain


TORCH_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
                "float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}


def capture_dtypes(chain, model, clip, vae):
    clip_dtypes = clip.cond_stage_model.dtypes
    if len(clip_dtypes) != 1:
        raise ManifestError("Cannot reconstruct a source CLIP with mixed loading dtypes.")
    dtypes = {"model": str(model.model_dtype()).removeprefix("torch."),
              "clip": str(next(iter(clip_dtypes))).removeprefix("torch."),
              "vae": str(vae.vae_dtype).removeprefix("torch.")}
    return validate_chain({**chain, "load_dtypes": dtypes})


def source_vae(state, config, metadata, dtype):
    weights = comfy.utils.state_dict_prefix_replace(state, {key: "" for key in config.vae_key_prefix}, filter_keys=True)
    return comfy.sd.VAE(sd=config.process_vae_state_dict(weights), metadata=metadata, dtype=TORCH_DTYPES[dtype])


def apply_lora(model, clip, lora, strength_model, strength_clip, metadata):
    if model is not None and clip is not None:
        return comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=metadata)
    # The other component's keys are intentionally unused when loading just MODEL or CLIP.
    lora = comfy.lora_convert.convert_lora(lora)
    if model is not None:
        patches = comfy.lora.load_lora(lora, comfy.lora.model_lora_keys_unet(model.model, {}), log_missing=False)
        model = model.clone()
        model.add_patches(patches, strength_model)
        if metadata:
            model.set_attachments("lora_metadata", metadata)
    if clip is not None:
        patches = comfy.lora.load_lora(lora, comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {}), log_missing=False)
        clip = clip.clone()
        clip.add_patches(patches, strength_clip)
        if metadata:
            clip.patcher.set_attachments("lora_metadata", metadata)
    return model, clip


def resolve_source(kind, name):
    source_name(name)
    path = folder_paths.get_full_path(kind, name)
    roots = [Path(root).resolve() for root in folder_paths.get_folder_paths(kind)]
    if path is None or not any(Path(path).resolve().is_relative_to(root) for root in roots):
        raise ManifestError(f"Missing or unsafe {kind} file: {name}")
    return path


def reconstruct(chain, *, output_clip=True, output_vae=True, output_model=True, model_options=None, disable_dynamic=False):
    validate_chain(chain)
    checkpoint = resolve_source("checkpoints", chain["base_model"])
    loras = [(item, resolve_source("loras", item["name"])) for item in chain["loras"]]
    options = {} if model_options is None else model_options.copy()
    te_options = {}
    if "load_dtypes" in chain:
        options["dtype"] = TORCH_DTYPES[chain["load_dtypes"]["model"]]
        te_options["dtype"] = TORCH_DTYPES[chain["load_dtypes"]["clip"]]
    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
        checkpoint, output_model=output_model, output_clip=output_clip, output_vae=output_vae and "load_dtypes" not in chain,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        model_options=options, te_model_options=te_options, disable_dynamic=disable_dynamic,
    )
    if output_vae and "load_dtypes" in chain:
        state, metadata = comfy.utils.load_torch_file(checkpoint, return_metadata=True)
        prefix = comfy.model_detection.unet_prefix_from_state_dict(state)
        config = comfy.model_detection.model_config_from_unet(state, prefix, metadata=metadata)
        vae = source_vae(state, config, metadata, chain["load_dtypes"]["vae"])
    for item, path in loras:
        if (model is None or item["strength_model"] == 0) and (clip is None or item["strength_clip"] == 0):
            continue
        lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        model, clip = apply_lora(model, clip, lora, item["strength_model"], item["strength_clip"], metadata)
    return model, clip, vae


def recover_chain(dynprompt, unique_id):
    if dynprompt is None or unique_id is None:
        raise ManifestError("Source recovery requires ComfyUI DYNPROMPT and UNIQUE_ID hidden inputs.")

    def literal(inputs, name):
        value = inputs.get(name)
        if isinstance(value, (list, dict)) or value is None:
            raise ManifestError(f"Cannot recover linked or missing source setting {name}; use literal loader settings.")
        return value

    traced = {}

    def trace(link, kind, visiting=()):
        if not isinstance(link, list) or len(link) != 2 or not isinstance(link[0], str):
            raise ManifestError(f"Expected a connected {kind} source.")
        node_id, slot = link
        if node_id in visiting:
            raise ManifestError("Cycle in TensorRT source graph.")
        key = (node_id, slot, kind)
        if key in traced:
            return traced[key]
        node = dynprompt.get_node(node_id)
        inputs = node["inputs"]
        cls = nodes.NODE_CLASS_MAPPINGS.get(node["class_type"])
        if cls in (nodes.CheckpointLoaderSimple, nodes.unCLIPCheckpointLoader) and slot == {"MODEL": 0, "CLIP": 1, "VAE": 2}[kind]:
            traced[key] = (source_name(literal(inputs, "ckpt_name")), [])
            return traced[key]
        if cls in (nodes.LoraLoader, nodes.LoraLoaderModelOnly) and kind in {"MODEL", "CLIP"} and slot == (0 if kind == "MODEL" else 1):
            if cls is nodes.LoraLoaderModelOnly and kind != "MODEL":
                raise ManifestError("Model-only LoRA has no CLIP output.")
            base, previous = trace(inputs["model" if kind == "MODEL" else "clip"], kind, (*visiting, node_id))
            strength = literal(inputs, "strength_model" if kind == "MODEL" else "strength_clip")
            traced[key] = (base, [*previous, (node_id, source_name(literal(inputs, "lora_name")), strength)])
            return traced[key]
        raise ManifestError(f"Cannot reconstruct {kind} through {node['class_type']} (node {node_id}, output {slot}). Use native checkpoint/LoRA loaders without merges or custom model changes.")

    inputs = dynprompt.get_node(str(unique_id))["inputs"]
    model = trace(inputs.get("model"), "MODEL")
    clip = trace(inputs.get("clip"), "CLIP")
    vae = trace(inputs.get("vae"), "VAE")
    if model[0] != clip[0] or vae[0] != model[0]:
        raise ManifestError("Exporter MODEL, CLIP, and VAE do not share a reconstructible source chain.")
    # Preserve each component's application order. Independent MODEL/CLIP steps commute.
    steps, dependencies = {}, {}
    for component, (_, history) in (("strength_model", model), ("strength_clip", clip)):
        previous = None
        for node_id, name, strength in history:
            item = steps.setdefault(node_id, {"name": name, "strength_model": 0.0, "strength_clip": 0.0})
            item[component] = strength
            predecessors = dependencies.setdefault(node_id, [])
            if previous is not None and previous not in predecessors:
                predecessors.append(previous)
            previous = node_id
    try:
        ordered = TopologicalSorter(dependencies).static_order()
        chain = validate_chain({"base_model": model[0], "loras": [steps[node_id] for node_id in ordered]})
    except CycleError as exc:
        raise ManifestError("MODEL and CLIP LoRA application orders conflict.") from exc
    resolve_source("checkpoints", chain["base_model"])
    for item in chain["loras"]:
        resolve_source("loras", item["name"])
    return chain
