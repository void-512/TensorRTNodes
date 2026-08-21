# TensorRT Nodes

This single ComfyUI custom-node package contains the existing separate nodes:

- **Export TensorRT Package (Fused Model)** builds fixed-resolution FP16 engines from fused `MODEL`, `CLIP`, and `VAE` inputs.
- **Load TensorRT Engine** loads the packaged engine, CLIP, and VAE for generation.
- **Load Merged Model for Detailer** releases TensorRT and loads the normal merged model from disk for a dynamic-size detailer stage.

Engine packages remain under `ComfyUI/models/tensorrt/<package_name>/`; their manifest format and workflow node types are unchanged.
