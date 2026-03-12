# comfyui-reference

A custom node for ComfyUI that turns a reference image set into a quick LoRA and applies it back to the current workflow.

It is aimed at fast character or style adaptation runs with minimal setup. The node prepares captions, runs a lightweight training profile, caches matching runs, and reuses the generated LoRA when possible.
