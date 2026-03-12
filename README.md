# ComfyUI Instant Reference

A custom node for ComfyUI that turns a reference image set into a quick LoRA and applies it back to the current workflow.

It is aimed at fast character or style adaptation runs with minimal setup. The node prepares captions, runs a lightweight training profile, caches matching runs, and reuses the generated LoRA when possible.
Under the hood, it uses `sd-scripts` for tagging and LoRA training.

## Nodes

### Instant Reference LoRA

Main training node. It takes a `MODEL`, `CLIP`, and reference `IMAGE` batch, prepares captions, runs the selected training profile, caches identical runs, and outputs the patched `MODEL`, patched `CLIP`, and generated `lora_path`.

### Reference Tagging Options

Helper node for caption generation settings. Use it to control WD tagger thresholds and basic caption cleanup such as prepending tags, appending tags, excluding tags, replacing tags, and underscore removal.

### Reference Train Options

Helper node for training overrides. Use it to override steps, learning rate, network size, alpha, resolution, seed, caching behavior, and whether to force retraining instead of reusing a cached result.

## Profiles

### SDXL Reference LoRA

Default SDXL-oriented profile based on `sdxl_train_network.py`. It trains a LoCon-style LoRA at `1024x1024`, uses `bf16`, keeps the run short with `50` default steps, and is meant for fast reference adaptation on SDXL checkpoints.

### Anima Reference LoRA

An Anima-oriented profile based on `anima_train_network.py`. It also runs at `1024x1024` with `50` default steps, trains a lightweight LoRA for the UNet only, and requires an additional `VAE` input through the profile slot.
