# Time Adapter Branch

Branch: `time_adapter`

## Goal

This branch adds a native time adapter for Kandinsky-5 I2V:
- `condition_fps`
- `latent_time`

The goal is to preserve the same simple control pattern:
- image
- prompt
- conditional FPS
- optional explicit latent time schedule

## Behavior

Checkpoint release:
- Hugging Face: `https://huggingface.co/kandinskylab/Time-Adapter`

This branch contains the native inference code for the time adapter, while the released TA-only step5750 checkpoint is published in the Hugging Face repository above.

The native time adapter is wired directly into DiT:
- `condition_fps` modulates the global time embedding
- `latent_time` modulates per-frame visual embeddings

This is a native model extension, not a PEFT/LoRA adapter.

## Main files

Core model:
- `kandinsky/models/dit.py`

Sampling path:
- `kandinsky/generation_utils.py`
- `kandinsky/i2v_pipeline.py`

Convenient entry points:
- `test.py`
- `examples/run_i2v_time_adapter.py`
- `configs/k5_lite_i2v_5s_time_adapter_release.yaml`

## Quickstart

Run from the `time_adapter` branch of `repos/kandinsky-5`.

Use the project environment with `requirements.txt` installed before running the examples.

### CLI

Uniform tempo with only global FPS control:

```bash
python test.py \
  --config ./configs/k5_lite_i2v_5s_time_adapter_release.yaml \
  --image ./assets/test_image.jpg \
  --prompt "A girl runs through a city street" \
  --video_duration 5 \
  --condition_fps 15 \
  --output_filename ./results/time_adapter_fps15.mp4
```

Explicit latent-time schedule from JSON:

```bash
python test.py \
  --config ./configs/k5_lite_i2v_5s_time_adapter_release.yaml \
  --image ./assets/test_image.jpg \
  --prompt "A girl runs through a city street" \
  --video_duration 5 \
  --condition_fps 24 \
  --latent_time_path ./examples/latent_time_ramp.json \
  --output_filename ./results/time_adapter_latent_time.mp4
```

### Minimal Python Example

```bash
python examples/run_i2v_time_adapter.py \
  --config configs/k5_lite_i2v_5s_time_adapter_release.yaml \
  --image ./assets/test_image.jpg \
  --prompt "A girl runs through a city street" \
  --condition_fps 30 \
  --output ./results/time_adapter_example.mp4
```

Inline latent time example:

```bash
python examples/run_i2v_time_adapter.py \
  --config configs/k5_lite_i2v_5s_time_adapter_release.yaml \
  --image ./assets/test_image.jpg \
  --prompt "A girl runs through a city street" \
  --condition_fps 24 \
  --latent_time_json '[0.0, 0.04, 0.08, 0.12, 0.20, 0.32, 0.48, 0.68, 0.92, 1.20]' \
  --output ./results/time_adapter_nonuniform.mp4
```

## Latent Time Format

`latent_time` is a JSON list of frame timestamps in seconds. Example:

```json
[0.0, 0.04, 0.08, 0.12, 0.20, 0.32, 0.48]
```

Smaller gaps between consecutive values usually correspond to slower perceived local motion. Larger gaps usually correspond to faster perceived local motion.

## Status

Native I2V time adapter is connected in the current branch and can be driven through:
- `test.py --condition_fps ... --latent_time_path ...`
- `examples/run_i2v_time_adapter.py`
