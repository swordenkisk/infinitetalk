# InfiniteTalk

Sparse-Frame Video Dubbing Framework with 4-Step Inference

## Quick Start

```bash
pip install infinitetalk

# Download LoRAs
python -m infinitetalk.download_models

# Generate video
infinitetalk generate \
  --image reference.jpg \
  --audio speech.wav \
  --output video.mp4 \
  --config configs/720p_4step.yaml
```

## Features

- **Sparse-Frame Generation**: Infinite length with identity consistency
- **4-Step Inference**: 10x faster than standard diffusion
- **Dual LoRA System**: FusioniX (consistency) + LightX2V (speed)
- **FP8 Quantization**: 50% memory reduction
- **Multi-GPU**: Ulysses sequence parallelism

## Architecture

See `docs/ARCHITECTURE.md` for technical details on Sparse Reference Anchor Points (SRAP).

## Installation

### From Source

```bash
git clone https://github.com/swordenkisk/infinitetalk.git
cd infinitetalk
bash scripts/setup_env.sh
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.1.0
- CUDA >= 11.8 (for GPU support)

## Usage

### Python API

```python
from infinitetalk.pipelines import InfiniteTalkPipeline

pipeline = InfiniteTalkPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-xl"
)
pipeline.load_loras("loras/fusionix.safetensors", "loras/lightx2v.safetensors")

video = pipeline(
    image=reference_image,
    audio=audio_features,
    num_frames=120,
    num_inference_steps=4
)
```

### Command Line

```bash
infinitetalk generate \
  --image face.jpg \
  --audio speech.wav \
  --output output.mp4 \
  --num-frames 120 \
  --resolution 720p
```

### Web UI

```bash
python -m infinitetalk.api.gradio_app
```

### API Server

```bash
python -m infinitetalk.api.server --host 0.0.0.0 --port 8000
```

## Configuration

See `configs/` for example configurations:

- `480p_4step.yaml`: 480p resolution, 4-step inference
- `720p_4step.yaml`: 720p resolution, 4-step inference
- `infinite_length.yaml`: For videos up to 55 minutes
- `lora_config.yaml`: LoRA settings

## Benchmarks

```bash
python scripts/benchmark.py --resolution 720p --num-frames 120
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{infinitetalk2024,
  author = {swordenkisk},
  title = {InfiniteTalk: Sparse-Frame Video Dubbing},
  year = {2024},
  url = {https://github.com/swordenkisk/infinitetalk}
}
```
