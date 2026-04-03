#!/bin/bash
# Download model weights and LoRAs

set -e

LORA_DIR="loras"
mkdir -p $LORA_DIR

echo "Downloading LoRA weights..."

# FusioniX LoRA for temporal consistency
if [ ! -f "$LORA_DIR/fusionix_temporal_consistency.safetensors" ]; then
    echo "Downloading FusioniX LoRA..."
    # Replace with actual HuggingFace URL
    wget -O $LORA_DIR/fusionix_temporal_consistency.safetensors \
        "https://huggingface.co/swordenkisk/infinitetalk-loras/resolve/main/fusionix_temporal_consistency.safetensors" || \
        echo "Note: FusioniX LoRA not available yet"
fi

# LightX2V LoRA for 4-step inference
if [ ! -f "$LORA_DIR/lightx2v_4step_distilled.safetensors" ]; then
    echo "Downloading LightX2V LoRA..."
    # Replace with actual HuggingFace URL
    wget -O $LORA_DIR/lightx2v_4step_distilled.safetensors \
        "https://huggingface.co/swordenkisk/infinitetalk-loras/resolve/main/lightx2v_4step_distilled.safetensors" || \
        echo "Note: LightX2V LoRA not available yet"
fi

echo "Download complete!"
