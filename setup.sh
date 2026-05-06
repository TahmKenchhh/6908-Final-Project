#!/bin/bash
set -e

echo "=== Voice Assistant Setup ==="

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/4] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    swig \
    liblgpio-dev \
    python3-pip \
    python3-venv

# ── 2. Python dependencies ────────────────────────────────────────────────────
echo ""
echo "[2/4] Installing Python packages..."
pip install -r requirements.txt

# ── 3. Piper voice model (HAL 9000) ───────────────────────────────────────────
echo ""
echo "[3/4] Downloading Piper TTS voice model (HAL 9000)..."

MODEL_DIR="$(dirname "$0")/models"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/hal.onnx" ]; then
    wget -q --show-progress "https://huggingface.co/campwill/HAL-9000-Piper-TTS/resolve/main/hal.onnx" \
        -O "$MODEL_DIR/hal.onnx"
    wget -q --show-progress "https://huggingface.co/campwill/HAL-9000-Piper-TTS/resolve/main/hal.onnx.json" \
        -O "$MODEL_DIR/hal.onnx.json"
    echo "HAL voice model downloaded."
else
    echo "HAL voice model already present, skipping."
fi

# ── 4. Ollama models ──────────────────────────────────────────────────────────
echo ""
echo "[4/4] Pulling Ollama models..."
ollama pull qwen3:1.7b      # text LLM
ollama pull qwen3-vl:2b     # local vision fallback

echo ""
echo "Optional: keep both models loaded in memory simultaneously to avoid"
echo "  switch latency. Add OLLAMA_MAX_LOADED_MODELS=2 to the Ollama service:"
echo "    sudo mkdir -p /etc/systemd/system/ollama.service.d"
echo '    echo -e "[Service]\\nEnvironment=\"OLLAMA_MAX_LOADED_MODELS=2\"" \\'
echo "      | sudo tee /etc/systemd/system/ollama.service.d/override.conf"
echo "    sudo systemctl daemon-reload && sudo systemctl restart ollama"

echo ""
echo "=== Setup complete! ==="
echo "Optional: export GEMINI_API_KEY=\"...\" for fast cloud vision (free tier)."
echo "Run the assistant with:  .venv/bin/python main.py"
