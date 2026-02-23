#!/bin/bash
# Piper Voice Trainer — One-click setup
# Run this once on your ML machine to install everything.
set -e

echo "=== Piper Voice Trainer Setup ==="

# System dependencies
echo "Installing system packages..."
sudo apt update
sudo apt install -y espeak-ng ffmpeg build-essential python3-dev python3-venv

# Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python packages (this will take a while — includes PyTorch)..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# Clone piper repo and install the training module
# (piper_train is not published on PyPI — must install from source)
PIPER_DIR="piper"
if [ ! -d "$PIPER_DIR" ]; then
    echo "Cloning piper repository..."
    git clone https://github.com/rhasspy/piper.git "$PIPER_DIR"
fi

echo "Installing piper training module..."
cd "$PIPER_DIR/src/python"

# Relax strict version pins that conflict with modern packages
sed -i 's/pytorch-lightning~=1.7.0/pytorch-lightning>=1.7.0/' setup.py
sed -i 's/torch~=/torch>=/' setup.py
sed -i 's/onnxruntime~=/onnxruntime>=/' setup.py

pip install -e .
if [ -f build_monotonic_align.sh ]; then
    bash build_monotonic_align.sh
fi
cd "$OLDPWD"

# Download pre-trained checkpoint for fine-tuning
CHECKPOINT_DIR="checkpoints"
CHECKPOINT_FILE="$CHECKPOINT_DIR/en_US-lessac-medium.ckpt"
CHECKPOINT_URL="https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"

mkdir -p "$CHECKPOINT_DIR"

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Downloading pre-trained Piper checkpoint (~300MB)..."
    wget -O "$CHECKPOINT_FILE" "$CHECKPOINT_URL"
else
    echo "Checkpoint already downloaded."
fi

echo ""
echo "=== Setup complete! ==="
echo "To run the trainer:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo "Then open http://localhost:7860 in your browser."
