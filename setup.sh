#!/bin/bash
# Piper Voice Trainer — One-click setup
# Run this once on your ML machine to install everything.
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

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
# Pin setuptools<70 — v70+ deprecated and v78+ removed pkg_resources,
# which pytorch-lightning 1.x and piper_train need at runtime.
pip install --upgrade pip wheel "setuptools<70"
pip install -r requirements.txt

# Clone piper repo and install the training module
# (piper_train is not published on PyPI — must install from source)
PIPER_DIR="piper"
if [ ! -d "$PIPER_DIR" ]; then
    echo "Cloning piper repository..."
    git clone https://github.com/rhasspy/piper.git "$PIPER_DIR"
fi

echo "Installing piper training dependencies..."
# piper_train uses Lightning 1.x APIs (add_argparse_args etc.) removed in 2.0
# setuptools provides pkg_resources, needed by piper_train at runtime
pip install "pytorch-lightning>=1.7,<2.0" "onnxruntime>=1.11.0" piper-phonemize "setuptools<70"

echo "Installing piper training module (skipping strict dep pins)..."
cd "$PROJECT_ROOT/$PIPER_DIR/src/python"

# Fix bug: preprocess crashes when utterances < CPU count (batch size becomes 0)
sed -i 's/raise ValueError("n must be at least one")/n = max(1, n)  # patched/' piper_train/preprocess.py

pip install --no-deps -e .

# Build the monotonic alignment Cython extension
# The stock setup.py derives a fully-qualified extension name from __init__.py
# files up the tree, causing --inplace to write the .so to a wrong relative path.
# We bypass it and build with an explicit short module name instead.
if [ -d piper_train/vits/monotonic_align ]; then
    echo "Building monotonic alignment extension..."
    pip install cython
    cd piper_train/vits/monotonic_align
    mkdir -p monotonic_align
    python -c "
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
ext = Extension('monotonic_align.core', sources=['core.pyx'],
                include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([ext]), script_args=['build_ext', '--inplace'])
"
fi

# Return to project root for checkpoint download
cd "$PROJECT_ROOT"

# Verify pkg_resources is available (pytorch-lightning 1.x needs it)
pip install "setuptools<70"
python -c "import pkg_resources; print('pkg_resources OK')"

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
