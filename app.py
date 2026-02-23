#!/usr/bin/env python3
"""Piper Voice Trainer — Create custom TTS voices from audio samples.

Upload WAV file(s) of a voice, and this app will:
1. Split audio into sentence-length segments
2. Transcribe each segment with Whisper
3. Create an LJSpeech training dataset
4. Fine-tune a Piper TTS model from a pre-trained checkpoint
5. Export to ONNX format ready for Piper inference

Requirements: bash setup.sh (installs everything)
Usage: python app.py  →  open http://localhost:7860
"""

import gc
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
WORKSPACE_DIR = BASE_DIR / "workspace"
CHECKPOINT_FILE = CHECKPOINT_DIR / "en_US-lessac-medium.ckpt"
SAMPLE_RATE = 22050

log = logging.getLogger("piper-trainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# GPU Detection
# ---------------------------------------------------------------------------

def detect_gpus():
    """Detect NVIDIA GPUs via nvidia-smi. Returns list of label strings."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                idx, name, mem = parts[0], parts[1], parts[2]
                gpus.append(f"{idx}: {name} ({mem} MB)")
        return gpus if gpus else ["0: Default GPU"]
    except Exception:
        return ["0: Default GPU"]


# ---------------------------------------------------------------------------
# Step 1 — Audio Preparation
# ---------------------------------------------------------------------------

def prepare_audio(input_files, output_dir):
    """Convert, normalize, and split audio into sentence-length segments.

    Returns list of (segment_name, wav_path) tuples.
    """
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and concatenate all input files
    combined = AudioSegment.empty()
    for f in input_files:
        audio = AudioSegment.from_file(f)
        combined += audio

    # Convert to Piper's expected format: 22050 Hz, mono, 16-bit
    combined = (combined
                .set_frame_rate(SAMPLE_RATE)
                .set_channels(1)
                .set_sample_width(2))

    # Normalize volume
    combined = combined.normalize()

    # Split on silence — tune thresholds to get sentence-length chunks
    silence_thresh = combined.dBFS - 16
    segments = split_on_silence(
        combined,
        min_silence_len=400,       # 400ms pause = sentence boundary
        silence_thresh=silence_thresh,
        keep_silence=200,          # keep 200ms padding on each side
    )

    # Filter: keep segments between 1s and 15s
    saved = []
    for i, seg in enumerate(segments):
        duration_s = len(seg) / 1000.0
        if duration_s < 1.0 or duration_s > 15.0:
            continue
        name = f"segment_{i:04d}"
        path = output_dir / f"{name}.wav"
        seg.export(str(path), format="wav",
                   parameters=["-ar", str(SAMPLE_RATE), "-ac", "1"])
        saved.append((name, str(path)))

    return saved


# ---------------------------------------------------------------------------
# Step 2 — Transcription
# ---------------------------------------------------------------------------

def transcribe_segments(segments, model_size="large-v3-turbo"):
    """Transcribe WAV segments using faster-whisper on GPU.

    Returns list of (segment_name, transcript_text) tuples.
    """
    from faster_whisper import WhisperModel

    log.info(f"Loading Whisper model: {model_size}")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    transcriptions = []
    for name, wav_path in segments:
        segs, _ = model.transcribe(wav_path, language="en")
        text = " ".join(s.text.strip() for s in segs).strip()
        # Clean up text: remove special chars that break espeak phonemization
        text = re.sub(r'[^\w\s.,!?\'-]', '', text)
        if text and len(text) > 2:
            transcriptions.append((name, text))

    # Free GPU memory before training
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return transcriptions


# ---------------------------------------------------------------------------
# Step 3 — Dataset Creation
# ---------------------------------------------------------------------------

def create_dataset(wavs_dir, transcriptions, dataset_dir):
    """Create LJSpeech-format dataset from segments and transcriptions.

    LJSpeech format:
      dataset_dir/wav/<name>.wav
      dataset_dir/metadata.csv  with lines: name|text
    """
    dataset_dir = Path(dataset_dir)
    wav_out = dataset_dir / "wav"
    wav_out.mkdir(parents=True, exist_ok=True)

    metadata_lines = []
    for name, text in transcriptions:
        src = Path(wavs_dir) / f"{name}.wav"
        dst = wav_out / f"{name}.wav"
        if src.exists():
            shutil.copy2(str(src), str(dst))
            metadata_lines.append(f"{name}|{text}")

    csv_path = dataset_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines) + "\n")

    return len(metadata_lines)


# ---------------------------------------------------------------------------
# Step 4 — Piper Preprocessing
# ---------------------------------------------------------------------------

def preprocess(dataset_dir, training_dir):
    """Run piper_train.preprocess to prepare training cache. Yields log lines."""
    cmd = [
        sys.executable, "-u", "-m", "piper_train.preprocess",
        "--language", "en",
        "--input-dir", str(dataset_dir),
        "--output-dir", str(training_dir),
        "--dataset-format", "ljspeech",
        "--single-speaker",
        "--sample-rate", str(SAMPLE_RATE),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    yield f"  Running: {' '.join(cmd)}"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        env=env,
        bufsize=1,
    )

    for line in process.stdout:
        yield line.rstrip()

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"Preprocessing failed with exit code {process.returncode}"
        )


# ---------------------------------------------------------------------------
# Step 5 — Training
# ---------------------------------------------------------------------------

def train(training_dir, max_epochs, batch_size, gpu_index):
    """Run piper_train fine-tuning. Yields log lines for progress display."""
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Pre-trained checkpoint not found: {CHECKPOINT_FILE}\n"
            f"Run setup.sh to download it."
        )

    # PyTorch 2.6+ / Lightning Fabric passes weights_only=True to torch.load,
    # which rejects the Piper checkpoint (contains pathlib.PosixPath).
    # We must allowlist PosixPath BEFORE piper_train imports anything.
    # Use python -c with a wrapper that patches torch, then runs the module.
    bootstrap = (
        "import pathlib, torch; "
        "torch.serialization.add_safe_globals([pathlib.PosixPath]); "
        "import runpy; "
        "runpy.run_module('piper_train', run_name='__main__', alter_sys=True)"
    )

    cmd = [
        sys.executable, "-c", bootstrap,
        "--dataset-dir", str(training_dir),
        "--accelerator", "gpu",
        "--devices", "1",
        "--batch-size", str(batch_size),
        "--validation-split", "0.0",
        "--num-test-examples", "0",
        "--max_epochs", str(max_epochs),
        "--resume_from_checkpoint", str(CHECKPOINT_FILE),
        "--checkpoint-epochs", "1",
        "--precision", "32",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )

    for line in process.stdout:
        yield line.rstrip()

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"Training failed with exit code {process.returncode}"
        )


# ---------------------------------------------------------------------------
# Step 6 — ONNX Export
# ---------------------------------------------------------------------------

def find_latest_checkpoint(training_dir):
    """Find the most recent training checkpoint file."""
    pattern = str(
        Path(training_dir) / "lightning_logs" / "version_*"
        / "checkpoints" / "*.ckpt"
    )
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {training_dir}/lightning_logs/"
        )
    return max(checkpoints, key=os.path.getmtime)


def export_onnx(training_dir, output_dir, voice_name):
    """Export trained checkpoint to Piper ONNX format.

    Returns (onnx_path, json_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = find_latest_checkpoint(training_dir)
    log.info(f"Exporting checkpoint: {checkpoint}")

    onnx_path = output_dir / f"{voice_name}.onnx"
    json_path = output_dir / f"{voice_name}.onnx.json"

    cmd = [
        sys.executable, "-m", "piper_train.export_onnx",
        str(checkpoint),
        str(onnx_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"ONNX export failed (exit {result.returncode}):\n"
            f"{result.stderr}\n{result.stdout}"
        )

    # Copy training config as the .onnx.json sidecar
    config_src = Path(training_dir) / "config.json"
    if config_src.exists():
        shutil.copy2(str(config_src), str(json_path))
    else:
        log.warning(f"Training config not found at {config_src}")

    return str(onnx_path), str(json_path)


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(wav_files, voice_name, max_epochs, batch_size,
                 gpu_choice, whisper_model):
    """Full training pipeline. Generator that yields UI updates.

    Yields: (log_text, onnx_file_path, json_file_path)
    """
    # Validate inputs
    if not wav_files:
        yield "ERROR: No audio files uploaded.", None, None
        return

    voice_name = (voice_name or "custom_voice").strip()
    voice_name = re.sub(r'[^a-zA-Z0-9_-]', '_', voice_name)

    gpu_index = gpu_choice.split(":")[0].strip() if gpu_choice else "0"

    # Set up workspace
    workspace = WORKSPACE_DIR / voice_name
    if workspace.exists():
        shutil.rmtree(workspace)

    wavs_dir = workspace / "wavs"
    dataset_dir = workspace / "dataset"
    training_dir = workspace / "training"
    output_dir = workspace / "output"

    log_lines = []

    def log_msg(msg):
        log_lines.append(msg)
        return "\n".join(log_lines)

    try:
        # ---- Step 1: Audio preparation ----
        yield log_msg("STEP 1/6: Preparing audio..."), None, None

        file_paths = [f.name if hasattr(f, 'name') else str(f) for f in wav_files]
        yield log_msg(f"  Loaded {len(file_paths)} file(s)"), None, None

        segments = prepare_audio(file_paths, wavs_dir)
        yield log_msg(f"  Split into {len(segments)} segments (1-15s each)"), None, None

        if len(segments) < 5:
            yield log_msg(
                f"\nERROR: Only {len(segments)} usable segments found.\n"
                f"Need at least 5 segments (50+ recommended for good quality).\n"
                f"Try a longer audio sample with more speech."
            ), None, None
            return

        if len(segments) < 50:
            yield log_msg(
                f"  WARNING: {len(segments)} segments is low. "
                f"50+ recommended for decent quality."
            ), None, None

        # ---- Step 2: Transcription ----
        yield log_msg(
            f"\nSTEP 2/6: Transcribing {len(segments)} segments "
            f"with Whisper ({whisper_model})..."
        ), None, None

        transcriptions = transcribe_segments(segments, whisper_model)
        yield log_msg(
            f"  Transcribed {len(transcriptions)} segments successfully"
        ), None, None

        if len(transcriptions) < 5:
            yield log_msg(
                f"\nERROR: Only {len(transcriptions)} segments had usable "
                f"transcriptions. Need cleaner audio with more speech."
            ), None, None
            return

        # Show a few sample transcriptions
        for name, text in transcriptions[:3]:
            yield log_msg(f"  [{name}] \"{text[:80]}\""), None, None

        # ---- Step 3: Dataset creation ----
        yield log_msg("\nSTEP 3/6: Creating LJSpeech dataset..."), None, None

        count = create_dataset(wavs_dir, transcriptions, dataset_dir)
        yield log_msg(f"  Dataset: {count} entries"), None, None

        # ---- Step 4: Preprocessing ----
        yield log_msg("\nSTEP 4/6: Preprocessing for Piper training..."), None, None

        for line in preprocess(dataset_dir, training_dir):
            if line.strip():
                yield log_msg(f"  {line}"), None, None

        yield log_msg("  Preprocessing complete"), None, None

        # ---- Step 5: Training ----
        yield log_msg(
            f"\nSTEP 5/6: Training model (up to epoch {max_epochs}, "
            f"batch_size={batch_size}, GPU={gpu_index})..."
        ), None, None
        yield log_msg(
            "  Fine-tuning from pre-trained checkpoint. This will take a while."
        ), None, None

        last_update = time.time()
        for line in train(training_dir, max_epochs, batch_size, gpu_index):
            now = time.time()
            # Throttle UI updates — show epoch/loss lines, skip noise
            is_important = any(kw in line.lower() for kw in
                               ["epoch", "step", "loss", "error", "warning",
                                "saving", "checkpoint"])
            if is_important or (now - last_update > 5.0):
                yield log_msg(f"  {line}"), None, None
                last_update = now

        yield log_msg("  Training complete!"), None, None

        # ---- Step 6: Export ----
        yield log_msg("\nSTEP 6/6: Exporting to ONNX..."), None, None

        onnx_path, json_path = export_onnx(
            training_dir, output_dir, voice_name
        )
        yield log_msg(f"  Exported: {onnx_path}"), None, None
        yield log_msg(f"  Config:   {json_path}"), None, None

        yield log_msg(
            f"\nDONE! Model ready.\n"
            f"Copy these files to your Pi's models/piper/ directory:\n"
            f"  {onnx_path}\n"
            f"  {json_path}\n"
            f"Then set piper_voice to \"{voice_name}\" in config.json"
        ), onnx_path, json_path

    except Exception as e:
        yield log_msg(f"\nERROR: {e}"), None, None
        log.exception("Pipeline failed")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    """Build the Gradio web interface."""
    available_gpus = detect_gpus()

    with gr.Blocks(
        title="Piper Voice Trainer",
    ) as app:

        gr.Markdown(
            "# Piper Voice Trainer\n"
            "Upload voice audio, get a Piper TTS model. "
            "That's it."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.File(
                    label="Voice Recording(s)",
                    file_count="multiple",
                    file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                    type="filepath",
                )
                voice_name = gr.Textbox(
                    label="Voice Name",
                    value="custom_voice",
                    placeholder="e.g. mcp_tron",
                    max_lines=1,
                )
                train_btn = gr.Button(
                    "Start Training",
                    variant="primary",
                    size="lg",
                )

                with gr.Accordion("Advanced Options", open=False):
                    max_epochs = gr.Slider(
                        label="Max Training Epochs",
                        minimum=3000,
                        maximum=10000,
                        value=6000,
                        step=500,
                        info="Higher = better quality but longer training. "
                             "Fine-tunes from epoch 2164 checkpoint.",
                    )
                    batch_size = gr.Slider(
                        label="Batch Size",
                        minimum=4,
                        maximum=64,
                        value=32,
                        step=4,
                        info="Lower this if you get GPU out-of-memory errors.",
                    )
                    gpu_choice = gr.Dropdown(
                        label="GPU",
                        choices=available_gpus,
                        value=available_gpus[0] if available_gpus else "0",
                    )
                    whisper_model = gr.Dropdown(
                        label="Whisper Model (for transcription)",
                        choices=[
                            "large-v3-turbo",
                            "large-v3",
                            "medium",
                            "small",
                        ],
                        value="large-v3-turbo",
                        info="Larger = more accurate transcription.",
                    )

            with gr.Column(scale=1):
                log_output = gr.Textbox(
                    label="Progress",
                    lines=25,
                    max_lines=40,
                    interactive=False,
                )
                with gr.Row():
                    onnx_output = gr.File(label="Model (.onnx)")
                    json_output = gr.File(label="Config (.onnx.json)")

        train_btn.click(
            fn=run_pipeline,
            inputs=[audio_input, voice_name, max_epochs, batch_size,
                    gpu_choice, whisper_model],
            outputs=[log_output, onnx_output, json_output],
        )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
