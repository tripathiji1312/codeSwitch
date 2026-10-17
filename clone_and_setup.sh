#!/usr/bin/env bash
# ==============================================================================
# CodeSwitch-TTS: GitHub Clone & Environment Setup Script
# Works on Linux environments and Kaggle GPU instances.
# ==============================================================================

set -e

# Default repository URL (can be overridden via REPO_URL environment variable)
REPO_URL="${REPO_URL:-https://github.com/tripathiji1312/codeSwitch.git}"
TARGET_DIR="${TARGET_DIR:-codeSwitch}"

echo "========================================================================"
echo " [CodeSwitch-TTS] Cloning Repository & Setting Up Environment"
echo "========================================================================"
echo "Repository: $REPO_URL"
echo "Target Dir: $TARGET_DIR"

# 1. Clone repository if not already present
if [ ! -d "$TARGET_DIR" ]; then
    echo ">>> Cloning repository from $REPO_URL..."
    git clone "$REPO_URL" "$TARGET_DIR"
else
    echo ">>> Target directory $TARGET_DIR already exists. Pulling latest..."
    cd "$TARGET_DIR" && git pull || true
    cd ..
fi

cd "$TARGET_DIR"

# 2. Staged Dependency Installation (Prevents Kaggle / pip version conflicts)
echo ">>> Installing dependencies (staged to avoid package conflicts)..."

# Stage 1: Parler-TTS from GitHub (brings its transformers/audiotools chain)
pip install -q -U git+https://github.com/huggingface/parler-tts.git

# Stage 2: Pin transformers to the exact version Parler-TTS v0.2.2 targets
pip install -q "transformers==4.46.1"

# Stage 3: Core acoustic, audio codec, and scientific libraries
pip install -q -r requirements.txt

# Stage 4: protobuf LAST and >= 5.26 (ships google.protobuf.runtime_version).
# Parler-TTS -> descript-audiotools pins protobuf<5 and silently downgrades it to
# 4.25.9, which breaks the Parler-TTS import and forces the noise-only fallback.
pip install -q "protobuf==5.29.5"

echo ">>> Active versions:"
python -c "import importlib.metadata as md; print('   protobuf:', md.version('protobuf'), '| transformers:', md.version('transformers'))"

echo "========================================================================"
echo " [CodeSwitch-TTS] Setup complete! Running benchmark verification..."
echo "========================================================================"

python kaggle_demo.py
