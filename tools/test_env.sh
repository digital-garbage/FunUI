#!/bin/sh
# Test environment for a machine with no GPU (a cloud session): CPU torch, pytest, a ComfyUI checkout.
# Safe to run again; each step is skipped when already there. Prints the command that runs the suite.
set -e
COMFY=${COMFYUI_ROOT:-$HOME/ComfyUI}
# The three together, from the CPU index: ComfyUI's requirements would otherwise pull a CUDA torchvision that no longer matches.
python3 -c "import torch, torchvision" 2>/dev/null || pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
python3 -c "import pytest" 2>/dev/null || pip install pytest
[ -f "$COMFY/comfy/supported_models.py" ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
pip install -r "$COMFY/requirements.txt"
# Optional: the Shot camera's tagger (without it those tests skip and the camera falls back to word rules).
python3 -c "import spacy" 2>/dev/null || pip install spacy || true
python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || python3 -m spacy download en_core_web_sm || true
HERE=$(cd "$(dirname "$0")" && pwd)
echo "COMFYUI_ROOT=$COMFY PYTHONPATH=$HERE python3 -m pytest -p cpuforce -q"
