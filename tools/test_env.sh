#!/bin/sh
# Test environment for a machine with no GPU (a cloud session): CPU torch, pytest, a ComfyUI checkout.
# Safe to run again; each step is skipped when already there. Prints the command that runs the suite.
set -e
COMFY=${COMFYUI_ROOT:-$HOME/ComfyUI}
# The three together, from the CPU index: ComfyUI's requirements would otherwise pull a CUDA torchvision that no longer matches.
python3 -c "import torch, torchvision" 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
python3 -c "import pytest" 2>/dev/null || pip install -q pytest
[ -f "$COMFY/comfy/supported_models.py" ] || git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
pip install -q -r "$COMFY/requirements.txt"
HERE=$(cd "$(dirname "$0")" && pwd)
echo "COMFYUI_ROOT=$COMFY PYTHONPATH=$HERE python3 -m pytest -p cpuforce -q"
