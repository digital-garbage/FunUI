"""pytest plugin: run ComfyUI on the CPU, so the suite works on a machine without a GPU.

    COMFYUI_ROOT=/path/to/ComfyUI PYTHONPATH=tools python -m pytest -p cpuforce
"""
import os
import sys


def pytest_configure(config):
    root = os.environ.get("COMFYUI_ROOT")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        from comfy.cli_args import args
    except ImportError:
        return
    args.cpu = True
