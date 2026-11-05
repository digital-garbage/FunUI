import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_failed_comfy_logger_import_leaves_no_stale_app_package():
    """Outside ComfyUI, `app` is our own folder; it must not stay cached and hide ComfyUI's real `app` later."""
    code = (
        "import sys; sys.path.insert(0, %r); import core.backend_log; "
        "print('app' in sys.modules and getattr(sys.modules['app'], '__file__', 1) is None)"
    ) % str(ROOT)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert out.stdout.strip().endswith("False"), out.stdout + out.stderr
