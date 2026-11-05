"""The Veda predictor: where it lives, and an optional download on the person's own click.

FunUI never hosts, bundles or mirrors the file. Its licence (the MiniMax H3 Community License) has conditions
on distribution, so the only route offered is a fetch straight from the official Hugging Face repository,
started by a button that sits next to the licence name and a link to its text. Streaming, the `.part` name and
the size and checksum checks are core/fetch.py's.
"""
import threading
from pathlib import Path

from .._core import fetch, log

SOURCE = "FunUI Veda"
HOST = "https://huggingface.co"
LICENSE = {"name": "MiniMax H3 Community License",
           "url": f"{HOST}/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/blob/main/LICENSE"}
# Sizes and checksums are the repositories' own listing (the Git LFS record of each file).
MODELS = (
    {"id": "t2va", "title": "T2VA / FL2VA", "hint": "Text, or first and last frames, to audio and video.",
     "repo": "Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview", "file": "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors",
     "size": 275415648, "sha256": "2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a"},
    {"id": "r2va", "title": "R2VA", "hint": "Reference images or videos to audio and video.",
     "repo": "Veda-Sparse/Minimax-H3-R2VA-Veda-Preview", "file": "minimax_h3_r2va_veda_preview_fp8.safetensors",
     "size": 275415752, "sha256": "23edace948693772be33d5cc945d302d2b037a29bd44bd57071bfbdd8b02517d"},
)
_state, _lock = {}, threading.Lock()


def folder() -> Path:
    import folder_paths
    return Path(folder_paths.models_dir) / "veda"


def files() -> list:
    """The predictor files in models/veda, by name. A `.part` is not one."""
    return sorted(p.name for p in folder().glob("*.safetensors")) if folder().is_dir() else []


def listing() -> dict:
    present = files()
    rows = []
    for m in MODELS:
        s = _state.get(m["id"]) or {}
        row = {k: m[k] for k in ("id", "title", "hint", "file", "size")}
        row["url"] = f"{HOST}/{m['repo']}/resolve/main/{m['file']}"
        row["state"] = "downloading" if s.get("state") == "downloading" else "ready" if m["file"] in present else "error" if s.get("state") == "error" else "absent"
        row.update(done=s.get("done", 0), error=s.get("error", ""))
        rows.append(row)
    return {"files": present, "models": rows, "license": LICENSE}


def download(model_id: str, get=None) -> bool:
    """Start the download in a thread. False when the id is unknown, the file is there already, or it is already running."""
    spec = next((m for m in MODELS if m["id"] == model_id), None)
    get = get or fetch.to_file
    with _lock:
        if spec is None or (_state.get(model_id) or {}).get("state") == "downloading" or spec["file"] in files():
            return False
        _state[model_id] = {"state": "downloading", "done": 0, "error": ""}

    def work():
        try:
            target = folder()
            target.mkdir(parents=True, exist_ok=True)
            get(f"{HOST}/{spec['repo']}/resolve/main/{spec['file']}", target / spec["file"],
                lambda done, total: _state[model_id].update(done=done), size=spec["size"], sha256=spec["sha256"])
            _state[model_id] = {"state": "ready", "done": spec["size"], "error": ""}
            log.info(SOURCE, f"Veda predictor {spec['title']} downloaded to {target / spec['file']}")
        except Exception as exc:  # noqa: BLE001 -- shown beside the button, and in the Log
            _state[model_id] = {"state": "error", "done": 0, "error": f"{type(exc).__name__}: {exc}"}
            log.warning(SOURCE, f"Veda predictor {spec['title']} could not be downloaded: {exc}")

    threading.Thread(target=work, name=f"funui-veda-{model_id}", daemon=True).start()
    return True
