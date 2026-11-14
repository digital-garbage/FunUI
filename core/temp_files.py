"""What ComfyUI has in its temp directory and in its output directory.

Everything FunUI generates (clips, pictures, upscales, interpolations, previews) lands in temp, and ComfyUI empties
temp when it starts: nothing of an earlier session is left to be shown as if it were new. FunUI writes nothing
into output; it is listed because other workflows on the same ComfyUI save there, with a way to delete.

Only media, and only what can be shown: a temp directory also fills with
whatever any node felt like writing, and a browser listing 400 .pt files is not
a media bin.
"""

from __future__ import annotations

import os

#: What ComfyUI serves through /view and a browser can draw.
KINDS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"},
    "video": {".mp4", ".webm", ".mov", ".mkv"},
    "audio": {".wav", ".mp3", ".flac", ".ogg"},
}

#: A long session can leave thousands. The window shows the newest; the rest are
#: on disk and nobody scrolls to them.
MAX_FILES = 500


#: Files the app makes for itself in temp (preview cuts, small stills, kept last pictures): copies of what is listed, never listed themselves.
WORKING = "funui_"


def kind_of(name: str) -> str | None:
    ext = os.path.splitext(name)[1].lower()
    for kind, exts in KINDS.items():
        if ext in exts:
            return kind
    return None


def temp_dir() -> str | None:
    """ComfyUI's temp directory, or None when ComfyUI is not here to ask."""
    try:
        import folder_paths
        return folder_paths.get_temp_directory()
    except Exception:  # noqa: BLE001
        return None


def output_dir() -> str | None:
    """ComfyUI's output directory, or None when ComfyUI is not here to ask."""
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:  # noqa: BLE001
        return None


def base_of(where: str) -> str | None:
    """The folder `where` ("temp" or "output") names; anything else names nothing."""
    return {"temp": temp_dir, "output": output_dir}.get(where, lambda: None)()


def listing(limit: int = MAX_FILES, where: str = "temp") -> dict:
    """`{files, path, detail}`, newest first.

    `detail` carries the reason there is nothing rather than leaving an empty
    list to be read as "nothing was made" -- which is a different thing from
    "this is not a ComfyUI install" and from "the directory was wiped".
    """
    limit = max(1, min(int(limit or MAX_FILES), MAX_FILES))
    saved = where == "output"
    base = base_of(where)
    if not base:
        return {"files": [], "path": None,
                "detail": f"ComfyUI is not here to ask where its {'saved' if saved else 'temp'} files go."}
    if not os.path.isdir(base):
        return {"files": [], "path": base,
                "detail": f"The {'output' if saved else 'temp'} directory does not exist yet. It is made when "
                          "something is first written to it."}

    found: list[dict] = []
    for root, _dirs, names in os.walk(base):
        for name in names:
            kind = kind_of(name)
            if not kind or (not saved and name.startswith(WORKING)):
                continue
            full = os.path.join(root, name)
            try:
                stat = os.stat(full)
            except OSError:
                continue                    # vanished between walk and stat
            sub = os.path.relpath(root, base)
            found.append({
                "filename": name,
                "subfolder": "" if sub == "." else sub.replace(os.sep, "/"),
                "kind": kind,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })

    found.sort(key=lambda f: f.get("mtime", 0), reverse=True)
    detail = "" if found else ("FunUI saves nothing here. What other workflows save shows up here." if saved
                                else "Nothing generated yet. These files are removed when ComfyUI restarts: export what you want to keep, or save it to the media bin.")
    return {"files": found[:limit], "path": base, "detail": detail}


def delete(filename, subfolder="", where: str = "output") -> dict:
    """Remove one media file from `where`. `{deleted: True}`, or `{deleted: False, detail}` saying why not.

    The name is resolved the way /view resolves it and must stay inside that folder; and only
    what the listing would show can go, so this cannot be turned on a model, a workflow or any
    other file that happens to live next to the pictures.
    """
    base = base_of(where)
    if not base or not isinstance(filename, str) or not filename or not kind_of(filename):
        return {"deleted": False, "detail": "Name a saved picture, video or sound."}
    base = os.path.realpath(base)
    try:
        full = os.path.realpath(os.path.join(base, subfolder if isinstance(subfolder, str) else "",
                                             os.path.basename(filename)))
        inside = os.path.commonpath((full, base)) == base and full != base
    except ValueError:                                    # a NUL byte in a name, or another drive
        inside = False
    if not inside:
        return {"deleted": False, "detail": "That file is not in this folder."}
    try:
        os.remove(full)
    except FileNotFoundError:
        return {"deleted": False, "detail": "That file is already gone."}
    except OSError as err:
        return {"deleted": False, "detail": f"Could not delete it: {err.strerror or err}"}
    return {"deleted": True}
