"""Stream a video file through a per-chunk function into a new file, keeping its sound.

Frames go through ffmpeg in small chunks (never the whole clip in RAM). Shared by the post-render
steps that rewrite a finished video (upscale, interpolate), which cannot import each other.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

CHUNK = 8                # frames per call, at most
CHUNK_PIXELS = 2_000_000   # ...and at most this many input pixels per call: a big upscale of 1080p is gigabytes per frame

FFMPEG_MISSING = "ffmpeg was not found on PATH: install it to preview, export or render clips."


class ClipError(Exception):
    """A clip cannot be used, with the sentence the person should read."""


def ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise ClipError(FFMPEG_MISSING)
    return path


def comfy_dir(kind: str) -> str | None:
    """ComfyUI's output or temp folder, or None when ComfyUI is not here to ask."""
    try:
        import folder_paths
        return folder_paths.get_output_directory() if kind == "output" else folder_paths.get_temp_directory()
    except Exception:                         # noqa: BLE001 -- tests, headless
        return None


def comfy_path(filename, subfolder="", kind="output", dir_of=None) -> str | None:
    """The file ComfyUI calls (filename, subfolder, kind), or None if the name does not
    resolve to a place under that folder."""
    if not isinstance(filename, str) or not filename:
        return None
    base = (dir_of or comfy_dir)("output" if kind != "temp" else "temp")
    if not base:
        return None
    base = os.path.realpath(base)
    full = os.path.realpath(os.path.join(base, subfolder if isinstance(subfolder, str) else "",
                                         os.path.basename(filename)))
    return full if os.path.commonpath((full, base)) == base else None


def probe(path):
    """-> (width, height, fps string, has sound)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ClipError("ffprobe was not found on PATH: install ffmpeg to upscale videos.")
    out = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    streams = json.loads(out.stdout).get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise ClipError(f"no video stream in {os.path.basename(path)}")
    return (int(video["width"]), int(video["height"]), video.get("r_frame_rate") or "25",
            any(s.get("codec_type") == "audio" for s in streams))


def _read(stream, n):
    buf = bytearray()
    while len(buf) < n:
        part = stream.read(n - len(buf))
        if not part:
            break
        buf += part
    return bytes(buf)


def scaled_fps(fps, k):
    """"24/1" -> "48/1", "30000/1001" -> "60000/1001": the frame rate after every frame became k."""
    num, _, den = str(fps).partition("/")
    return f"{int(num) * k}/{den or 1}"


def stream_file(src, dst, upscale, interrupted=lambda: None, fps_factor=1):
    """Stream `src` through `upscale` ([B,H,W,3] float 0-1 -> same, larger, or more frames when `fps_factor` > 1)
    into `dst`, played at `fps_factor` times the source's frame rate so the length and the sound stay in step.
    Returns the frames written. `interrupted()` raises to stop between chunks."""
    import numpy as np
    import torch
    w, h, fps, has_sound = probe(src)
    frame = w * h * 3
    chunk = max(1, min(CHUNK, CHUNK_PIXELS // max(1, w * h)))
    exe = ffmpeg()
    dec = subprocess.Popen([exe, "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc, frames = None, 0
    try:
        while True:
            interrupted()
            raw = _read(dec.stdout, frame * chunk)
            n = len(raw) // frame
            if n == 0:
                break
            x = torch.from_numpy(np.frombuffer(raw[:n * frame], dtype=np.uint8)
                                 .reshape(n, h, w, 3).copy()).float() / 255.0
            y = (upscale(x).clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
            if enc is None:
                oh, ow = y.shape[1], y.shape[2]
                cmd = [exe, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{ow}x{oh}", "-r", scaled_fps(fps, fps_factor), "-i", "-"]
                if has_sound:
                    cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
                cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-crf", "16",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
                enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            enc.stdin.write(y.tobytes())
            frames += y.shape[0]
        if enc is None:
            raise ClipError(f"no frames could be read from {os.path.basename(src)}")
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        if enc.wait() != 0:
            raise ClipError(f"ffmpeg could not encode the upscaled video: {err[-500:]}")
        return frames
    except BaseException:
        for p in (dec, enc):
            if p is not None and p.poll() is None:
                p.kill()
            if p is not None:
                p.wait()
                for pipe in (p.stdin, p.stdout, p.stderr):
                    try:
                        if pipe:
                            pipe.close()
                    except OSError:                        # a killed encoder's unflushed pipe
                        pass
        if os.path.exists(dst):
            os.remove(dst)
        raise
    finally:
        if dec.poll() is None:
            dec.kill()


def free_name(out_dir, filename, model):
    stem = os.path.splitext(os.path.basename(filename))[0]
    tag = os.path.splitext(os.path.basename(model))[0]
    n, name = 0, f"{stem}_{tag}.mp4"
    while os.path.exists(os.path.join(out_dir, name)):
        n += 1
        name = f"{stem}_{tag}_{n}.mp4"
    return name
