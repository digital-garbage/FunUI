"""Download one URL to one file, safely. The only downloader FunUI has: taggers and the Veda predictor use it.

Streams to a `.part` file and renames only when the file is complete, so a cut connection or a wrong file
never leaves anything that looks finished. A size or a SHA-256 that is known in advance is checked before the
rename; any failure removes the `.part` and raises OSError with a sentence for the person.
"""
import hashlib
import urllib.request


def to_file(url, dest, progress=lambda done, total: None, size=None, sha256=None, timeout=60):
    """Fetch `url` into `dest` (a pathlib.Path). `progress(done, total)` is called per megabyte; total is 0 when
    the server does not say. `size` / `sha256`: what the finished file must be."""
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "FunUI"}), timeout=timeout) as res, open(part, "wb") as out:
            total = int(res.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = res.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                progress(done, total)
        if total and done != total:
            raise OSError(f"the download stopped at {done} of {total} bytes")
        if size is not None and done != size:
            raise OSError(f"the file is {done} bytes, expected {size}: it was not kept")
        if sha256 is not None and digest.hexdigest() != sha256:
            raise OSError("the file does not match its published checksum: it was not kept")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
