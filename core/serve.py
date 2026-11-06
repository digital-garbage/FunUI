"""Static serving for the app and for module JS.

The resolution is a pure function returning a `Served` value, so the allowlist and
the traversal guard are testable without aiohttp or a running ComfyUI.
"""

import hashlib
import json
import re
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet

# The editor iterates fast; a cached stale module is worse than a re-read. So code is kept by the browser but asked
# about every time ("no-cache" means: check first): an unchanged file costs a short 304 answer and no body, a changed one
# arrives whole. The page itself goes one better, see `import_map`: its files are named by their content.
REVALIDATE = {"Cache-Control": "no-cache"}
NO_STORE = REVALIDATE            # the old name, for callers that only need "not cached blindly"

# Fonts are the exception: the bytes for a given filename never change (family,
# weight range and subset are all in the name), and they are by far the largest
# thing served. Re-fetching 133 KB on every reload to stay honest about a file
# that cannot go stale is a bad trade. Replacing a face means a new filename.
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}
IMMUTABLE_SUFFIXES = frozenset({".woff2"})

# Some platforms guess text/plain for .js. With ES modules that is fatal — the
# browser refuses the module outright — so every type we care about is explicit.
EXPLICIT_TYPES = {
    ".js": "text/javascript",
    ".css": "text/css",
    ".html": "text/html",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".json": "application/json",
}


@dataclass(frozen=True)
class Served:
    status: int
    body: bytes = b""
    content_type: str = "text/plain"
    headers: dict = field(default_factory=lambda: dict(REVALIDATE))


_HASHES: dict = {}       # path -> ((mtime_ns, size), digest): a file is read again only when it changed


_IMPORT = re.compile(r'(@import\s+")(\./[^"?]+\.css)(")')


def body_of(path: Path) -> bytes:
    """The file's bytes. A stylesheet's `@import "./x.css"` lines are named by the imported file's content, so a nested
    sheet is kept like any other file and is still fetched fresh the moment it changes (its parent's name changes with it)."""
    data = path.read_bytes()
    if path.suffix.lower() != ".css" or b"@import" not in data:
        return data
    def stamp(m):
        inner = (path.parent / m[2]).resolve()
        return f"{m[1]}{m[2]}?v={digest_of(inner)}{m[3]}" if inner.is_file() else m[0]
    return _IMPORT.sub(stamp, data.decode("utf-8")).encode("utf-8")


def digest_of(path: Path) -> str:
    stat = path.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    hit = _HASHES.get(path)
    if hit and hit[0] == stamp and path.suffix.lower() != ".css":
        return hit[1]
    digest = hashlib.sha1(body_of(path)).hexdigest()[:16]       # a sheet is re-derived: an imported file may have changed under it
    _HASHES[path] = (stamp, digest)
    return digest


def headers_for(path: Path) -> dict:
    if path.suffix.lower() in IMMUTABLE_SUFFIXES:
        return dict(IMMUTABLE)
    return {**REVALIDATE, "ETag": f'"{digest_of(path)}"'}


def import_map(root: Path, url_root: str) -> dict:
    """{url: url?v=content hash} for every script under `root` (served at `url_root`). The page lists it as an import map, so
    every `import` the browser makes asks for a name that changes exactly when the file does: that name is kept for good
    (see `serve`) and a reload fetches nothing that has not changed."""
    root = root.resolve()
    out = {}
    for path in sorted(root.rglob("*.js")):
        rel = path.relative_to(root)
        if any(part in ("node_modules", "tests") for part in rel.parts) or path.name.endswith(".test.js"):
            continue
        url = f"{url_root}/{rel.as_posix()}"
        out[url] = f"{url}?v={digest_of(path)}"
    return out


def _stamped(root: Path, url_root: str, url: str) -> str:
    path = root / url[len(url_root) + 1:]
    try:
        return f"{url}?v={digest_of(path.resolve())}" if path.is_file() and root.resolve() in path.resolve().parents else url
    except OSError:
        return url


def index_page(root: Path, url_root: str, allowed: FrozenSet[str], page: str = "index.html") -> "Served":
    """The app's page with its import map in the head, and its entry script started through the map (a script tag's own
    `src` is not mapped, an `import` is)."""
    served = serve(root, page, allowed)
    if served.status != 200:
        return served
    mapping = import_map(root, url_root)
    text = served.body.decode("utf-8")
    entry = f'<script type="module" src="{url_root}/boot.js"></script>'
    if entry in text and f"{url_root}/boot.js" in mapping:
        text = text.replace(entry, f'<script type="module">import "{url_root}/boot.js";</script>')
        tag = '<script type="importmap">' + json.dumps({"imports": mapping}).replace("</", "<\\/") + "</script>"
        text = text.replace("<head>", "<head>\n  " + tag, 1)
    text = re.sub(r'(?P<a>(?:href|src)=")(?P<u>' + re.escape(url_root) + r'/[^"?#]+\.(?:css|js))(?=")',        # plain <link>/<script> tags are not mapped: name them by content here
                  lambda m: m["a"] + _stamped(root, url_root, m["u"]), text)
    body = text.encode("utf-8")
    return Served(200, body, served.content_type, {**REVALIDATE, "ETag": f'"{hashlib.sha1(body).hexdigest()[:16]}"'})


def content_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in EXPLICIT_TYPES:
        return EXPLICIT_TYPES[suffix]
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def resolve(root: Path, rel: str, allowed: FrozenSet[str]) -> Path:
    """Path for `rel` under `root`, or raise.

    ValueError  -> the request tried to leave the root (403)
    LookupError -> nothing to serve here (404)
    """
    root = root.resolve()
    cleaned = (rel or "").lstrip("/")
    if "\x00" in cleaned:
        raise ValueError("null byte in path")
    try:
        target = (root / cleaned).resolve()
    except (OSError, ValueError) as exc:  # malformed path, symlink loop
        raise ValueError("unresolvable path") from exc

    # The guard is on the RESOLVED path, so `..` and symlinks are both covered.
    if target != root and root not in target.parents:
        raise ValueError("outside root")

    if target.is_dir():
        if ".html" not in allowed:
            raise LookupError("directory")
        target = target / "index.html"

    if target.suffix.lower() not in allowed:
        raise LookupError("extension not allowed")
    if not target.is_file():
        raise LookupError("not a file")
    return target


def serve(root: Path, rel: str, allowed: FrozenSet[str], if_none_match: str | None = None, version: str | None = None) -> Served:
    """`if_none_match`: the ETag the browser holds; the same one answers 304 with no body. `version`: the `?v=` the
    request carried; it makes the file kept for a year ONLY when it is the file's present hash, so an old name can never
    pin an old file under it."""
    try:
        target = resolve(root, rel, allowed)
    except ValueError:
        return Served(403)
    except LookupError:
        return Served(404)
    headers = headers_for(target)
    if version and "ETag" in headers and version == headers["ETag"].strip('"'):
        headers = {**IMMUTABLE, "ETag": headers["ETag"]}
    elif if_none_match and "ETag" in headers and headers["ETag"] in {t.strip().removeprefix("W/") for t in if_none_match.split(",")}:
        return Served(304, b"", content_type_for(target), headers)
    return Served(200, body_of(target), content_type_for(target), headers)
