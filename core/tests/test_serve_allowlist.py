"""Rule 1, transport half: only the app is served; modules/ never is."""

import pytest

from core import config
from core.serve import content_type_for, serve


@pytest.fixture()
def tree(tmp_path):
    """A miniature app/ + modules/ tree with the files that matter."""
    app = tmp_path / "app"
    (app / "composer").mkdir(parents=True)
    (app / "index.html").write_text("<p>shell</p>")
    (app / "boot.js").write_text("export const ok = 1;")
    (app / "composer" / "composer.css").write_text(":root{}")
    (app / "assets").mkdir()
    (app / "assets" / "Inter.woff2").write_bytes(b"\x00woff2")

    mods = tmp_path / "modules"
    (mods / "timing" / "audio_clock").mkdir(parents=True)
    (mods / "timing" / "audio_clock" / "sneaky.css").write_text(".x{color:red}")
    (mods / "timing" / "audio_clock" / "panel.html").write_text("<b>no</b>")
    (mods / "timing" / "audio_clock" / "module.py").write_text("x = 1")

    (tmp_path / "secret.txt").write_text("not yours")
    return tmp_path


def app_get(tree, rel):
    return serve(tree / "app", rel, config.APP_EXTS)


# --- the allowlist ---------------------------------------------------------

def test_app_serves_its_own_css_and_fonts(tree):
    assert app_get(tree, "composer/composer.css").status == 200
    assert app_get(tree, "assets/Inter.woff2").status == 200


def test_app_root_serves_index(tree):
    r = app_get(tree, "")
    assert r.status == 200 and b"shell" in r.body


# --- the traversal guard ---------------------------------------------------

@pytest.mark.parametrize("rel", [
    "../secret.txt",
    "../../etc/passwd",
    "composer/../../secret.txt",
    "/../secret.txt",
])
def test_traversal_is_forbidden(tree, rel):
    assert app_get(tree, rel).status == 403


def test_null_byte_is_forbidden(tree):
    assert app_get(tree, "boot.js\x00.png").status == 403


def test_symlink_escape_is_forbidden(tree):
    link = tree / "app" / "escape.js"
    link.symlink_to(tree / "secret.txt")
    assert app_get(tree, "escape.js").status == 403


# --- content types ---------------------------------------------------------

def test_js_is_explicitly_text_javascript(tree):
    # With ES modules a text/plain guess makes the browser refuse the module.
    assert app_get(tree, "boot.js").content_type == "text/javascript"


def test_known_types_are_explicit():
    from pathlib import Path
    assert content_type_for(Path("a.css")) == "text/css"
    assert content_type_for(Path("a.woff2")) == "font/woff2"
    assert content_type_for(Path("a.svg")) == "image/svg+xml"


def test_code_is_never_cached_blindly(tree):
    assert app_get(tree, "boot.js").headers["Cache-Control"] == "no-cache"


# --- missing files ---------------------------------------------------------

def test_missing_file_is_404(tree):
    assert app_get(tree, "nope.js").status == 404


# --- keeping modules between reloads -----------------------------------------

def test_an_unchanged_file_answers_304_with_no_body_and_a_changed_one_answers_whole(tree):
    first = app_get(tree, "boot.js")
    again = serve(tree / "app", "boot.js", config.APP_EXTS, if_none_match=first.headers["ETag"])
    assert again.status == 304 and again.body == b""
    (tree / "app" / "boot.js").write_text("export const ok = 2; // changed")
    changed = serve(tree / "app", "boot.js", config.APP_EXTS, if_none_match=first.headers["ETag"])
    assert changed.status == 200 and b"changed" in changed.body and changed.headers["ETag"] != first.headers["ETag"]


def test_a_file_is_kept_for_a_year_only_under_its_present_hash(tree):
    now = app_get(tree, "boot.js").headers["ETag"].strip('"')
    assert "immutable" in serve(tree / "app", "boot.js", config.APP_EXTS, version=now).headers["Cache-Control"]
    old = serve(tree / "app", "boot.js", config.APP_EXTS, version="0123456789abcdef")           # a name from before an edit
    assert old.status == 200 and "immutable" not in old.headers["Cache-Control"], "an old name must never pin a file"
    assert "immutable" not in serve(tree / "app", "boot.js", config.APP_EXTS, version="").headers["Cache-Control"]


def test_the_import_map_names_every_script_by_its_content_and_skips_tests(tree):
    from core.serve import import_map
    app = tree / "app"
    (app / "tests").mkdir()
    (app / "tests" / "a.js").write_text("x")
    (app / "b.test.js").write_text("x")
    one = import_map(app, "/fun/app/ui")
    assert set(one) == {"/fun/app/ui/boot.js"} and one["/fun/app/ui/boot.js"].startswith("/fun/app/ui/boot.js?v=")
    (app / "boot.js").write_text("export const ok = 3;")
    assert import_map(app, "/fun/app/ui") != one


def test_the_page_starts_the_app_through_the_import_map(tree):
    from core.serve import index_page
    app = tree / "app"
    (app / "index.html").write_text('<html><head><title>x</title></head><body><script type="module" src="/fun/app/ui/boot.js"></script></body></html>')
    page = index_page(app, "/fun/app/ui", config.APP_EXTS)
    text = page.body.decode()
    assert page.status == 200 and '<script type="importmap">' in text and text.index("importmap") < text.index('import "/fun/app/ui/boot.js"')
    assert 'src="/fun/app/ui/boot.js"' not in text and page.headers["Cache-Control"] == "no-cache" and page.headers["ETag"]
    (app / "index.html").write_text("<p>no entry script</p>")
    assert b"importmap" not in index_page(app, "/fun/app/ui", config.APP_EXTS).body        # a page it does not know is left as it is


def test_stylesheets_and_plain_scripts_in_the_page_are_named_by_content_and_follow_an_edit(tree):
    from core.serve import index_page
    app = tree / "app"
    (app / "a.css").write_text("a{color:red}")
    (app / "index.html").write_text('<html><head><link rel="stylesheet" href="/fun/app/ui/a.css"><link href="/fun/app/ui/missing.css"></head><body></body></html>')
    one = index_page(app, "/fun/app/ui", config.APP_EXTS)
    assert b'href="/fun/app/ui/a.css?v=' in one.body and b'href="/fun/app/ui/missing.css"' in one.body
    (app / "a.css").write_text("a{color:blue}")                       # an edit: the very next page names the new file
    two = index_page(app, "/fun/app/ui", config.APP_EXTS)
    assert two.body != one.body and two.headers["ETag"] != one.headers["ETag"]


def test_a_stylesheet_names_what_it_imports_by_content_and_an_edit_inside_it_changes_the_outer_name(tree):
    app = tree / "app"
    (app / "inner.css").write_text("b{color:red}")
    (app / "outer.css").write_text('@import "./inner.css";\na{color:blue}')
    one = serve(app, "outer.css", config.APP_EXTS)
    assert b'@import "./inner.css?v=' in one.body
    (app / "inner.css").write_text("b{color:green}")
    two = serve(app, "outer.css", config.APP_EXTS, if_none_match=one.headers["ETag"])
    assert two.status == 200 and two.headers["ETag"] != one.headers["ETag"], "an edit inside an import must not be answered 304"
