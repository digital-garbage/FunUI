"""A module with an `install(patcher, values, key)` that never lists it in PROVIDES is never installed (the
Load Modifiers node only calls what is provided), and its own tests still pass because they call it directly."""

import importlib
import pkgutil

import modules


def test_every_modifier_install_is_provided():
    missing = []
    for info in pkgutil.walk_packages(modules.__path__, "modules."):
        if not info.ispkg or info.name.rsplit(".", 1)[-1] in ("tests", "_core"):
            continue
        try:
            mod = importlib.import_module(info.name)
        except Exception:                               # noqa: BLE001 -- a missing dependency is not this test's business
            continue
        fn = getattr(mod, "install", None)
        if fn is not None and getattr(mod, "__file__", "").endswith("__init__.py") and hasattr(mod, "ID"):
            if getattr(mod, "PROVIDES", {}).get("modifier") is not fn:
                missing.append(info.name)
    assert not missing, f"install() not provided as a modifier: {missing}"
