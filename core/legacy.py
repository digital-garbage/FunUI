"""Names from before the 5.0.3 rename (FunPack -> FunUI), mapped forward on load.

Saved projects, imported workflows and stored settings still carry node ids like
`FunPackSampler` and keys like `funpack.clock`. They are rewritten as they are read,
so nothing saved under the old name stops working.
"""

import re

_OLD = re.compile(r"^(FunPack|funpack)(?=$|[._A-Z])")


def name(value):
    """`value` with a leading FunPack/funpack identifier renamed; anything else unchanged."""
    if not isinstance(value, str) or " " in value:
        return value
    return _OLD.sub(lambda m: "FunUI" if m.group(1) == "FunPack" else "funui", value, count=1)


def data(obj):
    """A JSON-shaped value with every old-style key and string value renamed."""
    if isinstance(obj, dict):
        return {name(k): data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [data(v) for v in obj]
    return name(obj)
