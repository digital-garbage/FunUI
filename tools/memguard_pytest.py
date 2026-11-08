"""pytest plugin for memguard: records the running test so the guard can report it when it kills."""
import os


def pytest_runtest_logstart(nodeid, location):
    path = os.environ.get("MEMGUARD_STATE")
    if path:
        with open(path, "w") as f:
            f.write(nodeid)
