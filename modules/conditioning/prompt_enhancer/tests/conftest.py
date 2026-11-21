import pytest


@pytest.fixture(autouse=True)
def _history_in_a_temp_folder(tmp_path, monkeypatch):
    """No test writes the chat history file of the real pack."""
    from modules.conditioning.prompt_enhancer import history
    monkeypatch.setattr(history, "FILE", tmp_path / "enhancer_history.json")
