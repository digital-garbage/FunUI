import types

import pytest
import torch


class _Stats:
    def get_buffer(self, name):
        return torch.full((4,), 2.0) if name == "std-of-means" else torch.full((4,), 0.5)


class _VideoVAE:                                      # the name is how the loader tells an LTX video VAE from an audio one
    per_channel_statistics = _Stats()


class _Fake:
    """A stand-in for ComfyUI's patcher around Lightricks' LatentUpsampler: doubles height and width."""
    load_device = torch.device("cpu")

    class model:                                       # noqa: N801
        pass

    def model_dtype(self):
        return torch.float32


@pytest.fixture
def stats():
    from core import latent_stats
    latent_stats._LAST.clear()
    latent_stats.remember(_VideoVAE())
    yield latent_stats
    latent_stats._LAST.clear()


def test_the_tables_normalise_and_un_normalise_each_other(stats):
    x = torch.randn(1, 4, 2, 3, 3)
    assert stats.available() and torch.allclose(stats.normalize(stats.un_normalize(x)), x, atol=1e-6)


def test_an_audio_vae_leaves_the_tables_alone(stats):
    class AudioVAE:
        per_channel_statistics = _Stats()
    stats._LAST.clear()
    stats.remember(AudioVAE())
    assert not stats.available()


def test_the_ltx_upscaler_runs_between_un_normalise_and_normalise_and_the_hand_over_is_trimmed(stats, comfyui, monkeypatch):
    from modules.sampling.two_stage import stages
    up = _Fake()
    seen = {}

    def double(latent):
        seen["in"] = latent.clone()
        return latent.repeat_interleave(2, -1).repeat_interleave(2, -2)
    up.model = types.SimpleNamespace(__call__=None)
    up.model = type("LatentUpsampler", (), {"__call__": staticmethod(double)})()
    import comfy.model_management as mm
    monkeypatch.setattr(mm, "load_models_gpu", lambda *a, **k: None)
    video = torch.randn(1, 4, 2, 4, 4)
    out = stages.enlarge(video, 6, 6, upscaler=up)
    assert torch.allclose(seen["in"], stats.un_normalize(video)) and out.shape == (1, 4, 2, 6, 6)
    stats._LAST.clear()                                # no tables: the run goes on with the plain resize
    assert stages.enlarge(video, 6, 6, upscaler=up).shape == (1, 4, 2, 6, 6)
