"""Combination rule (Dex): a feature is tested alone AND with the others, in every pair and all together, on the real
MiniMaxH3 class (tiny, CPU) through ComfyUI's own guider, with and without Two stages. A new feature goes into FEATURES.

Three things keep it honest:
  * the registry guard: every modifier a module provides must be in FEATURES (run here), in LTX_ONLY (run in
    test_combinations_ltx.py) or in EXEMPT with the reason it cannot run on a Mac. A new module fails until it is listed;
  * a feature counts as installed only when its own status line says it is active (not "needs ...", "failed ...");
  * a feature alone must change what the model is wired with (wrappers, hooks, options), and the ones that act without
    ratings must change the picture.
The runs cross the features with the sampler (euler and the SDE ones that use the Brownian noise) and Two stages on/off.
CFG stays 1 and Two stages keeps its default resize: Dex's decision, not a gap.

Not covered here and said so: triples (the all-on case catches what only shows with many on), and anything that needs a
GPU or real weights."""

import itertools

import pytest
import torch

from modules.sampling.two_stage.tests.test_two_stage import model  # noqa: F401 -- the module-scoped tiny H3

SIGMAS = (1.0, 0.9, 0.7, 0.3, 0.0)

FEATURES = {
    "reins": {"blocks": "2"},
    "attention_bandit": {},
    "q_steer": {"blocks": "2"},
    "attention_temperature": {"temperature": 1.1},
    "stas": {"block": 1},
    "noise_merge": {"seed": 5},
    "sharpness": {},
    "run_meter": {},
    "late_branch": {"block": 2},
    "decisiveness": {},
    "shot_memory": {},
    "turbo_sampler": {},
    "camera_move": {"pan_x": 0.3, "zoom": 1.2},
    "explore_first_step": {},
    "alg": {},
    "block_influence": {},
    "phrase_probe": {},
    "block_repeat": {"blocks": "1-2"},
    "dynashift": {},
    "momentum": {},
    "output_guidance": {},
    "prompt_nudge": {},
    "score_slider": {},
    "shadow_negative": {},
    "trajectory_guidance": {},
    "velocity_bias": {},
    "video_detail": {},
}
#: Modifiers that need the LTX audio+video or spatial-latent model: run in test_combinations_ltx.py.
LTX_ONLY = {"audio_coupling", "context_windows", "identity_bank", "sharpen", "temporal_style"}
#: Modifiers that cannot run here, and why. Said loudly: a feature in this list is not proven on this machine.
EXEMPT = {
    "ptd": "needs a PTD adapter file in ComfyUI/models/ptd (none on this Mac)",
    "taste": "always present: it is the rating store every other feature reads",
    "two_stage": "a dimension of every run here, not a feature in the subsets",
    "block_map": "follows a saved map, there is no switch to put in a subset: every run here has a map saved, test_block_map_is_wired_only_when_a_map_is_saved and the bare run below cover no map",
}
SUBSETS = [()] + [(f,) for f in FEATURES] + list(itertools.combinations(FEATURES, 2)) + [tuple(FEATURES)]


def _name(subset):
    return "+".join(subset) or "plain"


def _save_a_map(tmp_path, monkeypatch, n):
    """An uneven saved map for the tiny model, so the Block map really reshapes the features that follow it."""
    from modules.sampling.block_map import measure
    monkeypatch.setattr(measure, "FOLDER", tmp_path / "maps")
    monkeypatch.setattr(measure, "SWITCH", tmp_path / "block_map.enabled")
    grid = measure.groups(n)
    res = {"when": "test", "blocks": n, "groups": grid, "points": list(measure.POINTS), "size": 1, "delta": measure.DELTA}
    for kind in measure.KINDS:
        res[kind] = {"leverage": [[0.5 + g + 0.3 * p for p in range(len(measure.POINTS))] for g in range(len(grid))],
                     "fragility": [[0.1 * (g + 1) for p in range(len(measure.POINTS))] for g in range(len(grid))]}
    measure.save(res)


def _stage_lora(model, stage):
    """The model with a LoRA kept for one stage, the way the LoRA loader leaves it."""
    from core import dit_hooks
    from modules.loaders.lora.keys import match
    gen = torch.Generator().manual_seed(4)
    lora = {"transformer.transformer_blocks.0.ff.net.2.lora_A.weight": torch.randn(4, 64, generator=gen),
            "transformer.transformer_blocks.0.ff.net.2.lora_B.weight": torch.randn(64, 4, generator=gen)}
    patches, _how, _dropped = match(model, None, lora)
    kept = model.clone()
    kept.set_attachments(dit_hooks.STAGE_PATCHES, ((stage, patches, 1.0),))
    return kept


STAGES = {                                   # one stage, and Two stages with its default resize (CFG is 1: H3 and LTX run at 1)
    "one-stage": None,
    "two-stages": {"enabled": True, "size": 0.5, "steps": 2, "back_up": 0.5},
}
SAMPLERS = ["euler", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_sde"]      # the SDE ones draw Brownian noise


def _status_line(status, name):
    for line in str(status).splitlines():
        if line.strip().startswith(name + ":"):
            return line.strip()
    return None


NOT_ACTIVE = ("needs ", "failed", "inactive", "unavailable", "switched off")


def _fingerprint(patched):
    """What the model is wired with: enough to tell a patcher that a feature touched from one it did not."""
    to = patched.model_options.get("transformer_options", {})
    return (sorted(map(str, patched.wrappers)), sorted(str(k) for k, v in patched.wrappers.items() if v),
            sorted(sum(len(w) for w in v.values()) for v in patched.wrappers.values()),
            sorted(map(str, patched.callbacks)), sorted(str(k) for k in to), sorted(map(str, patched.model_options)),
            sorted(map(str, getattr(patched, "attachments", {}))), sorted(map(str, patched.object_patches)))


def _load(model, subset, stage_settings, lora_stage=None):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    settings = {"taste": {"key": "fox"}}
    for f in subset:
        settings[f] = {"enabled": True, **FEATURES[f]}
    if stage_settings:
        settings["two_stage"] = dict(stage_settings)
    patched, status = FunUILoadModifiers.execute(_stage_lora(model, lora_stage) if lora_stage else model.clone(), settings).result
    for f in subset:
        line = _status_line(status, f)
        assert line, f"never installed (a test that installs nothing proves nothing): {f}: {status}"
        assert not any(bad in line.lower() for bad in NOT_ACTIVE), f"{f} did not switch on: {line}"
    return patched, status


def _run(model, subset, staged, stage=None, sampler="euler", cfg=1.0, variant=None):
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor
    stage_settings = variant if variant is not None else (STAGES["two-stages"] if staged else None)
    patched, _status = _load(model, subset, stage_settings, stage)
    guider = comfy.samplers.CFGGuider(patched)
    cond = [[torch.randn(1, 5, 48), {}]]
    guider.set_conds(cond, [[torch.randn(1, 5, 48), {}]] if cfg != 1.0 else cond)
    guider.set_cfg(cfg)
    latent = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 3)))
    noise = NestedTensor((torch.randn(1, 24, 2, 8, 8), torch.randn(1, 32, 2, 3)))
    return guider.sample(noise, latent, comfy.samplers.sampler_object(sampler), torch.tensor(SIGMAS),
                         seed=1, disable_pbar=True)


def _check(out, label):
    video, audio = out.unbind()
    assert torch.isfinite(video).all() and torch.isfinite(audio).all(), label
    assert video.shape == (1, 24, 2, 8, 8) and audio.shape == (1, 32, 2, 3), label
    assert video.dtype == audio.dtype == torch.float32 and video.device.type == "cpu", label       # the input's dtype and device come back


@pytest.fixture
def fresh(model, tmp_path, monkeypatch):
    from core import dit_hooks
    from modules.sampling.block_influence import measure as influence
    from modules.sampling.phrase_probe import measure as probe
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    monkeypatch.setattr(influence, "SWITCH", tmp_path / "block_influence.enabled")       # the two recording tools act only while recording is on
    monkeypatch.setattr(probe, "SWITCH", tmp_path / "phrase_probe.enabled")
    influence.set_enabled(True)
    probe.set_enabled(True)
    _save_a_map(tmp_path, monkeypatch, dit_hooks.block_count(model))
    return model


def test_every_modifier_a_module_provides_is_in_the_matrix_or_exempt_with_a_reason():
    import importlib
    import pkgutil

    import modules
    found = set()
    for info in pkgutil.walk_packages(modules.__path__, "modules."):
        if not info.ispkg or info.name.rsplit(".", 1)[-1] in ("tests", "_core"):
            continue
        try:
            mod = importlib.import_module(info.name)
        except Exception:                                   # noqa: BLE001 -- a missing dependency is not this test's business
            continue
        if hasattr(mod, "install") and hasattr(mod, "ID") and getattr(mod, "PROVIDES", {}).get("modifier") is mod.install:
            found.add(mod.ID)
    listed = set(FEATURES) | LTX_ONLY | set(EXEMPT)
    assert found <= listed, f"add these to FEATURES, LTX_ONLY or EXEMPT (with the reason): {sorted(found - listed)}"
    assert listed - found <= set(), f"listed but no module provides it: {sorted(listed - found)}"
    assert all(EXEMPT[k] for k in EXEMPT)


RECORDING_TOOLS = ("block_influence", "phrase_probe")        # they hook the model only while recording is switched on
#: Installed and active, but they record during the run and leave nothing to see on the patcher (their own tests run them).
NOTHING_TO_SEE_BEFORE_THE_RUN = ("block_influence",)


@pytest.mark.parametrize("feature", [f for f in FEATURES if f not in NOTHING_TO_SEE_BEFORE_THE_RUN])
def test_a_feature_alone_is_really_wired_in(fresh, feature):
    from modules.sampling.block_influence import measure as influence
    from modules.sampling.phrase_probe import measure as probe
    influence.set_enabled(False)                         # the baseline has no recording tool hooked in
    probe.set_enabled(False)
    base, _ = _load(fresh, (), None)
    if feature in RECORDING_TOOLS:
        {"block_influence": influence, "phrase_probe": probe}[feature].set_enabled(True)
    patched, status = _load(fresh, (feature,), None) if feature not in RECORDING_TOOLS else _load_recording(fresh, feature)
    assert _fingerprint(patched) != _fingerprint(base), f"{feature} switched on and changed nothing the model is wired with: {_status_line(status, feature)}"


def _load_recording(model, feature):
    """The recording tools say 'off' unless recording is on, which is the point: they are loaded with it on."""
    return _load(model, (feature,), None)


#: Features that act on this tiny model without ratings: they must change the picture, not just install.
ACTS = ("momentum", "video_detail", "noise_merge", "sharpness", "late_branch", "decisiveness", "turbo_sampler", "camera_move")


@pytest.mark.parametrize("feature", ACTS)
def test_a_feature_that_needs_no_ratings_changes_the_picture(fresh, feature):
    torch.manual_seed(0)
    base = _run(fresh, (), False)
    torch.manual_seed(0)
    out = _run(fresh, (feature,), False)
    assert any(not torch.equal(a, b) for a, b in zip(out.unbind(), base.unbind())), f"{feature} ran and changed nothing"


@pytest.mark.parametrize("staged", [False, True], ids=["one-stage", "two-stages"])
@pytest.mark.parametrize("subset", SUBSETS, ids=_name)
def test_every_combination_finishes_with_a_finite_picture_and_sound(fresh, subset, staged):
    _check(_run(fresh, subset, staged), _name(subset))


ALONE_AND_ALL = [()] + [(f,) for f in FEATURES] + [tuple(FEATURES)]


@pytest.mark.parametrize("variant", STAGES, ids=list(STAGES))
@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("subset", [(), tuple(FEATURES)] + [(f,) for f in ("reins", "stas", "noise_merge", "turbo_sampler", "camera_move")], ids=_name)
def test_features_cross_the_sampler_and_the_two_stages_setting(fresh, subset, sampler, variant):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Should have")             # torchsde's own warning about the Brownian start
        out = _run(fresh, subset, False, sampler=sampler, variant=STAGES[variant])
    _check(out, f"{_name(subset)} {sampler} {variant}")


@pytest.mark.parametrize("stage", ["First", "Second", "Both"])
@pytest.mark.parametrize("subset", [(), tuple(FEATURES)], ids=_name)
def test_a_lora_kept_for_one_stage_runs_with_every_feature(fresh, subset, stage):
    _check(_run(fresh, subset, True, stage), (_name(subset), stage))


def test_block_map_is_wired_only_when_a_map_is_saved(model, tmp_path, monkeypatch):
    from core import dit_hooks
    from modules.sampling.block_map import measure
    monkeypatch.setattr(measure, "FOLDER", tmp_path / "maps")
    monkeypatch.setattr(measure, "SWITCH", tmp_path / "block_map.enabled")
    without, _ = _load(model, (), None)
    _save_a_map(tmp_path, monkeypatch, dit_hooks.block_count(model))
    with_map, status = _load(model, (), None)
    assert with_map.model_options["transformer_options"].get("funui_block_map") is not None
    assert without.model_options["transformer_options"].get("funui_block_map") is None, status


@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("subset", [(), tuple(FEATURES)], ids=_name)
def test_no_saved_map_is_a_plain_run_with_every_feature(model, tmp_path, monkeypatch, subset, sampler):
    from modules.sampling.block_influence import measure as influence
    from modules.sampling.block_map import measure
    from modules.sampling.phrase_probe import measure as probe
    from modules.system.taste import store
    monkeypatch.setattr(measure, "FOLDER", tmp_path / "maps")
    monkeypatch.setattr(measure, "SWITCH", tmp_path / "block_map.enabled")
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    monkeypatch.setattr(influence, "SWITCH", tmp_path / "block_influence.enabled")
    monkeypatch.setattr(probe, "SWITCH", tmp_path / "phrase_probe.enabled")
    influence.set_enabled(True)
    probe.set_enabled(True)
    _check(_run(model, subset, True, sampler=sampler), (_name(subset), sampler))


# --- LoRA slots: the loader node itself, up to three slots, each for the first stage, the second or both --------------------

def _lora_file(block, seed):
    gen = torch.Generator().manual_seed(seed)
    return {f"transformer.transformer_blocks.{block}.ff.net.2.lora_A.weight": torch.randn(4, 64, generator=gen),
            f"transformer.transformer_blocks.{block}.ff.net.2.lora_B.weight": torch.randn(64, 4, generator=gen)}


SLOT_SETS = {
    "one-both": ("Both",),
    "first-and-second": ("First", "Second"),
    "all-three": ("Both", "First", "Second"),
    "both-twice": ("Both", "Both"),
}


def _loaded(model, stages, monkeypatch):
    from modules.loaders.lora import nodes
    files = {f"l{i}.safetensors": _lora_file(i % 2, i) for i in range(len(stages))}
    monkeypatch.setattr(nodes.folder_paths, "get_full_path_or_raise", lambda kind, name: name)
    monkeypatch.setattr(nodes.comfy.utils, "load_torch_file", lambda path, safe_load=True: dict(files[path]))
    kw = {}
    for i, stage in enumerate(stages, 1):
        sfx = "" if i == 1 else f"_{i}"
        kw[f"lora_name{sfx}"], kw[f"strength_model{sfx}"], kw[f"apply_to_stage{sfx}"] = f"l{i - 1}.safetensors", 1.0, stage
    out = nodes.FunUILoraLoader.execute(model=model.clone(), **kw)
    patched, _clip, status = out[0], out[1], out[2]
    assert "matched nothing" not in status and "not applied" not in status, status
    return patched


@pytest.mark.parametrize("slots", SLOT_SETS, ids=list(SLOT_SETS))
@pytest.mark.parametrize("subset", [(), tuple(FEATURES)], ids=_name)
def test_lora_slots_with_stages_and_features_run_and_change_the_picture(fresh, monkeypatch, slots, subset):
    stages = SLOT_SETS[slots]
    torch.manual_seed(0)
    plain = _run(fresh, subset, True)
    with_loras = _loaded(fresh, stages, monkeypatch)
    torch.manual_seed(0)
    out = _run(with_loras, subset, True)
    _check(out, (slots, _name(subset)))
    assert any(not torch.equal(a, b) for a, b in zip(out.unbind(), plain.unbind())), f"the LoRAs ({slots}) changed nothing"
