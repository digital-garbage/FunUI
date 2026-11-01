"""Turbo sampler: the step maths on toy models, then the real wrapper through ComfyUI's guider on the tiny H3 (CPU)."""

import pytest
import torch

from modules.sampling.turbo import SETTINGS, churn, install, make, make_on_top, step_to
from modules._core import dit_hooks
from modules.sampling.two_stage.tests.test_two_stage import model  # noqa: F401 -- the module-scoped fixture


def _settings(**over):
    values = {name: spec["default"] for name, spec in SETTINGS.items() if name != "enabled"}
    values.update(over)
    return values


class Behind:
    """A guider stand-in: which stage Two stages says it is in, and the whole schedule."""
    def __init__(self, stage=None, full=None):
        setattr(self, dit_hooks.STAGE, stage)
        setattr(self, dit_hooks.FULL, full)


class Oracle:
    """A model that always knows the clean picture `target` (a flow model with a perfect prediction)."""
    def __init__(self, target, behind=None):
        self.target, self.calls, self.inner_model = target, [], behind

    def __call__(self, x, sigma, **kw):
        self.calls.append(float(sigma[0]))
        return self.target.to(x)


def test_eta_zero_is_euler():
    x, x0 = torch.randn(1, 4, 2, 3, 3), torch.randn(1, 4, 2, 3, 3)
    sigma, to = torch.tensor(0.8), torch.tensor(0.5)
    euler = x + ((x - x0) / sigma) * (to - sigma)
    assert torch.allclose(step_to(x, x0, sigma, to, 0.0, None), euler, atol=1e-5)


def test_perfect_model_lands_on_the_target_for_any_step_count_and_split():
    target = torch.randn(1, 4, 2, 3, 3)
    for sigmas in ([1.0, 0.5, 0.0], [1.0, 0.9, 0.7, 0.4, 0.2, 0.0]):
        s = torch.tensor(sigmas)
        for behind in (None, Behind(2, torch.tensor([1.0] * 4 + sigmas))):
            oracle = Oracle(target, behind)
            out = make(_settings(noise_start=0.8, polish=0.0))(oracle, torch.randn_like(target), s)
            assert torch.allclose(out, target, atol=1e-5)
            assert oracle.calls == pytest.approx([float(v) for v in sigmas[:-1]], rel=1e-6)


def test_polish_adds_one_call_in_the_last_stage_only():
    target = torch.randn(1, 4, 2, 3, 3)
    s = torch.tensor([1.0, 0.5, 0.0])
    for last, calls in ((True, 3), (False, 2)):
        oracle = Oracle(target, Behind(None if last else 1, torch.tensor([1.0, 0.5, 0.3, 0.0])))
        out = make(_settings(polish=0.05))(oracle, torch.randn_like(target), s)
        assert len(oracle.calls) == calls and torch.allclose(out, target, atol=1e-5)
        if last:
            assert oracle.calls[-1] == pytest.approx(0.05)


def test_same_seed_same_result_and_noise_actually_used():
    target = torch.randn(1, 4, 2, 3, 3)
    s = torch.tensor([1.0, 0.7, 0.4, 0.0])
    noisy_start = torch.randn_like(target)

    class Sloppy(Oracle):                              # answers differ with x, so injected noise shows in the result
        def __call__(self, x, sigma, **kw):
            return 0.5 * self.target.to(x) + 0.5 * x * (1 - sigma[0])
    run = lambda eta, seed: make(_settings(noise_start=eta, noise_end=eta, second_order=0.0))(
        Sloppy(target), noisy_start.clone(), s, extra_args={"seed": seed})
    assert torch.equal(run(0.7, 1), run(0.7, 1))
    assert not torch.allclose(run(0.7, 1), run(0.7, 2))
    assert not torch.allclose(run(0.7, 1), run(0.0, 1))


def test_follows_the_input_dtype_and_nested_sound_gets_no_noise():
    from comfy.nested_tensor import NestedTensor
    video, audio = torch.randn(1, 4, 2, 3, 3, dtype=torch.float64), torch.randn(1, 6, 2, 3, dtype=torch.float64)
    seen = []

    class Sound:
        def __call__(self, x, sigma, **kw):
            seen.append(x.unbind()[1].clone())
            return NestedTensor((torch.zeros_like(x.unbind()[0]), torch.zeros_like(x.unbind()[1])))
    s = torch.tensor([1.0, 0.6, 0.3, 0.0], dtype=torch.float32)       # sigmas in another dtype than the latent
    out = make(_settings(noise_start=1.0, polish=0.1, second_order=1.0))(
        Sound(), NestedTensor((video, audio)), s)
    v, a = out.unbind()
    assert v.dtype == torch.float64 and a.dtype == torch.float64
    # the sound is stepped by plain Euler toward zero: x * (to / sigma) each time
    assert torch.allclose(seen[1], audio * (0.6 / 1.0), atol=1e-9)


def _guided(model, values, sigmas, with_two_stages=None, sampler="euler"):
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor
    patched = model.clone()
    if with_two_stages:
        from modules.sampling.two_stage import install as two
        two(patched, with_two_stages, "funui.two_stage")
    note = install(patched, values, "funui.turbo")
    calls = []
    hook = patched.model.diffusion_model.register_forward_pre_hook(
        lambda m, a, kw: calls.append(float(kw["transformer_options"]["sigmas"].reshape(-1)[0])), with_kwargs=True)
    guider = comfy.samplers.CFGGuider(patched)
    cond = [[torch.randn(1, 5, 48), {}]]
    guider.set_conds(cond, cond)
    guider.set_cfg(1.0)
    latent = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 3)))
    noise = NestedTensor((torch.randn(1, 24, 2, 8, 8), torch.randn(1, 32, 2, 3)))
    try:
        out = guider.sample(noise, latent, comfy.samplers.sampler_object(sampler), torch.tensor(sigmas),
                            seed=1, disable_pbar=True)
    finally:
        hook.remove()
    return note, out, calls


def test_off_installs_nothing(model):
    assert install(model.clone(), {"enabled": False}, "k") is None


@pytest.mark.parametrize("sigmas,cut", [((1.0, 0.8, 0.5, 0.25, 0.0), 2), ((1.0, 0.9, 0.7, 0.55, 0.4, 0.2, 0.0), 3),
                                        ((1.0, 0.6, 0.3, 0.0), 1)])
def test_runs_through_two_stages_with_any_split(model, sigmas, cut):
    values = {"enabled": True, **_settings(polish=0.05)}
    note, out, calls = _guided(model, values, sigmas, {"enabled": True, "size": 0.5, "steps": cut})
    assert "fresh noise" in note
    v, a = out.unbind()
    assert torch.isfinite(v).all() and torch.isfinite(a).all()
    assert len(calls) == len(sigmas) - 1 + 1                 # one call per step plus the polish call
    assert calls[cut - 1] != calls[cut] and calls[-1] == pytest.approx(0.05)
    assert calls[cut] == pytest.approx(sigmas[cut], rel=1e-4)


def test_without_two_stages_and_without_polish(model):
    values = {"enabled": True, **_settings(polish=0.0)}
    _, out, calls = _guided(model, values, (1.0, 0.7, 0.3, 0.0))
    assert len(calls) == 3 and torch.isfinite(out.unbind()[0]).all()


def test_registered_in_comfys_sampler_list_and_runs_like_the_module(model):
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor
    assert "funui_turbo" in comfy.samplers.SAMPLER_NAMES
    patched = model.clone()
    guider = comfy.samplers.CFGGuider(patched)
    cond = [[torch.randn(1, 5, 48), {}]]
    guider.set_conds(cond, cond)
    guider.set_cfg(1.0)
    sigmas = comfy.samplers.calculate_sigmas(patched.get_model_object("model_sampling"), "sgm_uniform", 4)
    noise = NestedTensor((torch.randn(1, 24, 2, 8, 8), torch.randn(1, 32, 2, 3)))
    latent = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 3)))
    out = guider.sample(noise, latent, comfy.samplers.sampler_object("funui_turbo"), sigmas, seed=1, disable_pbar=True)
    assert torch.isfinite(out.unbind()[0]).all()


def test_polish_is_on_by_default_at_a_quarter(model):
    assert SETTINGS["polish"]["default"] == 0.25
    _, _, calls = _guided(model, {"enabled": True, **_settings()}, (1.0, 0.7, 0.3, 0.0))
    assert len(calls) == 4 and calls[-1] == pytest.approx(0.25)


@pytest.mark.parametrize("sampler", ["euler", "er_sde", "dpmpp_2m", "res_multistep", "dflow_ord1", "uni_pc", "ddim"])
def test_the_switch_takes_over_whatever_sampler_is_chosen(model, sampler):
    """The same rule runs whichever sampler was picked: one model call per step plus the polish call, and the
    last call is the polish sigma (a sampler's own loop would make a different number of calls)."""
    import modules.sampling.distilled_flow  # noqa: F401 -- registers the dflow samplers
    values = {"enabled": True, **_settings(polish=0.05)}
    note, out, calls = _guided(model, values, (1.0, 0.7, 0.3, 0.0), sampler=sampler)
    assert "fresh noise" in note and len(calls) == 4 and calls[-1] == pytest.approx(0.05)
    assert torch.isfinite(out.unbind()[0]).all()


def test_churn_with_no_fresh_noise_changes_nothing_and_follows_dtype_and_device():
    x, d = torch.randn(1, 3, 2, 4, 4, dtype=torch.float64), torch.randn(1, 3, 2, 4, 4, dtype=torch.float64)
    n = torch.randn_like(x)
    sigma = torch.tensor([0.6], dtype=torch.float32)                  # a different dtype from the picture
    assert torch.equal(churn(x, d, sigma, 0.0, n), d)
    out = churn(x, d, sigma, 0.5, n)
    assert out.dtype == torch.float64 and out.device == x.device and not torch.equal(out, d)


def test_an_euler_step_from_the_churned_answer_lands_on_the_turbo_step():
    x, d, n = torch.randn(1, 3, 2, 4, 4), torch.randn(1, 3, 2, 4, 4), torch.randn(1, 3, 2, 4, 4)
    sigma, to, eta = torch.tensor(0.7), torch.tensor(0.4), 0.45
    answer = churn(x, d, sigma, eta, n)
    euler = x + ((x - answer) / sigma) * (to - sigma)                 # what Euler does with an answer
    assert torch.allclose(euler, step_to(x, d, sigma, to, eta, n), atol=1e-5)


def _euler(model, x, sigmas, extra_args=None, callback=None, disable=None, **_):
    ones = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        d = (x - model(x, sigmas[i] * ones, **(extra_args or {}))) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
    return x


def test_on_top_of_a_sampler_with_no_fresh_noise_and_no_polish_is_that_sampler():
    target, sigmas = torch.randn(1, 3, 2, 4, 4), torch.tensor([1.0, 0.7, 0.3, 0.0])
    noise = torch.randn_like(target)
    plain = _euler(Oracle(target), noise.clone(), sigmas)
    top = make_on_top(_euler, _settings(noise_start=0.0, noise_end=0.0, polish=0.0))(Oracle(target), noise.clone(), sigmas)
    assert torch.equal(plain, top)


def test_on_top_of_a_sampler_adds_fresh_noise_before_the_last_step_and_one_polish_call():
    target, sigmas = torch.randn(1, 3, 2, 4, 4), torch.tensor([1.0, 0.7, 0.3, 0.0])
    noise = torch.randn_like(target)
    oracle = Oracle(target)
    run = make_on_top(_euler, _settings(noise_start=0.5, noise_end=0.5, polish=0.1))
    a = run(oracle, noise.clone(), sigmas, extra_args={"seed": 3})
    b = run(Oracle(target), noise.clone(), sigmas, extra_args={"seed": 3})
    assert torch.equal(a, b)                                          # the same seed gives the same picture
    assert oracle.calls == pytest.approx([1.0, 0.7, 0.3, 0.1])         # one call per step, then the polish call
    assert not torch.equal(a, _euler(Oracle(target), noise.clone(), sigmas))
