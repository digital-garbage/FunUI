"""SA-Solver on a GPU whose CUDA runtime compiler is incomplete: the coefficient table is made on the CPU instead of failing the run."""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy and torch."""


def test_a_failed_gpu_compile_falls_back_to_the_cpu_and_other_errors_still_raise():
    import torch
    from comfy.k_diffusion import sa_solver
    from modules.sampling.sampler import nodes

    real = sa_solver.compute_exponential_coeffs
    calls = []

    def flaky(s, t, solver_order, tau_t):
        calls.append(s.device.type)
        if len(calls) == 1:
            raise RuntimeError("nvrtc: error: failed to open libnvrtc-builtins.so.13.0.")
        return real(s, t, solver_order, tau_t)
    try:
        sa_solver.compute_exponential_coeffs = flaky
        assert nodes.guard_sa_solver()
        s, t = torch.tensor(0.1), torch.tensor(0.5)
        out = sa_solver.compute_exponential_coeffs(s, t, 3, 1.0)
        assert out.shape == (3,) and calls == ["cpu", "cpu"]

        def broken(s, t, solver_order, tau_t):
            raise RuntimeError("something else broke")
        sa_solver.compute_exponential_coeffs = broken
        nodes.guard_sa_solver()
        with pytest.raises(RuntimeError, match="something else"):
            sa_solver.compute_exponential_coeffs(s, t, 3, 1.0)
    finally:
        sa_solver.compute_exponential_coeffs = real
