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


@pytest.mark.parametrize("dtype", ["float32", "float64", "bfloat16"])
def test_the_cpu_table_comes_back_in_the_dtype_and_device_of_the_inputs(dtype):
    import torch
    from comfy.k_diffusion import sa_solver
    from modules.sampling.sampler import nodes

    real, dt = sa_solver.compute_exponential_coeffs, getattr(torch, dtype)
    seen = []

    def failing_on_the_device(s, t, solver_order, tau_t):
        seen.append(s.dtype)
        if len(seen) == 1:
            raise RuntimeError("nvrtc: error: failed to open libnvrtc-builtins.so.13.0.")
        return real(s, t, solver_order, tau_t)
    try:
        sa_solver.compute_exponential_coeffs = failing_on_the_device
        nodes.guard_sa_solver()
        s, t = torch.tensor(0.1, dtype=dt), torch.tensor(0.5, dtype=dt)
        out = sa_solver.compute_exponential_coeffs(s, t, 3, 1.0)
        assert out.dtype == dt and out.device == s.device and out.shape == (3,)
        assert seen[1] in (torch.float32, torch.float64)          # the CPU worked in a full-precision type
        want = real(s.float(), t.float(), 3, 1.0)
        assert torch.allclose(out.float(), want, rtol=0.05, atol=0.05)
    finally:
        sa_solver.compute_exponential_coeffs = real
