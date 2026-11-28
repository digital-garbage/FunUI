import pytest
import torch


def test_the_official_anchors_are_kept_at_eight_steps_and_other_counts_follow_their_curve():
    from modules.sampling.schedule.nodes import DISTILLED_SIGMAS, distilled
    assert torch.allclose(distilled(8), torch.tensor(DISTILLED_SIGMAS))
    for n in (1, 4, 5, 12, 30):
        s = distilled(n)
        assert len(s) == n + 1 and float(s[0]) == 1.0 and float(s[-1]) == 0.0
        assert bool((s[1:] <= s[:-1]).all())


def test_ltxv_gives_a_falling_schedule_of_the_asked_length():
    from modules.sampling.schedule.nodes import ltxv
    s = ltxv(6, {"samples": torch.zeros(1, 128, 4, 8, 8)})
    assert len(s) == 7 and float(s[-1]) == 0.0 and bool((s[1:] <= s[:-1]).all())
