"""Шум усилителя после ФД: sigma^2 = (F-1) 4kT (fs/2) / RL."""
import numpy as np
import pytest

from mwp_converter.noise import K_B, amp_current_noise


@pytest.mark.parametrize("nf", [1.0, 3.0, 6.0])
def test_variance(nf):
    fs, RL, Tc = 396e9, 50.0, 25.0
    n = amp_current_noise(2 ** 20, fs, RL, nf, Tc, 5)
    want = (10 ** (nf / 10) - 1) * 4 * K_B * (Tc + 273.15) * (fs / 2) / RL
    assert np.var(n) == pytest.approx(want, rel=0.03)
