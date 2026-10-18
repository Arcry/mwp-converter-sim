"""Дрейф смещения: процесс Орнштейна-Уленбека."""
import numpy as np
import pytest

from mwp_converter.model import run_channel
from mwp_converter.noise import ou_process
from mwp_converter.params import Components, Impairments
from helpers import cfg_default


def test_constant_within_run():
    b = ou_process(0.02, 1.0, 2 ** 16, 316e9, 1)
    assert np.ptp(b) < 1e-2 * 0.02


def test_initial_std():
    b0 = np.array([ou_process(0.02, 1.0, 2, 1e3, s)[0] for s in range(3000)])
    assert np.std(b0) == pytest.approx(0.02, rel=0.05)


def test_autocorrelation_at_tau():
    fs, tau, n = 1e3, 1.0, 2 ** 21
    b = ou_process(1.0, tau, n, fs, 5)
    lag = int(tau * fs)
    r = np.mean(b[:-lag] * b[lag:]) / np.mean(b * b)
    assert r == pytest.approx(np.exp(-1), abs=0.05)


def test_drift_changes_odd_line():
    cfg = cfg_default(flo=[9.0])
    base = run_channel(0, cfg.plan, cfg.sim, Impairments(), Components(), n=2 ** 15)
    dr = run_channel(0, cfg.plan, cfg.sim, Impairments(bias_drift=True, drift_sigma=0.05, drift_ens=2),
                     Components(), n=2 ** 15)
    assert np.std(dr.I - base.I) > 1e-6


@pytest.mark.parametrize("name", ["default", "realistic"])
def test_ensemble_matches_analytic_same_draws(name):
    from mwp_converter.params import load_config, quick
    from mwp_converter.plots import drift_ensemble
    import conftest

    cfg = quick(load_config(conftest.ROOT / "configs" / f"{name}.toml"))
    ev, lv, an = drift_ensemble(cfg, 5.0, 8)
    assert abs(np.median(lv) - np.median(an)) < 0.5
