"""Многонесущий вход: композит, веса, PSD, общий T2."""
import numpy as np
import pytest

from mwp_converter.model import _branch, _input_stage, _setup, mzm_field, run_channel
from mwp_converter.params import Carrier, Components, Impairments, Plan, Route, SimParams, resolve_sps
from mwp_converter.signals import drive_stats, make_multicarrier_baseband, make_psk_baseband
from mwp_converter.spectra import rf_psd_abs
from helpers import cfg_quick

CS = tuple(Carrier(f, seed_offset=k) for k, f in enumerate((29.6, 29.8, 30.0, 30.2)))
PLAN = Plan(carriers=CS, routes=tuple(Route.from_output(CS, k, 19.8) for k in range(4)))
SIM = SimParams(Rs=0.15, n_sym=2 ** 10)


def test_k1_matches_old_input():
    cfg = cfg_quick()
    sps = resolve_sps(cfg.plan, cfg.sim)
    old = make_psk_baseband(cfg.sim.M, cfg.sim.n_sym, sps, cfg.sim.alpha, cfg.sim.seed, cfg.sim.pulse)
    new = make_multicarrier_baseband(cfg.plan, cfg.sim, sps)
    np.testing.assert_allclose(new.x, old.x, rtol=1e-12, atol=0)
    assert (new.symbols == old.symbols).all() and (new.bits == old.bits).all()


def test_weights_and_power():
    sps = resolve_sps(PLAN, SIM)
    mb = make_multicarrier_baseband(PLAN, SIM, sps)
    assert float(np.sum(mb.a ** 2)) == pytest.approx(1.0)
    assert np.mean(np.abs(mb.x) ** 2) == pytest.approx(1.0, abs=0.02)
    lv = Plan(carriers=(Carrier(29.8, level_dB=0.0), Carrier(30.0, level_dB=-6.0)),
              routes=(Route(5.0, 0),))
    mb2 = make_multicarrier_baseband(lv, SIM, resolve_sps(lv, SIM))
    assert mb2.a[0] ** 2 / mb2.a[1] ** 2 == pytest.approx(10 ** 0.6)
    st = drive_stats(mb, SIM)
    assert sum(b ** 2 for b in st["beta_k"]) == pytest.approx(SIM.beta_m ** 2)
    assert st["papr_dB"] > 3 and st["ibo_dB"] == pytest.approx(20 * np.log10(np.pi / 2 / SIM.beta_m))


def test_zero_carrier_not_renormalized():
    sps = resolve_sps(PLAN, SIM)
    mb = make_multicarrier_baseband(PLAN, SIM, sps, zero=1)
    assert mb.a[1] == 0 and mb.a[0] == pytest.approx(0.5)


def test_psd_peaks_at_carriers():
    n = 2 ** 17
    res = run_channel(0, PLAN, SIM, Impairments(), Components(), n=n)       # CW мультитон
    f, p = rf_psd_abs(res.meta["u2_in"], res.fs, 2 ** 16)
    for c in CS:
        m = np.abs(f - c.f_GHz) < 0.02
        assert abs(f[m][np.argmax(p[m])] - c.f_GHz) < 0.01


def test_shared_T2_equals_direct():
    cfg = cfg_quick()
    plan, sim, imp, comps = cfg.plan, cfg.sim, Impairments(loss_dB=(0, 0, 3.0, 0)), Components()
    res = run_channel(0, plan, sim, imp, comps, baseband=None, n=2 ** 14)
    direct = 10 ** (-3.0 / 20) * mzm_field(res.fields["d"], res.meta["u2_in"], "quad", sim.Vpi, None, None, res.fs)
    np.testing.assert_allclose(res.fields["e"], direct, rtol=1e-12, atol=1e-15)
    sps, fs, n, x, t, bb = _setup(plan, sim, None, 2 ** 14)
    T2, extra = _input_stage(plan, sim, imp, comps, bb, x, t, fs, sps)
    fields, ase = _branch(0, plan, sim, imp, comps, T2, fs, t, "null")
    np.testing.assert_allclose(fields["e"], res.fields["e"], rtol=1e-12, atol=0)
    assert ase is None
