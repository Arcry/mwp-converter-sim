"""C/I многонесущего входа (прогоны 1-2), маршруты примера и сценарии router.toml."""
from dataclasses import replace

import numpy as np
import pytest

from mwp_converter.interference import ci_runs
from mwp_converter.model import run_all, simulate
from mwp_converter.params import Impairments, load_config, quick, resolve_sps, scenario_plans
from mwp_converter.signals import make_multicarrier_baseband, metrics, receive_line
from helpers import CFG


def router(**kw):
    cfg = quick(load_config(CFG / "router.toml"))
    return replace(cfg, sim=replace(cfg.sim, **kw)) if kw else cfg


@pytest.fixture(scope="module")
def ci_by_beta():
    out = {}
    for bm in (0.5, 0.1):
        out[bm] = ci_runs(router(beta_m=bm), 1)
    return out


def test_ci_in_range(ci_by_beta):
    assert 15 < ci_by_beta[0.5]["ci_dB"] < 35


def test_ci_grows_at_low_drive(ci_by_beta):
    # IM3 ~ beta^3 против сигнала ~ beta: C/I ~ beta^-2 (ожидаемо +28 дБ при 0,5 -> 0,1)
    assert ci_by_beta[0.1]["ci_dB"] > ci_by_beta[0.5]["ci_dB"] + 20


def test_frequency_reuse_ber_zero(ci_by_beta):
    assert ci_by_beta[0.5]["ber_mc"] == 0 and ci_by_beta[0.5]["f_out"] == pytest.approx(19.8)
    assert ci_by_beta[0.5]["evm_pct"] < 20


def test_multicast_ber_zero():
    cfg = quick(load_config(CFG / "router.toml"))
    plan = scenario_plans(cfg)["multicast"]
    cfg = replace(cfg, plan=plan)
    mb = make_multicarrier_baseband(plan, cfg.sim, resolve_sps(plan, cfg.sim))
    res = run_all(plan, cfg.sim, Impairments(), cfg.comps, baseband=mb)
    assert plan.out_freqs() == pytest.approx([19.8, 12.0, 24.0])
    for i, r in enumerate(res):
        f, sb = plan.line_info(i)
        rx, s = receive_line(r.I, r.fs, f, sb, cfg.sim, mb, carrier=0)
        assert metrics(rx, s, 4)["ber_mc"] == 0, i


def test_inverted_needs_conjugation():
    cfg = quick(load_config(CFG / "router.toml"))
    plan = scenario_plans(cfg)["inverted"]
    cfg = replace(cfg, plan=plan)
    res, tx, fs = simulate(plan, cfg.sim, Impairments(), cfg.comps, 0)
    f, sb = plan.line_info(0)
    assert sb == "-" and f == pytest.approx(19.8)
    ok = metrics(*receive_line(res.I, fs, f, "-", cfg.sim, tx, carrier=0), 4)
    bad = metrics(*receive_line(res.I, fs, f, "+", cfg.sim, tx, carrier=0), 4)
    assert ok["ber_mc"] == 0 and bad["ber_mc"] > 0.3


# ---------- этап 2: ACI/IMD, NPR, эквалайзер ----------
def test_aci_imd_lo_sum_matches_total(ci_by_beta):
    for bm, r in ci_by_beta.items():
        assert abs(10 * np.log10((r["ACI"] + r["IMD"] + r["P0"]) / r["P2"])) < 1.0, bm
    assert ci_by_beta[0.5]["ci_imd_dB"] < ci_by_beta[0.5]["ci_aci_dB"]      # при beta_m = 0,5 доминирует IMD
    assert ci_by_beta[0.1]["ci_imd_dB"] <= 100                              # при 0,1 IMD пренебрежимо мала (потолок 100 дБ)
    assert ci_by_beta[0.5]["ci_lo_dB"] > 50                                 # линия гетеродина 4f_i ниже полезной


def test_aci_linear_in_drive(ci_by_beta):
    assert abs(ci_by_beta[0.1]["ci_aci_dB"] - ci_by_beta[0.5]["ci_aci_dB"]) < 2.0


def test_npr_higher_at_low_drive():
    from mwp_converter.interference import npr

    cfg = router()
    lo, hi = npr(replace(cfg, sim=replace(cfg.sim, beta_m=0.1)), 1), npr(replace(cfg, sim=replace(cfg.sim, beta_m=0.5)), 1)
    assert lo > hi + 15


def test_equalizer_reduces_isi():
    cfg = router(beta_m=0.1)
    r = ci_runs(cfg, 1, split=False)
    assert r["evm_eq_pct"] < 3.0 and r["evm_eq_pct"] < r["evm_pct"] / 2


def test_equalizer_off_returns_none():
    cfg = router(beta_m=0.1)
    r = ci_runs(replace(cfg, sim=replace(cfg.sim, equalizer="none")), 0, split=False)
    assert r["evm_eq_pct"] is None


@pytest.mark.slow
def test_drive_sweep_optimum_with_noise():
    from mwp_converter.interference import drive_sweep

    cfg = load_config(CFG / "router_realistic.toml")
    cfg = quick(replace(cfg, sim=replace(cfg.sim, n_sym=2 ** 9)))
    rows, best = drive_sweep(cfg, 1, [0.1, 0.33, 1.1])
    c = {r["beta"]: r["cni_dB"] for r in rows}
    assert best == pytest.approx(0.33) and c[0.33] > c[1.1] and c[0.33] > c[0.1]


def test_static_imp_keeps_drivers_in_lo_run():
    from mwp_converter.interference import _slot_power, static_imp

    cfg = router()
    p_off = _slot_power(cfg, 1, cfg.sim, static_imp(cfg.imp), "all")[0]
    p_on = _slot_power(cfg, 1, cfg.sim, static_imp(replace(cfg.imp, drivers=True)), "all")[0]
    assert abs(p_on / p_off - 1) > 0.01        # гармоники драйвера входят в P0
    noisy = replace(cfg.imp, drivers=True, rin_dBHz=-150.0, pd_noise=True, input_cn_dB=20.0, lo_pn=True,
                    bias_drift=True, amp_nf_dB=3.0)
    si = static_imp(noisy)
    assert si.drivers and not si.pd_noise and si.input_cn_dB is None and not si.lo_pn and not si.bias_drift


def test_run_all_wdm_leak_matches_full_field_path():
    """Утечка WDM: последовательный расчёт с |E_g|^2 совпадает с прежним путём (полные поля всех веток)."""
    from mwp_converter.model import _branch, _input_stage, _setup, photocurrent, run_all as ra

    cfg = quick(load_config(CFG / "router_realistic.toml"))
    cfg = replace(cfg, sim=replace(cfg.sim, n_sym=2 ** 8), imp=Impairments())
    plan, sim, comps = cfg.plan, cfg.sim, cfg.comps
    assert comps.wdm_isolation_dB and not comps.wdm.is_ideal
    sps = resolve_sps(plan, sim)
    mb = make_multicarrier_baseband(plan, sim, sps)
    sps, fs, n, x, tt, bb = _setup(plan, sim, mb, None)
    T2, _ = _input_stage(plan, sim, cfg.imp, comps, bb, x, tt, fs, sps)
    done = [_branch(i, plan, sim, cfg.imp, comps, T2, fs, tt, "null") for i in range(plan.N)]
    iso, offs = 10 ** (-comps.wdm_isolation_dB / 10), plan.offsets_GHz()
    new = ra(plan, sim, cfg.imp, comps, baseband=mb, keep_fields=False)
    for i in range(plan.N):
        xt = sum(iso * np.abs(done[j][0]["g"]) ** 2 for j in range(plan.N)
                 if j != i and abs(abs(offs[i] - offs[j]) - plan.wdm_spacing_GHz) < 1e-9)
        old = photocurrent(done[i][0]["g"], sim, cfg.imp, comps.pd, fs, seed=sim.seed + 7 * (i + 1),
                           extra_power=None if np.isscalar(xt) else xt, amp_seed=sim.seed + 11 * (i + 1))
        np.testing.assert_allclose(new[i].I, old, rtol=1e-12, atol=1e-18 * np.abs(old).max())
        assert "frag" in new[i].meta and isinstance(new[i].fields["g"], tuple)
