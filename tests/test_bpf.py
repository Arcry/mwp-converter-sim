"""Выходной BPF: АЧХ, подавление паразитных линий, EVM, центр auto/фиксированный."""
from dataclasses import replace

import numpy as np
import pytest

from mwp_converter.bpf import apply_bpf, bpf_for, bpf_response, group_delay_ns, suppression_table
from mwp_converter.model import simulate
from mwp_converter.params import Components, Impairments
from mwp_converter.response import Bpf, band_report_for_output
from mwp_converter.signals import metrics, receive_line
from helpers import cfg_quick


def evm_after(cfg, plan, comps, sim=None):
    sim = sim or cfg.sim
    res, tx, fs = simulate(plan, sim, Impairments(), comps, 0)
    Ib = apply_bpf(res.I, fs, plan, sim, comps.bpf, 0)
    f, sb = plan.line_info(0)
    rx, s = receive_line(Ib, fs, f, sb, sim, tx)
    return metrics(rx, s, sim.M)["evm_pct"], res, Ib, fs


def test_center_attenuation_and_edges():
    cfg = cfg_quick()
    fs = 396e9
    f0, bw = bpf_for(cfg.plan, cfg.sim, Bpf(), 0)
    assert f0 == pytest.approx(19.0) and bw == pytest.approx(2 * 1.35)
    h = np.abs(bpf_response(fs, cfg.plan, cfg.sim, Bpf(), 0, [f0, f0 - bw / 2, f0 + bw / 2]))
    assert 20 * np.log10(h[0]) > -0.5
    assert 20 * np.log10(h[1]) == pytest.approx(-3.0, abs=0.1)
    gd = group_delay_ns(fs, cfg.plan, cfg.sim, Bpf(), 0, np.array([18.5, 19.0, 19.5]))
    assert np.all(gd > 0)


def test_cheby1_runs():
    cfg = cfg_quick()
    comps = Components(bpf=Bpf(kind="cheby1", ripple_dB=1.0))
    evm, *_ = evm_after(cfg, cfg.plan, comps)
    assert evm < 8.0


def test_2f1_suppressed_and_evm():
    cfg = cfg_quick()
    evm, res, Ib, fs = evm_after(cfg, cfg.plan, Components())
    assert evm < 5.0
    rows = {r["line"]: r for r in suppression_table(res.I, Ib, fs, cfg.plan, cfg.sim, 0)}
    assert rows["2fi"]["after_dBc"] < rows["2fi"]["before_dBc"] - 30
    assert rows["2fi"]["after_dBc"] < -30


def test_fixed_center_vs_auto():
    cfg = cfg_quick()
    plan2 = replace(cfg.plan, f_lo_GHz=(6.5, 9.0, 17.0))
    fixed = Components(bpf=Bpf(center=(19.0,)))
    auto = Components(bpf=Bpf(center=("auto",)))
    rep = band_report_for_output(plan2, fixed, cfg.sim.Rs, cfg.sim.alpha_eff, 1)
    assert any(r["component"] == "bpf" and r["level"] == "out" and "outside BPF" in r.get("msg", "") for r in rep)
    assert not any(r["component"] == "bpf" and r["level"] == "out"
                   for r in band_report_for_output(plan2, auto, cfg.sim.Rs, cfg.sim.alpha_eff, 1))
    ref, *_ = evm_after(cfg, cfg.plan, auto)
    e_auto, *_ = evm_after(cfg, plan2, auto)
    e_fix, *_ = evm_after(cfg, plan2, fixed)
    assert abs(e_auto - ref) < 0.5
    assert e_fix > e_auto + 5


def test_spurious_in_band_flagged():
    cfg = cfg_quick()
    plan = replace(cfg.plan, f_lo_GHz=(5.5, 9.0, 17.0))
    comps = Components(bpf=Bpf(center=(22.0,), bw_GHz=6.0))   # 4f1 = 22 ГГц внутри полосы
    rep = band_report_for_output(plan, comps, cfg.sim.Rs, cfg.sim.alpha_eff, 1)
    assert any(r["component"] == "bpf" and r["line"] == "4fi" and r["level"] == "out" for r in rep)
