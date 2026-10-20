"""Оптический фильтр перед ФД: несимметричная АЧХ, подавление линий, образ соседа."""
from dataclasses import replace

import numpy as np
import pytest

from mwp_converter.model import measure_lines, run_channel
from mwp_converter.params import Carrier, Components, Impairments, Plan, Route, SimParams, load_config, resolve_sps
from mwp_converter.response import OpticalFilter, band_report
from mwp_converter.spectra import coherent_length, line_amplitudes
from helpers import CFG, cfg_default


def test_asymmetric_response():
    of = OpticalFilter("lorentz", 0.1)
    plan = Plan(30.0, (9.0,))
    c = of.centers(plan, 0)
    assert c == pytest.approx((9.0, 21.0))
    assert abs(of.H([9e9], c)[0]) > 0.99 and abs(of.H([-9e9], c)[0]) < 0.01     # только +f_i
    up = of.centers(Plan(14.0, (8.0,), "upper"), 0)
    assert up == pytest.approx((8.0, -22.0))


def test_lines_fc_and_2f1_suppressed_and_useful_minus_6db():
    cfg = cfg_default(flo=[9.0])
    # -6 дБ — главный порядок по индексам: при beta1 = 1, beta_m = 0,5 боковые высших порядков
    # (±3f_i и др.) тоже дают вклад в линию и эффект только -4,9 дБ, поэтому берём малые индексы
    sim = replace(cfg.sim, beta1=0.3, beta_m=0.2)
    names = ["2fi-fc", "fc", "2fi"]
    base = measure_lines(0, cfg.plan, sim, Components(), names)
    comps = Components(ofilt=OpticalFilter("supergauss", 2.0, 2))
    flt = measure_lines(0, cfg.plan, sim, comps, names)
    assert 20 * np.log10(flt["2fi-fc"][1] / base["2fi-fc"][1]) == pytest.approx(-6.02, abs=0.3)
    for n in ("fc", "2fi"):
        assert flt[n][1] / flt["2fi-fc"][1] < 1e-2, n      # >= 40 дБ (линия обнуляется)


def test_neighbour_image_lorentz():
    sim = SimParams(Rs=0.15, n_sym=2 ** 10)
    cs = (Carrier(29.8), Carrier(30.0))
    plan = Plan(carriers=cs, routes=(Route.from_output(cs, 0, 19.8),))
    fs = resolve_sps(plan, sim) * sim.Rs * 1e9
    freqs = [19.8, 20.0]
    n = coherent_length(fs, freqs + [0.0], 2 ** 16)
    res = run_channel(0, plan, sim, Impairments(), Components(ofilt=OpticalFilter("lorentz", 0.1)), n=n)
    a = line_amplitudes(res.I, fs, freqs)
    # отношение образа соседа (20,0 ГГц) к полезной линии (19,8 ГГц): |h(0,2 ГГц)| = 1/sqrt(17) = -12,3 дБ
    # (без фильтра у линий собственный дисбаланс +0,45 дБ от вкладов других пар боковых, поэтому берём абсолют)
    assert 20 * np.log10(a[1] / a[0]) == pytest.approx(-12.3, abs=0.5)


def test_band_report_and_config():
    cfg = load_config(CFG / "router.toml")
    comps = replace(cfg.comps, ofilt=OpticalFilter("lorentz", 0.1))
    rep = band_report(cfg.plan, comps, cfg.sim.Rs, cfg.sim.alpha_eff)
    pb = [r for r in rep if r["component"] == "ofilt" and r["line"] == "passband"]
    assert len(pb) == 2 * cfg.plan.N and all(r["level"] == "ok" for r in pb)
    assert any(r["component"] == "ofilt" and r["line"].startswith("img(") for r in rep)
    assert load_config(CFG / "router.toml").comps.ofilt.is_none
    txt = (CFG / "router.toml").read_text() + '\n[components.ofilt]\nkind = "lorentz"\nfwhm_GHz = 0.1\n'
    c = load_config(text=txt)
    assert c.comps.ofilt.kind == "lorentz" and c.comps.ofilt.fwhm_GHz == 0.1
