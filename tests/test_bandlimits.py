"""Полосы компонентов, отчёт о полосах, произвольные частоты."""

import numpy as np
import pytest

from mwp_converter.analytic import lines_null
from mwp_converter.model import measure_lines, run_channel
from mwp_converter.params import Components, Impairments, Plan, SimParams, resolve_sps
from mwp_converter.response import Response, band_report
from mwp_converter.spectra import line_amplitudes
from helpers import cfg_default

N9 = ["2fi", "fc", "2fi+fc", "2fi-fc", "3fc"]


def test_mzm1_rc():
    cfg = cfg_default(flo=[9.0])
    comps = Components(mzm1_rf=Response("rc", 10.0))
    meas = measure_lines(0, cfg.plan, cfg.sim, comps, N9 + ["DC"])
    h = abs(comps.mzm1_rf.H(np.array([9e9]))[0])
    ref = lines_null(1.0 * h, 0.5)
    for n in N9 + ["DC"]:
        assert meas[n][1] == pytest.approx(ref[n], rel=1e-3), n


def test_pd_rc():
    cfg = cfg_default(flo=[9.0])
    comps = Components(pd=Response("rc", 20.0))
    ideal = measure_lines(0, cfg.plan, cfg.sim, Components(), ["2fi+fc", "fc"])
    meas = measure_lines(0, cfg.plan, cfg.sim, comps, ["2fi+fc", "fc"])
    for n in ("2fi+fc", "fc"):
        h = abs(comps.pd.H(np.array([meas[n][0] * 1e9]))[0])
        assert meas[n][1] == pytest.approx(ideal[n][1] * h, rel=1e-3)


def test_band_report_wdm():
    plan = Plan(30.0, (17.0,), "lower")
    comps = Components(wdm=Response("supergauss", 80.0, 2), wdm_isolation_dB=25.0)
    rep = band_report(plan, comps)
    r = [x for x in rep if x["component"] == "wdm" and abs(x["f_GHz"] - 47.0) < 1e-9]
    assert r and r[0]["level"] == "out"


def test_arbitrary_frequencies_flattop():
    plan = Plan(29.3, (5.17,), "lower")
    sim = SimParams(n_sym=2 ** 11)
    r = run_channel(0, plan, sim, Impairments(), Components(), n=2 ** 16)
    ref = lines_null(1.0, 0.5)
    fi, fc = 5.17, 29.3
    freqs = [2 * fi, fc, 2 * fi + fc, abs(2 * fi - fc)]
    a = line_amplitudes(r.I, r.fs, freqs) / (0.7 * 1e-2 / 4)
    for v, n in zip(a, ["2fi", "fc", "2fi+fc", "2fi-fc"]):
        assert v == pytest.approx(ref[n], rel=2e-3), n


@pytest.mark.parametrize("fc,flo", [(30.0, (5.5, 9.0, 17.0)), (29.3, (5.17,)), (14.0, (8.0,)),
                                    (10.0, (2.0, 40.0))])
def test_auto_fs(fc, flo):
    plan, sim = Plan(fc, flo), SimParams()
    fneed = max(max(4 * f + 3 * fc, 3 * f + 3 * fc + (1 + sim.alpha) * sim.Rs) for f in flo)
    assert resolve_sps(plan, sim) * sim.Rs >= 2.5 * fneed
    assert resolve_sps(plan, sim) % 2 == 0 and resolve_sps(plan, sim) >= 64


def test_useful_chain_ignores_3fc():
    from mwp_converter.response import band_report_for_output
    from helpers import CFG
    from mwp_converter.params import load_config

    r = load_config(CFG / "realistic.toml", dict(flo=[10.57]))
    rep = band_report_for_output(r.plan, r.comps, r.sim.Rs, r.sim.alpha, 1)
    assert rep and all(x["line"] not in ("3fc", "fc") for x in rep if x["component"] == "pd")
    assert not any(x["level"] == "out" and "3fc" in x["line"] for x in rep)
    full = band_report(r.plan, r.comps)
    assert any(x["level"] == "out" and x["line"] == "3fc" for x in full)


def test_useful_chain_wdm_sidebands():
    from mwp_converter.response import band_report_for_output
    from helpers import CFG
    from mwp_converter.params import load_config

    for direction, expect_out in (("lower", False), ("upper", True)):
        r = load_config(CFG / "realistic.toml", dict(flo=[10.57], direction=direction))
        rep = band_report_for_output(r.plan, r.comps, r.sim.Rs, r.sim.alpha, 1)
        w = [x for x in rep if x["component"] == "wdm"]
        assert w and all("MUX+DEMUX" in x["line"] for x in w)
        assert any(x["level"] == "out" for x in w) == expect_out
        if direction == "lower":
            assert sorted(round(x["f_GHz"], 2) for x in w) == [10.57, 19.43]
        else:
            assert any(abs(x["f_GHz"] - 40.57) < 1e-9 and x["level"] == "out" for x in w)
