"""Многотоновая аналитика против CW-прогонов (МЦМ1 в нуле)."""
import numpy as np
import pytest

from mwp_converter.analytic import ci_multitone, lines_multitone, lines_null, multitone_products
from mwp_converter.model import run_channel
from mwp_converter.params import Carrier, Components, Impairments, Plan, Route, SimParams, resolve_sps
from mwp_converter.spectra import coherent_length, line_amplitudes

SIM = SimParams(Rs=0.15, n_sym=2 ** 10)


def measure(fcs, fi, freqs):
    plan = Plan(carriers=tuple(Carrier(f) for f in fcs), routes=(Route(fi, 0),))
    fs = resolve_sps(plan, SIM) * SIM.Rs * 1e9
    n = coherent_length(fs, list(freqs) + [0.0], 2 ** 16)
    res = run_channel(0, plan, SIM, Impairments(), Components(), n=n)
    a = line_amplitudes(res.I, fs, freqs) / (SIM.R * 1e-2 / 4)
    return a, plan


def betas_of(K):
    return [SIM.beta_m / np.sqrt(K)] * K


def test_two_tones_lines_and_im3():
    fcs, fi = [29.8, 30.0], 5.5
    L = lines_multitone(SIM.beta1, betas_of(2), 0, fcs, fi)
    names = ["out", "im3(2c1-c2)", "im3(2c2-c1)", "fc"]
    a, _ = measure(fcs, fi, [L[n][0] for n in names])
    for v, n in zip(a, names):
        assert v == pytest.approx(L[n][1], rel=2e-3), n


def test_three_tones_triple_product():
    fcs, fi = [29.8, 30.0, 30.3], 5.5
    L = lines_multitone(SIM.beta1, betas_of(3), 0, fcs, fi)
    names = ["out", "im3(c1+c2-c3)", "im3(2c1-c2)"]
    a, _ = measure(fcs, fi, [L[n][0] for n in names])
    for v, n in zip(a, names):
        assert v == pytest.approx(L[n][1], rel=5e-3), n


def test_single_tone_equals_lines_null():
    L = lines_multitone(1.0, [0.5], 0, [30.0], 9.0)
    R = lines_null(1.0, 0.5)
    assert L["out"][1] == pytest.approx(R["2fi+fc"], rel=1e-12)
    assert L["fc"][1] == pytest.approx(R["fc"], rel=1e-12)
    assert L["2fi"][1] == pytest.approx(R["2fi"], rel=1e-12)
    assert L["DC"][1] == pytest.approx(R["DC"], rel=1e-12)
    assert L["out"][0] == pytest.approx(12.0)
    U = lines_multitone(1.0, [0.5], 0, [30.0], 9.0, "upper")
    assert U["out"][0] == pytest.approx(48.0)


def test_ci_multitone_example():
    fcs = [29.6, 29.8, 30.0, 30.2]
    betas = [0.25] * 4
    r = ci_multitone(1.0, betas, 1, fcs, 5.0)
    assert 15 < r["ci_dB"] < 30 and r["n"] >= 2
    assert len(multitone_products(1.0, betas, fcs)) == 4 + 12 + 12


def test_ci_multitone_uses_carrier_levels():
    from mwp_converter.params import Carrier
    from mwp_converter.signals import carrier_weights

    cs = [Carrier(29.6), Carrier(29.8, level_dB=-6.0), Carrier(30.0), Carrier(30.2)]
    a_eq = carrier_weights([Carrier(c.f_GHz) for c in cs])
    a_ne = carrier_weights(cs)
    assert float(np.sum(a_ne ** 2)) == pytest.approx(1.0) and a_ne[1] < a_eq[1]
    fcs = [c.f_GHz for c in cs]
    eq = ci_multitone(1.0, list(0.5 * a_eq), 0, fcs, 5.0)["ci_dB"]
    ne = ci_multitone(1.0, list(0.5 * a_ne), 0, fcs, 5.0)["ci_dB"]
    assert abs(ne - eq) > 0.5
    plan = Plan(carriers=tuple(cs), routes=(Route(5.0, 0),))
    from mwp_converter.signals import make_multicarrier_baseband
    mb = make_multicarrier_baseband(plan, SIM, resolve_sps(plan, SIM))
    np.testing.assert_allclose(mb.a, a_ne)
