"""EDFA: ASE в двух поляризациях, сигнал-ASE и ASE-ASE шум фототока."""
import numpy as np
import pytest
from scipy.signal import welch

from mwp_converter.model import photocurrent
from mwp_converter.noise import edfa_field, n_ase
from mwp_converter.params import Impairments, SimParams
from mwp_converter.response import Response

FS = 100e9
N = 2 ** 20
G, NF, LAM = 16.0, 5.0, 1550.0


def test_ase_variance():
    E = np.zeros(N, complex)
    out, nperp = edfa_field(E, G, NF, FS, 1, LAM)
    want = n_ase(G, NF, LAM) * FS
    assert np.var(out) == pytest.approx(want, rel=0.03)
    assert np.var(nperp) == pytest.approx(want, rel=0.03)


def test_ase_false_exact():
    E = np.random.default_rng(0).normal(size=1000) + 0j
    out, nperp = edfa_field(E, G, NF, FS, 1, LAM, ase=False)
    np.testing.assert_array_equal(out, np.sqrt(10 ** (G / 10)) * E)
    assert not nperp.any()


def _psd(ase_with_signal):
    Pin = 1e-3 if ase_with_signal else 0.0
    E = np.sqrt(Pin) * np.ones(N, complex)
    out, nperp = edfa_field(E, G, NF, FS, 3, LAM)
    sim = SimParams()
    I = photocurrent(out, sim, Impairments(), Response("ideal"), FS, extra_power=np.abs(nperp) ** 2)
    f, p = welch(I - I.mean(), fs=FS, nperseg=2 ** 12, detrend=False)
    return np.mean(p[1:-1]), Pin


def test_signal_ase_plus_ase_ase_psd():
    p, Pin = _psd(True)
    R, N0, g = 0.7, n_ase(G, NF, LAM), 10 ** (G / 10)
    assert p == pytest.approx(4 * R ** 2 * N0 * (g * Pin + N0 * FS), rel=0.1)


def test_ase_only_psd():
    p, _ = _psd(False)
    R, N0 = 0.7, n_ase(G, NF, LAM)
    assert p == pytest.approx(4 * R ** 2 * N0 ** 2 * FS, rel=0.1)


def test_warn_for_ideal_wdm():
    from mwp_converter.params import Plan
    from mwp_converter.response import band_report
    from mwp_converter.params import Components
    rep = band_report(Plan(30.0, (5.5,)), Components(), imp=Impairments(edfa=True))
    assert any(r["component"] == "edfa" and r["level"] == "warn" for r in rep)
    assert not any(r["component"] == "edfa" for r in band_report(Plan(30.0, (5.5,)), Components(),
                                                                  imp=Impairments()))
