"""Фазовый шум гетеродина: СПМ по таблице L(f) и удвоение на линии 2f_i."""
import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt, welch

from mwp_converter.model import run_channel
from mwp_converter.noise import LO_PRESETS, interp_L, lo_table, phase_noise
from mwp_converter.params import Components, Impairments
from helpers import cfg_default


def test_interp_and_presets():
    t = LO_PRESETS["synth_typical"]
    assert interp_L(t, 1e6) == pytest.approx(-115.0)
    assert interp_L(t, 10.0) == pytest.approx(-75.0)       # левее таблицы: крайнее значение
    assert interp_L(t, 1e9) == pytest.approx(-145.0)
    assert interp_L(t, 10 ** 5.5) == pytest.approx(-105.0)  # линейно в дБ по lg f
    assert lo_table(Impairments(lo_profile="dvbs2_mask")) == LO_PRESETS["dvbs2_mask"]
    assert lo_table(Impairments(lo_profile="table", lo_table=((1e3, -80.0), (1e6, -120.0)))) == ((1e3, -80.0), (1e6, -120.0))


def test_psd_equals_2L():
    fs, n = 4e9, 2 ** 22
    phi = phase_noise(LO_PRESETS["synth_typical"], n, fs, 3)
    f, p = welch(phi, fs=fs, nperseg=2 ** 16, detrend=False)
    for f0 in (1e6, 1e7):
        m = (f > f0 * 0.9) & (f < f0 * 1.1)
        got = 10 * np.log10(p[m].mean())
        assert got == pytest.approx(10 * np.log10(2) + interp_L(LO_PRESETS["synth_typical"], f0), abs=1.5)


def test_2f1_line_has_6db_more_phase_noise():
    cfg = cfg_default(flo=[9.0])
    plan, sim = cfg.plan, cfg.sim
    n = 2 ** 19
    imp = Impairments(lo_pn=True)
    r = run_channel(0, plan, sim, imp, Components(), n=n)
    fs, t = r.fs, r.t
    phi = phase_noise(lo_table(imp), n, fs, sim.seed + 500)
    z = r.I * np.exp(-2j * np.pi * 18e9 * t)
    z = sosfiltfilt(butter(4, 300e6 / (fs / 2), output="sos"), z)
    ph = np.unwrap(np.angle(z))
    f, p_out = welch(ph - ph.mean(), fs=fs, nperseg=2 ** 16, detrend=False)
    _, p_lo = welch(phi, fs=fs, nperseg=2 ** 16, detrend=False)
    m = (f > 5e6) & (f < 20e6)
    assert 10 * np.log10(p_out[m].mean() / p_lo[m].mean()) == pytest.approx(6.0, abs=0.5)
