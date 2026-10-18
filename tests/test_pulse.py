"""Форма импульса rect: спектр sinc^2, EVM."""
from dataclasses import replace

import numpy as np
import pytest

from mwp_converter.bpf import apply_bpf
from mwp_converter.model import simulate
from mwp_converter.params import Components, Impairments
from mwp_converter.response import Bpf
from mwp_converter.signals import make_psk_baseband, metrics, receive_line
from scipy.signal import welch
from helpers import cfg_quick


def test_rect_spectrum_shape():
    sps, rs = 64, 1.0
    tx = make_psk_baseband(4, 2 ** 14, sps, 0.35, 1, "rect")
    # detrend=False: иначе вычитание среднего в каждом сегменте занижает PSD у нуля
    f, p = welch(tx.x, fs=sps * rs, nperseg=2 ** 11, return_onesided=False, detrend=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    m = (f > 0.8) & (f < 1.2)
    f_null = f[m][np.argmin(p[m])]
    assert f_null == pytest.approx(1.0, rel=0.02)
    side = p[(f > 1.1) & (f < 1.9)].max() / p[np.abs(f) < 0.05].mean()
    assert 10 * np.log10(side) == pytest.approx(-13.3, abs=0.5)


def run_evm(pulse, bpf):
    cfg = cfg_quick()
    sim = replace(cfg.sim, pulse=pulse)
    res, tx, fs = simulate(cfg.plan, sim, Impairments(), Components(), 0)
    I = apply_bpf(res.I, fs, cfg.plan, sim, Bpf(), 0) if bpf else res.I
    f, sb = cfg.plan.line_info(0)
    rx, s = receive_line(I, fs, f, sb, sim, tx)
    return metrics(rx, s, sim.M)


def test_rect_ideal_evm_and_bpf():
    raw = run_evm("rect", False)
    assert raw["evm_pct"] < 3.0 and raw["ber_mc"] == 0
    assert run_evm("rect", True)["evm_pct"] > run_evm("rrc", True)["evm_pct"]


def test_rect_band_width():
    cfg = cfg_quick()
    assert replace(cfg.sim, pulse="rect").alpha_eff == 1.0
    assert cfg.sim.alpha_eff == 0.35
