"""Нелинейность драйвера: HD2, IM3 через OIP3, среднее."""
import numpy as np
import pytest

from mwp_converter.noise import driver_poly

FS, N, RL, OIP3 = 2.0 ** 14, 2 ** 14, 50.0, 33.0  # FS = N: целые частоты попадают в бины
A_IP3 = np.sqrt(2 * RL * 10 ** ((OIP3 - 30) / 10))


def amp(y, f):
    X = np.abs(np.fft.rfft(y)) / (len(y) / 2)
    return X[int(round(f * len(y) / FS))]


def test_hd2():
    t = np.arange(N) / FS
    A, f = 1.0, 125.0
    for hd2 in (-30.0, -40.0):
        y = driver_poly(A * np.cos(2 * np.pi * f * t), A, OIP3, hd2, RL)
        assert 20 * np.log10(amp(y, 2 * f) / amp(y, f)) == pytest.approx(hd2, abs=0.1)


def test_im3():
    t = np.arange(N) / FS
    A, f1, f2 = 0.03 * A_IP3, 100.0, 120.0
    y = driver_poly(A * (np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)), 1.0, OIP3, -300.0, RL)
    assert 20 * np.log10(amp(y, 2 * f1 - f2) / amp(y, f1)) == pytest.approx(20 * np.log10((A / A_IP3) ** 2), abs=0.1)


def test_mean_zero():
    u = np.sin(2 * np.pi * 5 * np.arange(N) / FS)
    assert abs(driver_poly(u, 1.0, OIP3, -30.0, RL).mean()) < 1e-12
