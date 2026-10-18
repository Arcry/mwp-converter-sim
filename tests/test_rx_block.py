"""Поблочная подстройка фазы приёмника."""
import numpy as np
import pytest

from mwp_converter.signals import metrics


def make(n=4096, evm=0.01, seed=0):
    rng = np.random.default_rng(seed)
    s = np.exp(1j * (np.pi / 2) * rng.integers(0, 4, n))
    return s + evm / np.sqrt(2) * (rng.normal(size=n) + 1j * rng.normal(size=n)), s


def test_linear_phase_ramp_removed():
    rx, s = make()
    ramp = np.exp(1j * np.deg2rad(10) * np.arange(len(s)) / len(s))
    ref = metrics(rx, s, 4)["evm_pct"]
    assert metrics(rx * ramp, s, 4, block=256)["evm_pct"] == pytest.approx(ref, abs=0.1)
    assert metrics(rx * ramp, s, 4)["evm_pct"] > ref + 0.3   # без блока набег заметен


def test_no_ramp_no_change():
    rx, s = make()
    assert abs(metrics(rx, s, 4, block=256)["evm_pct"] - metrics(rx, s, 4)["evm_pct"]) < 0.05
