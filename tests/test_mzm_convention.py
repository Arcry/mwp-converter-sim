"""Конвенция МЦМ: beta -> драйв, нуль и квадратура, конечный ER."""
import numpy as np
import pytest

from mwp_converter.model import beta_to_drive, mzm_field

VPI = 2.0
T = np.arange(4096) / 4096
WT = 2 * np.pi * 50 * T


@pytest.mark.parametrize("beta", [0.3, 1.0, 2.0])
def test_null(beta):
    u = beta_to_drive(beta, VPI) * np.sin(WT)
    out = np.abs(mzm_field(np.ones_like(T) + 0j, u, "null", VPI)) ** 2
    np.testing.assert_allclose(out, np.sin(beta * np.sin(WT)) ** 2, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("beta", [0.3, 1.0, 2.0])
def test_quadrature(beta):
    u = beta_to_drive(beta, VPI) * np.sin(WT)
    out = np.abs(mzm_field(np.ones_like(T) + 0j, u, "quad", VPI)) ** 2
    np.testing.assert_allclose(out, np.cos(beta * np.sin(WT) + np.pi / 4) ** 2, rtol=1e-9, atol=1e-12)


def test_er_300_is_ideal():
    u = beta_to_drive(1.0, VPI) * np.sin(WT)
    a = np.abs(mzm_field(np.ones_like(T) + 0j, u, "null", VPI, er_dB=300.0)) ** 2
    b = np.abs(mzm_field(np.ones_like(T) + 0j, u, "null", VPI, er_dB=None)) ** 2
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("er", [20.0, 30.0, 40.0])
def test_er_ratio(er):
    one = np.ones(2) + 0j
    p_null = np.abs(mzm_field(one, np.zeros(2), "null", VPI, er_dB=er)[0]) ** 2
    p_max = np.abs(mzm_field(one, np.zeros(2), "max", VPI, er_dB=er)[0]) ** 2
    assert abs(10 * np.log10(p_null / p_max) + er) < 0.1
