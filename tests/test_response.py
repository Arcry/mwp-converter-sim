"""АЧХ компонентов."""
import numpy as np
import pytest

from mwp_converter.response import Response, apply
import conftest

CSV = conftest.ROOT / "configs" / "example_s21.csv"


def db(r, f_GHz):
    return 20 * np.log10(np.abs(r.H(np.array([f_GHz * 1e9]))[0]))


def test_rc_skin_3db():
    # -3,01 дБ у rc первого порядка (точное значение 10 lg 2), допуск 0,02 дБ
    assert db(Response("rc", 40.0), 40.0) == pytest.approx(-3.0, abs=0.02)
    assert db(Response("skin", 40.0), 40.0) == pytest.approx(-3.0, abs=0.01)


def test_brickwall():
    with pytest.warns(UserWarning):
        r = Response("brickwall", 10.0)
        assert abs(r.H(np.array([20e9]))[0]) == 0
        assert abs(r.H(np.array([5e9]))[0]) == 1


def test_supergauss():
    r = Response("supergauss", 80.0, order=2)
    assert 10 * np.log10(abs(r.H(np.array([40e9]))[0]) ** 2) == pytest.approx(-3.0, abs=0.02)
    assert abs(r.H(np.array([-40e9]))[0]) == pytest.approx(abs(r.H(np.array([40e9]))[0]))


def test_csv():
    r = Response("csv", csv_path=str(CSV))
    assert db(r, 20.0) == pytest.approx(-1.2, abs=1e-9)
    assert db(r, 40.0) == pytest.approx(-4.0, abs=1e-9)
    assert db(r, 30.0) == pytest.approx(-2.5, abs=1e-9)
    with pytest.warns(UserWarning):
        assert db(r, 100.0) == pytest.approx(-9.0, abs=1e-9)


def test_apply_ideal_identity():
    x = np.random.default_rng(0).normal(size=1000)
    np.testing.assert_allclose(apply(x, 1e9, Response("ideal")), x, rtol=1e-12)
    z = x + 1j * x[::-1]
    np.testing.assert_allclose(apply(z, 1e9, Response("ideal")), z, rtol=1e-12)


def test_apply_real_and_complex_lowpass():
    fs = 1e11
    t = np.arange(4000) / fs
    x = np.sin(2 * np.pi * 10e9 * t)
    r = Response("rc", 10.0)
    y = apply(x, fs, r)
    assert np.sqrt(2) * np.std(y) == pytest.approx(abs(r.H(np.array([10e9]))[0]), rel=1e-3)
