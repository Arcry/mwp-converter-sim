"""Сверка API OptiCommPy (шаг 3.1): имена параметров и сигнатуры."""
import inspect

import numpy as np
from optic.comm.metrics import calcEVM, fastBERcalc, theoryBER
from optic.comm.modulation import demodulateGray, grayMapping, modulateGray
from optic.dsp.core import firFilter, pnorm, pulseShape
from optic.models.devices import basicLaserModel, mzm, photodiode
from optic.utils import parameters


def names(f):
    return list(inspect.signature(f).parameters)


def test_signatures():
    assert names(mzm) == ["Ei", "u", "param"]
    assert names(photodiode) == ["E", "param"]
    assert names(basicLaserModel) == ["param"]
    assert names(modulateGray) == ["bits", "M", "constType"]
    assert names(demodulateGray) == ["symb", "M", "constType"]
    assert names(grayMapping) == ["M", "constType"]
    assert names(calcEVM) == ["symb", "M", "constType", "symbTx"]
    assert names(fastBERcalc) == ["rx", "tx", "M", "constType", "px"]
    assert names(theoryBER) == ["M", "EbN0", "constType"]
    assert names(pulseShape) == ["param"]
    assert names(firFilter) == ["h", "x"]
    assert names(pnorm) == ["x"]


def test_mzm_bias_points():
    p = parameters()
    p.Vpi, p.ER = 2.0, 300
    p.Vb = 2.0
    assert abs(mzm(np.ones(2) + 0j, np.zeros(2), p)[0]) ** 2 < 1e-12
    p.Vb = 1.0
    assert abs(abs(mzm(np.ones(2) + 0j, np.zeros(2), p)[0]) ** 2 - 0.5) < 1e-12


def test_photodiode_ideal():
    p = parameters()
    p.ideal, p.R = True, 0.7
    E = np.random.default_rng(0).normal(size=8) + 1j * np.random.default_rng(1).normal(size=8)
    assert np.allclose(photodiode(E, p), 0.7 * np.abs(E) ** 2)
