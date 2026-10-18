"""CW-прогон против (13'), идеальная модель."""
import numpy as np
import pytest

from mwp_converter.analytic import lines_null
from mwp_converter.model import measure_lines
from mwp_converter.params import Components
from mwp_converter.spectra import coherent_length, line_amplitudes
from helpers import cfg_default

NAMES = ["DC", "2fi", "fc", "2fi+fc", "2fi-fc", "3fc", "2fi+3fc", "2fi-3fc", "4fi+fc", "4fi-fc"]


@pytest.fixture(scope="module")
def meas():
    cfg = cfg_default(flo=[9.0])
    return measure_lines(0, cfg.plan, cfg.sim, Components(), NAMES)


def test_ratios_to_dc(meas):
    ref = lines_null(1.0, 0.5)
    dc = meas["DC"][1]
    for n in NAMES[1:]:
        assert meas[n][1] / dc == pytest.approx(ref[n] / ref["DC"], rel=1e-3)


@pytest.mark.parametrize("name,val", [("fc", 0.683), ("2fi", 0.706), ("2fi+fc", 0.310),
                                      ("2fi-fc", 0.310), ("3fc", 0.030)])
def test_absolute(meas, name, val):
    # значения плана округлены до 3 знаков: допуск abs=6e-4; точное сравнение — ниже (rel 1e-3)
    assert meas[name][1] == pytest.approx(val, abs=6e-4)
    assert meas[name][1] == pytest.approx(lines_null(1.0, 0.5)[name], rel=1e-3)


def test_absent_lines():
    from mwp_converter.model import run_channel

    cfg = cfg_default(flo=[9.0])
    plan, sim = cfg.plan, cfg.sim
    fs = 316e9
    fr = [18, 48, 12, 39, 21, 57, 3]
    n = coherent_length(fs, [0, 18, 30, 48, 12, 90, 108, 72, 66, 6] + fr[3:], 2 ** 16)
    r = run_channel(0, plan, sim, cfg.imp, Components(), n=n)
    a = line_amplitudes(r.I, r.fs, fr)
    wanted = np.mean(a[[1, 2]])
    for k in (3, 4, 5, 6):
        assert 20 * np.log10((a[k] + 1e-30) / wanted) < -80
