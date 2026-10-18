"""Уход смещения MZM1: нечётные линии f_i, f_i±f_c появляются только при bias_err != 0."""
import numpy as np
import pytest

from mwp_converter.analytic import bias_error_ratio
from mwp_converter.model import measure_lines
from mwp_converter.params import Components, Impairments
from helpers import cfg_default

NAMES = ["fi", "fi+fc", "fi-fc", "2fi+fc", "2fi-fc"]


def meas(e):
    cfg = cfg_default(flo=[9.0])
    return measure_lines(0, cfg.plan, cfg.sim, Components(), NAMES, imp=Impairments(bias_err=e))


def test_zero_error_no_odd_lines():
    m = meas(0.0)
    for n in ("fi+fc", "fi-fc"):
        assert 20 * np.log10(m[n][1] / m["2fi+fc"][1]) < -80


@pytest.mark.parametrize("e", [0.01, 0.03, -0.03, 0.1])
def test_ratio_matches_analytic(e):
    m = meas(e)
    for n in ("fi+fc", "fi-fc"):
        assert m[n][1] / m["2fi+fc"][1] == pytest.approx(bias_error_ratio(1.0, e), rel=1e-2)
    assert m["fi+fc"][1] > 0.04 * m["2fi+fc"][1]
