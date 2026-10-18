"""МЦМ1 в квадратуре против (16)."""
import pytest

from mwp_converter.analytic import lines_quad
from mwp_converter.model import measure_lines
from mwp_converter.params import Components
from helpers import cfg_default

NAMES = ["DC", "fi", "fc", "3fi", "3fc", "fi+fc", "fi-fc", "3fi+fc", "3fi-fc"]


def test_quad():
    cfg = cfg_default(flo=[9.0])
    meas = measure_lines(0, cfg.plan, cfg.sim, Components(), NAMES, bias1="quad")
    ref = lines_quad(1.0, 0.5)
    for n in NAMES:
        assert meas[n][1] == pytest.approx(ref[n], rel=1e-3), n
