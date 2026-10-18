"""Входной шум: C/N на входе LNA в полосе Rs, каскад по Фриису."""
from dataclasses import replace

import pytest

from mwp_converter.model import simulate
from mwp_converter.noise import nf_cascade
from mwp_converter.params import Components, Impairments
from mwp_converter.signals import metrics, receive_line
from helpers import cfg_quick


def snr_out(cn, **kw):
    cfg = cfg_quick()
    sim = replace(cfg.sim, beta_m=0.2)
    imp = Impairments(input_cn_dB=cn, **kw)
    res, tx, fs = simulate(cfg.plan, sim, imp, Components(), 0)
    f, sb = cfg.plan.line_info(0)
    rx, s = receive_line(res.I, fs, f, sb, sim, tx)
    return metrics(rx, s, sim.M)["snr_evm_dB"]


def test_nf_cascade():
    assert nf_cascade(2, 30, 8) == pytest.approx(2.02, abs=0.01)
    assert nf_cascade(0, 0, 8) == pytest.approx(8.0, abs=0.01)


@pytest.mark.parametrize("cn", [10.0, 20.0])
def test_snr_equals_cn_for_nf0(cn):
    assert snr_out(cn, nf_lna_dB=0.0, nf_driver_dB=0.0) == pytest.approx(cn, abs=0.4)


@pytest.mark.parametrize("cn", [10.0, 20.0])
def test_snr_with_cascade_nf(cn):
    assert snr_out(cn) == pytest.approx(cn - 2.02, abs=0.4)


def test_x_in_saved_and_reference_clean():
    cfg = cfg_quick()
    imp = Impairments(input_cn_dB=15.0)
    res, tx, fs = simulate(cfg.plan, cfg.sim, imp, Components(), 0)
    assert res.meta["x_in"].shape == res.meta["x"].shape
    assert abs(res.meta["x_in"] - res.meta["x"]).mean() > 1e-3
    assert (res.meta["x"] == tx.x).all()
