"""Приёмник линии: BER/EVM, инверсия спектра (сопряжение iff 2 f_i > f_c)."""
from dataclasses import replace

import pytest

from mwp_converter.model import simulate
from mwp_converter.params import Components, Impairments
from mwp_converter.signals import metrics, receive_line
from helpers import cfg_quick


@pytest.fixture(scope="module")
def runs():
    cfg = cfg_quick()
    out = {}
    for bm in (0.5, 0.1):
        sim = replace(cfg.sim, beta_m=bm)
        out[bm] = (cfg, sim) + tuple(simulate(cfg.plan, sim, Impairments(), Components(), i)
                                     for i in (0, 2))
    return out


def ber_evm(cfg, sim, res, tx, fs, i, sideband, f=None):
    fl = f if f is not None else cfg.plan.line_info(i)[0]
    rx, s = receive_line(res.I, fs, fl, sideband, sim, tx)
    return metrics(rx, s, sim.M)


def test_rule_matches_plan(runs):
    plan = runs[0.5][0].plan
    assert plan.line_info(0)[1] == "+"      # 2*5.5 < 30: инверсии нет
    assert plan.line_info(2)[1] == "-"      # 2*17 > 30: инверсия


def test_channel1_no_conjugation(runs):
    cfg, sim, (r1, tx, fs), _ = runs[0.5]
    f = cfg.plan.line_info(0)[0]
    assert f == pytest.approx(19.0)
    assert ber_evm(cfg, sim, r1, tx, fs, 0, "+")["ber_mc"] == 0
    assert ber_evm(cfg, sim, r1, tx, fs, 0, "-")["ber_mc"] > 0.3


def test_channel3_conjugation(runs):
    cfg, sim, _, (r3, tx, fs) = runs[0.5]
    assert ber_evm(cfg, sim, r3, tx, fs, 2, "-")["ber_mc"] == 0
    assert ber_evm(cfg, sim, r3, tx, fs, 2, "+")["ber_mc"] > 0.3


def test_upper_and_fc(runs):
    cfg, sim, (r1, tx, fs), _ = runs[0.5]
    for f, sb in ((cfg.plan.line_info(0, "upper")), (cfg.plan.fc_GHz, "+")):
        assert ber_evm(cfg, sim, r1, tx, fs, 0, sb, f)["ber_mc"] == 0


def test_evm(runs):
    for bm, lim in ((0.5, 15.0), (0.1, 3.0)):
        cfg, sim, (r1, tx, fs), (r3, _, _) = runs[bm]
        assert ber_evm(cfg, sim, r1, tx, fs, 0, "+")["evm_pct"] < lim
        assert ber_evm(cfg, sim, r3, tx, fs, 2, "-")["evm_pct"] < lim
