"""Вносимые потери: поле умножается на sqrt(10^(-IL/10)) после МЦМ1, MUX, МЦМ2, DEMUX."""
from dataclasses import replace

import numpy as np
import pytest

from mwp_converter.model import simulate
from mwp_converter.params import Components, Impairments
from mwp_converter.signals import metrics, receive_line
from helpers import cfg_quick


def test_losses_scale_current_not_evm():
    cfg = cfg_quick()
    r0, tx, fs = simulate(cfg.plan, cfg.sim, Impairments(), Components(), 0)
    r1, _, _ = simulate(cfg.plan, cfg.sim, Impairments(loss_dB=(5, 3, 5, 3)), Components(), 0)
    assert r1.I.mean() == pytest.approx(r0.I.mean() * 10 ** (-1.6), rel=1e-6)
    f, sb = cfg.plan.line_info(0)
    ev = []
    for r in (r0, r1):
        rx, s = receive_line(r.I, fs, f, sb, cfg.sim, tx)
        ev.append(metrics(rx, s, cfg.sim.M)["evm_pct"])
    assert ev[1] == pytest.approx(ev[0], rel=1e-6)


def test_default_has_no_loss():
    assert Impairments().loss_dB == (0.0, 0.0, 0.0, 0.0)
    assert replace(Impairments(), loss_dB=(1, 0, 0, 0)).loss_dB[0] == 1
    assert np.allclose(Impairments().loss_dB, 0)
