"""Спектр входного PM-сигнала u2 на МЦМ2."""
import numpy as np

from mwp_converter.model import simulate
from mwp_converter.params import Components, Impairments
from mwp_converter.response import Response
from mwp_converter.spectra import rf_psd_abs
from helpers import cfg_quick


def test_u2_spectrum():
    cfg = cfg_quick()
    res, tx, fs = simulate(cfg.plan, cfg.sim, Impairments(), Components(), 0)
    fc, bw = cfg.plan.fc_GHz, (1 + cfg.sim.alpha) * cfg.sim.Rs
    f, p = rf_psd_abs(res.meta["u2_in"], fs, 2 ** 15)
    near = np.abs(f - fc) < 3
    centroid = (f[near] * p[near]).sum() / p[near].sum()
    # у RRC-спектра плоская вершина шириной (1-alpha)*Rs: положение максимума случайно в её пределах,
    # поэтому точное положение центра проверяется по центроиду (±0,05 ГГц), а максимум — по вершине
    assert abs(centroid - fc) < 0.05
    assert abs(f[p.argmax()] - fc) < (1 - cfg.sim.alpha) * cfg.sim.Rs / 2 + 0.05
    out = np.abs(f - fc) > bw / 2
    assert 10 * np.log10(p[out].max() / p.max()) < -30


def test_meta_keys_and_response():
    cfg = cfg_quick()
    comps = Components(mzm2_rf=Response("rc", 20.0))
    res, tx, fs = simulate(cfg.plan, cfg.sim, Impairments(), comps, 0)
    for k in ("u2_in", "u2_mzm", "symbols", "x"):
        assert k in res.meta
    assert np.std(res.meta["u2_mzm"]) < np.std(res.meta["u2_in"])
    np.testing.assert_array_equal(res.meta["symbols"], tx.symbols)
