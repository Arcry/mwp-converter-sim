"""Бюджет шумов: вклад каждого источника этапа 1 в SNR на выходе канала (по EVM)."""
from dataclasses import replace

import numpy as np

from .model import simulate
from .params import Impairments
from .signals import metrics, receive_line


def _snr(plan, sim, imp, comps, i):
    res, tx, fs = simulate(plan, sim, imp, comps, i)
    f, sb = plan.line_info(i)
    rx, s_ref = receive_line(res.I, fs, f, sb, sim, tx)
    m = metrics(rx, s_ref, sim.M, sim.rx_block)
    return m["evm_pct"], m["snr_evm_dB"]


def noise_budget(cfg, i=0, cn_dB=None):
    """Бюджет шумов канала i.

    База ("distortion floor"): полосы, потери, статические неидеальности (ER, смещение),
    EDFA без ASE (если включён в cfg.imp); все источники шума выключены. Затем каждый источник
    включается поверх базы по одному: input C/N+NF, фазовый шум гетеродина (synth_typical; маска
    DVB-S2 — справочно, в сумму не входит), RIN+ширина линии, шум ФД, усилитель после ФД, EDFA ASE,
    драйверы (OIP3/HD2), дрейф смещения (одна реализация). Параметры источников берутся из cfg.imp, а если
    источник там выключен — значения по умолчанию (RIN -150 дБ/Гц, 1 МГц, NF усилителя 3 дБ,
    EDFA 16 дБ / 5 дБ). Строки "all sources" (все источники сразу) и "sum of 1/SNR" =
    1/SNR_floor + sum(1/SNR_k - 1/SNR_floor).
    Возвращает list dict: name, evm_pct, snr_dB, inv, delta_inv (последние две — линейные).
    """
    plan, sim, comps, ci = cfg.plan, cfg.sim, cfg.comps, cfg.imp
    cn = cn_dB if cn_dB is not None else (ci.input_cn_dB if ci.input_cn_dB is not None else 20.0)
    base = Impairments(er_dB=ci.er_dB, bias_err=ci.bias_err, rx_power_dBm=ci.rx_power_dBm,
                       loss_dB=ci.loss_dB, nf_lna_dB=ci.nf_lna_dB, g_lna_dB=ci.g_lna_dB,
                       nf_driver_dB=ci.nf_driver_dB, input_bw_factor=ci.input_bw_factor,
                       edfa=ci.edfa, edfa_G_dB=ci.edfa_G_dB, edfa_NF_dB=ci.edfa_NF_dB,
                       edfa_lambda_nm=ci.edfa_lambda_nm, edfa_ase=False)
    rin = ci.rin_dBHz if ci.rin_dBHz is not None else -150.0
    lw = ci.linewidth_Hz if ci.linewidth_Hz > 0 else 1e6
    amp = ci.amp_nf_dB if ci.amp_nf_dB is not None else 3.0
    src = {
        "input C/N + NF": dict(input_cn_dB=cn),
        "LO phase noise (synth)": dict(lo_pn=True, lo_profile="synth_typical"),
        "RIN + linewidth": dict(rin_dBHz=rin, linewidth_Hz=lw),
        "PD shot + thermal": dict(pd_noise=True),
        "amplifier after PD": dict(amp_nf_dB=amp),
        "EDFA ASE": dict(edfa=True, edfa_ase=True),
        "drivers (OIP3, HD2)": dict(drivers=True, oip3_dBm=ci.oip3_dBm, hd2_lo_dBc=ci.hd2_lo_dBc,
                                    hd2_sig_dBc=ci.hd2_sig_dBc),
        "bias drift (1 draw)": dict(bias_drift=True, drift_sigma=ci.drift_sigma, drift_tau_s=ci.drift_tau_s),
    }
    ref = "LO phase noise (DVB-S2, ref.)"   # справочно: в сумму и в "all sources" не входит
    rows = []

    def add(name, imp):
        evm, snr = _snr(plan, sim, imp, comps, i)
        rows.append(dict(name=name, evm_pct=evm, snr_dB=snr, inv=10 ** (-snr / 10)))

    add("distortion floor", base)
    for name, kw in src.items():
        add(name, replace(base, **kw))
        if name == "LO phase noise (synth)":
            add(ref, replace(base, lo_pn=True, lo_profile="dvbs2_mask"))
    allkw = {k: v for d in src.values() for k, v in d.items()}
    add("all sources", replace(base, **allkw))
    floor = rows[0]["inv"]
    for r in rows:
        r["delta_inv"] = r["inv"] - floor
    inv_sum = floor + sum(r["delta_inv"] for r in rows[1:-1] if r["name"] != ref)
    rows.append(dict(name="sum of 1/SNR", evm_pct=100 * np.sqrt(inv_sum), snr_dB=-10 * np.log10(inv_sum),
                     inv=inv_sum, delta_inv=inv_sum - floor))
    return rows
