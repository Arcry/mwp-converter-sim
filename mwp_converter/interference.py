"""Помехи многонесущего входа: C/I (прогоны 1-3: ACI/IMD), NPR, свип драйва с оптимумом C/(N+I)."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy import signal

from .bpf import apply_bpf
from .model import run_channel
from .params import Impairments, resolve_sps
from .signals import make_multicarrier_baseband, measure
from .spectra import band_power  # noqa: F401  (реэкспорт)


def _psd(x, fs, sps):
    f, p = signal.welch(x, fs=fs, nperseg=min(16 * sps, len(x)))
    return f / 1e9, p


def static_imp(imp):
    """Неидеальности без случайных источников (RIN, ширина линии, шумы ФД, входной шум, шум усилителя,
    ASE, фазовый шум гетеродина, дрейф смещения). Детерминированные (драйверы, rx_power_dBm, ER,
    bias_err, потери, усиление EDFA) сохраняются: для прогонов 0 и 3."""
    return replace(imp, rin_dBHz=None, linewidth_Hz=0.0, pd_noise=False, input_cn_dB=None, amp_nf_dB=None,
                   edfa_ase=False, lo_pn=False, bias_drift=False)


def _slot_power(cfg, i, sim, imp, zero):
    """Мощность в окне выходной линии ветки i после BPF; композит без несущей zero (None — все)."""
    plan, comps = cfg.plan, cfg.comps
    sps = resolve_sps(plan, sim)
    f_out = plan.line_info(i)[0]
    bw = (1 + sim.alpha_eff) * sim.Rs
    mb = make_multicarrier_baseband(plan, sim, sps, zero=zero)
    res = run_channel(i, plan, sim, imp, comps, baseband=mb)
    Ib = apply_bpf(res.I, res.fs, plan, sim, comps.bpf, i)
    f, p = _psd(Ib, res.fs, sps)
    return band_power(f, p, f_out, bw), res, mb, Ib


def ci_runs(cfg, i, imp=None, split=True):
    """C/I ветки i по прогонам 0-3.

    Прогон 1: все несущие -> P1 в окне выходной линии после BPF, EVM/BER этой несущей.
    Прогон 2: a_k -> 0 без перенормировки остальных -> P2 (вся мощность в слоте без несущей k).
    Прогон 0 (split=True): все a_k = 0, только линии гетеродина (например 4f_i) -> P0 (без шумов).
    Прогон 3 (split=True): прогон 2 при beta_3 = max(beta_m/10, 0.05) без шумов -> P3; ACI =
    max(P3 - P0, 0)*(beta_m/beta_3)^2 (линейная часть ~ beta^2 без немодулированных линий гетеродина;
    нижняя граница 0,05 — иначе P3 упирается в числовой пол ~1e-22 фототока).
    I_mod = P2 - P0, IMD = max(I_mod - ACI, 0). C = P1 - P2.
    Возвращает dict: carrier, f_out, P0..P3, ACI, IMD, ci_dB (C/I_mod), ci_aci_dB, ci_imd_dB,
    ci_lo_dB (C/LO-spur, дБ; C/IMD, C/LO ограничены 100 дБ), EVM/BER (evm_pct, evm_eq_pct, ber_mc, n_err,
    snr_evm_dB).
    """
    plan, sim = cfg.plan, cfg.sim
    imp = cfg.imp if imp is None else imp
    k = plan.routes[i].carrier
    f_out, sb = plan.line_info(i)
    P1, res, mb, Ib = _slot_power(cfg, i, sim, imp, None)
    m = measure(Ib, res.fs, f_out, sb, sim, mb, carrier=k)
    P2 = _slot_power(cfg, i, sim, imp, k)[0]
    C = max(P1 - P2, 1e-300)
    cap = lambda x: float(10 * np.log10(C / max(x, C * 1e-10)))   # noqa: E731  (потолок 100 дБ)
    out = dict(carrier=k, f_out=f_out, P1=P1, P2=P2, evm_pct=m["evm_pct"], evm_eq_pct=m["evm_eq_pct"],
               ber_mc=m["ber_mc"], n_err=m["n_err"], snr_evm_dB=m["snr_evm_dB"], rx_norm=m["rx_norm"])
    if split:
        si = static_imp(imp)
        P0 = _slot_power(cfg, i, sim, si, "all")[0]
        b3 = max(sim.beta_m / 10, 0.05)     # ниже ~0,05 мощность в слоте тонет в числовом шуме (~1e-22)
        P3 = _slot_power(cfg, i, replace(sim, beta_m=b3), si, k)[0]
        aci = max(P3 - P0, 0.0) * (sim.beta_m / b3) ** 2
        imod = max(P2 - P0, 0.0)
        imd = max(imod - aci, 0.0)
        out.update(P0=P0, P3=P3, ACI=aci, IMD=imd, ci_dB=cap(imod), ci_aci_dB=cap(aci), ci_imd_dB=cap(imd),
                   ci_lo_dB=cap(P0))
    else:
        out["ci_dB"] = float(10 * np.log10(C / max(P2, 1e-300)))
    return out


def noise_load(plan, sim, sps, k, n, seed=7):
    """Гауссов шум в полосе band_GHz() (композитная огибающая) с вырезом шириной R_s на несущей k.

    Нормирован mean|x|^2 = 1, поэтому beta_m задаётся по RMS как у обычного входа.
    """
    fs = sps * sim.Rs * 1e9
    rng = np.random.default_rng(seed)
    X = np.fft.fft(rng.normal(size=n) + 1j * rng.normal(size=n))
    f = np.fft.fftfreq(n, 1 / fs) / 1e9 + plan.f_mid()
    lo, hi = plan.band_GHz((1 + sim.alpha_eff) * sim.Rs)
    fck = plan.carriers[k].f_GHz
    mask = (f >= lo) & (f <= hi) & (np.abs(f - fck) > sim.Rs / 2)
    x = np.fft.ifft(X * mask)
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def npr(cfg, i, beta_m=None, imp=None, seed=7):
    """NPR ветки i (дБ): шумовая нагрузка полосы с вырезом на несущей ветки, измерение — по фототоку
    до BPF. NPR = средняя PSD в соседних окнах (отстройки 0,75-1,25 R_s от несущей, внутри
    нагруженной полосы) / средняя PSD внутри выреза (±R_s/4), всё перенесено на выход ветки.
    """
    plan, comps = cfg.plan, cfg.comps
    sim = cfg.sim if beta_m is None else replace(cfg.sim, beta_m=beta_m)
    imp = Impairments() if imp is None else imp
    k = plan.routes[i].carrier
    sps = resolve_sps(plan, sim)
    x = noise_load(plan, sim, sps, k, sim.n_sym * sps, seed)
    res = run_channel(i, plan, sim, imp, comps, baseband=SimpleNamespace(x=x, symbols=None))
    f, p = _psd(res.I, res.fs, sps)
    fi, fck, rs = plan.f_lo_GHz[i], plan.carriers[k].f_GHz, sim.Rs
    lo, hi = plan.band_GHz((1 + sim.alpha_eff) * rs)

    def to_out(pf):                      # частота входной полосы -> выход ветки
        return abs(2 * fi - pf) if plan.routes[i].sideband == "lower" else 2 * fi + pf

    def win_power(a, b):                 # окно [a, b] входной полосы -> средняя PSD на выходе
        o = sorted((to_out(a), to_out(b)))
        m = (f >= o[0]) & (f <= o[1])
        return float(p[m].mean())

    notch = win_power(fck - rs / 4, fck + rs / 4)
    adj = [win_power(fck + s * a, fck + s * b) for s in (-1, 1) for a, b in ((0.75 * rs, 1.25 * rs),)
           if lo <= fck + s * 1.25 * rs <= hi]
    if not adj:
        raise ValueError("нет соседних окон внутри нагруженной полосы")
    return float(10 * np.log10(np.mean(adj) / notch))


def drive_sweep(cfg, i, betas, with_npr=False, aci_ref_beta=0.5):
    """Свип beta_m: EVM (без и с эквалайзером), C/(N+I), C/IMD, C/ACI.

    C/(N+I) = C / P2 (P2 — вся мощность в слоте без несущей, включая шумы и линии гетеродина).
    ACI(beta) = ACI_ref (beta/beta_ref)^2: линейная часть, поэтому прогоны 0 и 3 делаются один раз.
    IMD = max(P2 - P0 - ACI, 0). Возвращает (rows, beta_opt): beta* — максимум C/(N+I) на сетке.
    """
    sim0 = cfg.sim
    ref = ci_runs(replace(cfg, sim=replace(sim0, beta_m=aci_ref_beta)), i)
    aci_ref, P0 = ref["ACI"], ref["P0"]
    rows = []
    for b in betas:
        c = replace(cfg, sim=replace(sim0, beta_m=float(b)))
        r = ci_runs(c, i, split=False)
        C = max(r["P1"] - r["P2"], 1e-300)
        aci = aci_ref * (b / aci_ref_beta) ** 2
        imd = max(r["P2"] - P0 - aci, 0.0)
        row = dict(beta=float(b), cni_dB=r["ci_dB"], evm_pct=r["evm_pct"], evm_eq_pct=r["evm_eq_pct"],
                   ber_mc=r["ber_mc"], ci_aci_dB=float(10 * np.log10(C / max(aci, C * 1e-10))),
                   ci_imd_dB=float(10 * np.log10(C / max(imd, C * 1e-10))))
        if with_npr:
            row["npr_dB"] = npr(c, i)
        rows.append(row)
    best = max(rows, key=lambda r: r["cni_dB"])["beta"]
    return rows, best
