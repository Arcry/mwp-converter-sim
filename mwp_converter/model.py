"""Цепочка схемы: лазер -> МЦМ1 -> MUX -> МЦМ2 -> DEMUX -> ФД. Обёртки над OptiCommPy."""
from dataclasses import dataclass, field

import numpy as np
from optic.models.devices import basicLaserModel, mzm, photodiode
from optic.utils import parameters

from .noise import (amp_current_noise, driver_poly, edfa_field, input_noise, lo_table, ou_process,
                    phase_noise)
from .params import Impairments, resolve_sps
from .response import apply, band_report

ER_IDEAL = 300.0  # дБ: «бесконечный» ER (по умолчанию в OptiCommPy 60 дБ)


def beta_to_drive(beta, Vpi):
    """Амплитуда драйва u0 = 2 beta Vpi / pi.

    В выводе фаза на плечо pi V/Vpi (beta = pi V0/Vpi); в OptiCommPy аргумент
    pi u/(2 Vpi), отсюда множитель 2.
    """
    return 2 * beta * Vpi / np.pi


def _dbm_to_W(p):
    return 1e-3 * 10 ** (p / 10)


def laser_field(sim, imp, n, fs, seed=None):
    """Поле лазера (огибающая). Идеал — константа sqrt(P).

    Фазовый шум — basicLaserModel (RIN_var=0). RIN — свой вещественный dP:
    E = sqrt(max(P+dP,0)) exp(j phi), sigma^2 = 10^(RIN/10) P^2 Fs/2.
    """
    P = _dbm_to_W(sim.P_dBm)
    if imp.rin_dBHz is None and imp.linewidth_Hz == 0:
        return np.sqrt(P) * np.ones(n, complex)
    p = parameters()
    p.P, p.lw, p.RIN_var, p.Fs, p.Ns, p.seed = sim.P_dBm, imp.linewidth_Hz, 0.0, fs, n, seed
    E = basicLaserModel(p)
    if imp.rin_dBHz is not None:
        sigma = np.sqrt(10 ** (imp.rin_dBHz / 10) * P ** 2 * fs / 2)
        rng = np.random.default_rng(None if seed is None else seed + 73)
        E = E * np.sqrt(np.maximum(P + rng.normal(0, sigma, n), 0) / P)
    return E


def mzm_field(E, u, bias, Vpi, er_dB=None, response=None, fs=None, bias_err=0.0):
    """Поле на выходе МЦМ (mzm из OptiCommPy).

    bias: 'null' (Vb=+Vpi), 'quad' (Vb=+Vpi/2), 'max' (Vb=0) или число (В).
    er_dB=None — ER=300 дБ. response — АЧХ РЧ-входа, применяется к драйву.
    bias_err — уход смещения, доля Vpi (Vb += bias_err*Vpi).
    """
    if response is not None:
        u = apply(u, fs, response)
    Vb = {"null": Vpi, "quad": Vpi / 2, "max": 0.0}[bias] if isinstance(bias, str) else float(bias)
    Vb += bias_err * Vpi
    p = parameters()
    p.Vpi, p.Vb, p.ER = Vpi, Vb, ER_IDEAL if er_dB is None else er_dB
    return mzm(E, u, p)


def photocurrent(E, sim, imp, pd_resp, fs, seed=None, extra_power=None, amp_seed=None):
    """Фототок. extra_power — добавка оптической мощности (помеха WDM), Вт.

    Идеал — photodiode(ideal=True); с шумами — ideal=False, без встроенного ФНЧ
    (bandwidthLimitation=False, B=fs/2); полоса ФД — свой apply(I, pd_resp). Затем (если
    imp.amp_nf_dB) добавляется шум усилителя после ФД (noise.amp_current_noise).
    """
    Pin = np.abs(E) ** 2
    if extra_power is not None:
        Pin = Pin + extra_power
    E = np.sqrt(Pin).astype(complex)
    if imp.rx_power_dBm is not None:
        E = E * np.sqrt(_dbm_to_W(imp.rx_power_dBm) / np.mean(Pin))
    p = parameters()
    p.R, p.RL, p.Tc, p.seed = sim.R, sim.RL, sim.Tc, seed
    if not imp.pd_noise:
        p.ideal = True
    else:
        p.ideal, p.Fs, p.B = False, fs, fs / 2
        p.bandwidthLimitation, p.shotNoise, p.thermalNoise = False, True, True
    I = apply(photodiode(E, p), fs, pd_resp)
    if imp.amp_nf_dB is not None:
        I = I + amp_current_noise(len(I), fs, sim.RL, imp.amp_nf_dB, sim.Tc, amp_seed)
    return I


@dataclass
class ChannelResult:
    """Результат расчёта канала."""

    t: np.ndarray
    fs: float
    fields: dict
    I: np.ndarray
    band_report: list
    meta: dict = field(default_factory=dict)


def _chain(i, plan, sim, imp, comps, baseband, bias1, n):
    from .signals import rf_drive

    sps = resolve_sps(plan, sim)
    fs = sps * sim.Rs * 1e9
    if baseband is not None:
        n = len(baseband.x)
        x = baseband.x
    else:
        n = n or sim.n_sym * sps
        x = np.ones(n, complex)
    t = np.arange(n) / fs
    fi = plan.f_lo_GHz[i] * 1e9
    E0 = laser_field(sim, imp, n, fs, seed=sim.seed + 1000 * (i + 1))
    phi = phase_noise(lo_table(imp), n, fs, sim.seed + 500 * (i + 1)) if imp.lo_pn else 0.0
    a1 = beta_to_drive(sim.beta1, sim.Vpi)
    u1 = a1 * np.sin(2 * np.pi * fi * t + phi)
    if imp.drivers:
        u1 = driver_poly(u1, a1, imp.oip3_dBm, imp.hd2_lo_dBc, sim.RL)
    if imp.bias_drift:
        u1 = u1 + sim.Vpi * ou_process(imp.drift_sigma, imp.drift_tau_s, n, fs,
                                       sim.seed + 900 * (i + 1) + imp.drift_ens)
    L = [10 ** (-d / 20) for d in imp.loss_dB]   # потери по полю: MZM1, MUX, MZM2, DEMUX
    Ea = L[0] * mzm_field(E0, u1, bias1, sim.Vpi, imp.er_dB, comps.mzm1_rf, fs, imp.bias_err)
    Ed = L[1] * apply(Ea, fs, comps.wdm)
    x_in = x if imp.input_cn_dB is None else input_noise(x, fs, sps, imp, sim, sim.seed + 200)
    u2 = rf_drive(x_in, plan.fc_GHz * 1e9, sim.beta_m, sim.Vpi, t)
    if imp.drivers:
        u2 = driver_poly(u2, beta_to_drive(sim.beta_m, sim.Vpi), imp.oip3_dBm, imp.hd2_sig_dBc, sim.RL)
    extra = dict(u2_in=u2, u2_mzm=apply(u2, fs, comps.mzm2_rf), x=x, x_in=x_in,
                 symbols=None if baseband is None else baseband.symbols, ase_power=None)
    Ee = L[2] * mzm_field(Ed, u2, "quad", sim.Vpi, imp.er_dB, comps.mzm2_rf, fs)
    n_perp = None
    Eg_in = Ee
    if imp.edfa:
        Eg_in, n_perp = edfa_field(Ee, imp.edfa_G_dB, imp.edfa_NF_dB, fs, sim.seed + 300 * (i + 1),
                                   imp.edfa_lambda_nm, imp.edfa_ase)
    Eg = L[3] * apply(Eg_in, fs, comps.wdm)
    if n_perp is not None and imp.edfa_ase:
        extra["ase_power"] = np.abs(L[3] * apply(n_perp, fs, comps.wdm)) ** 2
    return t, fs, dict(a=Ea, d=Ed, e=Ee, g=Eg), extra


def run_channel(i, plan, sim, imp, comps, baseband=None, bias1="null", n=None, extra_power=None):
    """Канал i: laser -> MZM1 -> MUX -> MZM2 -> DEMUX -> PD. baseband=None — CW-тон (x=1)."""
    t, fs, fields, extra = _chain(i, plan, sim, imp, comps, baseband, bias1, n)
    ase = extra.pop("ase_power")
    if ase is not None:
        extra_power = ase if extra_power is None else extra_power + ase
    I = photocurrent(fields["g"], sim, imp, comps.pd, fs, seed=sim.seed + 7 * (i + 1),
                     extra_power=extra_power, amp_seed=sim.seed + 11 * (i + 1))
    return ChannelResult(t, fs, fields, I, band_report(plan, comps, sim.Rs, sim.alpha_eff, imp),
                         dict(channel=i, bias1=bias1, SpS=resolve_sps(plan, sim), **extra))


def run_all(plan, sim, imp, comps, baseband=None, bias1="null", n=None):
    """Все каналы; при неидеальном WDM добавляет помехи соседей (некогерентно)."""
    chains = [_chain(i, plan, sim, imp, comps, baseband, bias1, n) for i in range(len(plan.f_lo_GHz))]
    offs = plan.offsets_GHz()
    leak = not comps.wdm.is_ideal and comps.wdm_isolation_dB is not None
    rep = band_report(plan, comps, sim.Rs, sim.alpha_eff, imp)
    out = []
    for i, (t, fs, fields, chain_meta) in enumerate(chains):
        xt = None
        if leak:
            xt = sum(10 ** (-comps.wdm_isolation_dB / 10) * np.abs(chains[j][2]["g"]) ** 2
                     for j in range(len(chains))
                     if j != i and abs(abs(offs[i] - offs[j]) - plan.wdm_spacing_GHz) < 1e-9)
            xt = xt if not np.isscalar(xt) else None
        ase = chain_meta.pop("ase_power")
        if ase is not None:
            xt = ase if xt is None else xt + ase
        I = photocurrent(fields["g"], sim, imp, comps.pd, fs, seed=sim.seed + 7 * (i + 1),
                         extra_power=xt, amp_seed=sim.seed + 11 * (i + 1))
        out.append(ChannelResult(t, fs, fields, I, rep, dict(channel=i, bias1=bias1,
                                                              SpS=resolve_sps(plan, sim), **chain_meta)))
    return out


def simulate(plan, sim, imp, comps, i=0, bias1="null"):
    """QPSK/PSK-прогон канала i: возвращает (ChannelResult, Baseband, fs)."""
    from .signals import make_psk_baseband

    sps = resolve_sps(plan, sim)
    tx = make_psk_baseband(sim.M, sim.n_sym, sps, sim.alpha, sim.seed, sim.pulse)
    return run_channel(i, plan, sim, imp, comps, baseband=tx, bias1=bias1), tx, sps * sim.Rs * 1e9


def measure_lines(i, plan, sim, comps, names, bias1="null", imp=None, n_target=2 ** 16):
    """CW-прогон канала i и амплитуды линий names в единицах R*P0/4.

    Возвращает dict имя -> (f_GHz, амплитуда). Шумы выключены (imp по умолчанию идеальный).
    Длина записи подбирается так, чтобы линии попали в бины (иначе flat-top).
    """
    from .analytic import LINE_COEFFS, line_freq_GHz
    from .spectra import coherent_length, line_amplitudes

    imp = imp or Impairments()
    fs = resolve_sps(plan, sim) * sim.Rs * 1e9
    fi, fc = plan.f_lo_GHz[i], plan.fc_GHz
    allf = [0.0 if k == "DC" else line_freq_GHz(k, fi, fc) for k in LINE_COEFFS]
    n = coherent_length(fs, allf, n_target) or n_target
    res = run_channel(i, plan, sim, imp, comps, bias1=bias1, n=n)
    freqs = [0.0 if k == "DC" else line_freq_GHz(k, fi, fc) for k in names]
    amps = line_amplitudes(res.I, fs, freqs) / (sim.R * _dbm_to_W(sim.P_dBm) / 4)
    return {k: (f, float(a)) for k, f, a in zip(names, freqs, amps)}
