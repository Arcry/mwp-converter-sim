"""Цепочка схемы: лазер -> МЦМ1 -> MUX -> МЦМ2 -> DEMUX -> ФД. Обёртки над OptiCommPy."""
from dataclasses import dataclass, field

import numpy as np
from optic.models.devices import basicLaserModel, mzm, photodiode
from optic.utils import parameters

from .noise import (amp_current_noise, driver_poly, edfa_field, input_noise, lo_table, ou_process,
                    phase_noise)
from .params import Impairments, resolve_sps
from .response import apply, apply_ofilt, band_report

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


def _setup(plan, sim, baseband, n):
    """Сетка модели: (SpS, fs, n, x, t, baseband). baseband=None: CW (K = 1 — тон, K > 1 — мультитон)."""
    from .signals import make_multicarrier_baseband

    sps = resolve_sps(plan, sim)
    fs = sps * sim.Rs * 1e9
    if baseband is None:
        n = n or sim.n_sym * sps
        baseband = make_multicarrier_baseband(plan, sim, sps, cw=True, n=n) if plan.K > 1 else None
    if baseband is not None:
        n = len(baseband.x)
        x = baseband.x
    else:
        x = np.ones(n, complex)
    return sps, fs, n, x, np.arange(n) / fs, baseband


def _input_stage(plan, sim, imp, comps, baseband, x, t, fs, sps):
    """Вход МЦМ2, общий для всех веток: композит -> входной шум -> драйв u2 (на f_mid) -> T2.

    T2 = mzm_field(1, u2): комплексная передаточная функция МЦМ2 (поле на выходе при единичном
    входе); ветка i получает E_e,i = L2 * E_d,i * T2 (МЦМ линеен по полю).
    """
    from .signals import drive_stats, rf_drive

    a_max2, span = 1.0, 0.0
    if hasattr(baseband, "a"):
        a_max2 = float(np.max(baseband.a) ** 2)
        span = max(baseband.f_GHz) - min(baseband.f_GHz)
    x_in = x if imp.input_cn_dB is None else input_noise(x, fs, sps, imp, sim, sim.seed + 200, a_max2, span)
    u2 = rf_drive(x_in, plan.f_mid() * 1e9, sim.beta_m, sim.Vpi, t)
    if imp.drivers:
        u2 = driver_poly(u2, beta_to_drive(sim.beta_m, sim.Vpi), imp.oip3_dBm, imp.hd2_sig_dBc, sim.RL)
    T2 = mzm_field(np.ones(len(t), complex), u2, "quad", sim.Vpi, imp.er_dB, comps.mzm2_rf, fs)
    extra = dict(u2_in=u2, u2_mzm=apply(u2, fs, comps.mzm2_rf), x=x, x_in=x_in,
                 symbols=None if baseband is None else baseband.symbols)
    if hasattr(baseband, "a"):
        extra.update(drive_stats(baseband, sim))
    return T2, extra


def _branch(i, plan, sim, imp, comps, T2, fs, t, bias1):
    """Ветка i: лазер -> МЦМ1 -> MUX -> (x T2, потери) -> [EDFA] -> DEMUX. Возвращает (fields, ase_power)."""
    n = len(t)
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
    Ee = L[2] * Ed * T2
    n_perp = None
    Eg_in = Ee
    if imp.edfa:
        Eg_in, n_perp = edfa_field(Ee, imp.edfa_G_dB, imp.edfa_NF_dB, fs, sim.seed + 300 * (i + 1),
                                   imp.edfa_lambda_nm, imp.edfa_ase)
    Eg = L[3] * apply(Eg_in, fs, comps.wdm)
    cen = None if comps.ofilt.is_none else comps.ofilt.centers(plan, i)
    if cen is not None:                      # оптический фильтр: после DEMUX, до ФД (к E_g и к n_perp)
        Eg = apply_ofilt(Eg, fs, comps.ofilt, cen)
    ase = None
    if n_perp is not None and imp.edfa_ase:
        nf = L[3] * apply(n_perp, fs, comps.wdm)
        if cen is not None:
            nf = apply_ofilt(nf, fs, comps.ofilt, cen)
        ase = np.abs(nf) ** 2
    return dict(a=Ea, d=Ed, e=Ee, g=Eg), ase


def _meta(i, plan, sim, bias1, extra):
    r = plan.routes[i]
    return dict(channel=i, bias1=bias1, SpS=resolve_sps(plan, sim), carrier=r.carrier,
                f_out=plan.line_info(i)[0], sideband=r.sideband, **extra)


def _thin_fields(fields, fs, nmax=100000):
    """Прореженные оптические спектры вместо полных полей: {точка: (f_GHz, дБ)} (max-pooling)."""
    from .spectra import optical_spectrum

    out = {}
    for k, E in fields.items():
        f, p = optical_spectrum(E, fs)
        if len(f) > nmax:
            m = int(np.ceil(len(f) / nmax))
            n = len(f) // m * m
            f, p = f[:n].reshape(-1, m).mean(1), p[:n].reshape(-1, m).max(1)
        out[k] = (f, p)
    return out


def run_channel(i, plan, sim, imp, comps, baseband=None, bias1="null", n=None, extra_power=None):
    """Ветка i: laser -> MZM1 -> MUX -> MZM2 -> DEMUX -> PD. baseband=None — CW (тон или мультитон)."""
    sps, fs, n, x, t, baseband = _setup(plan, sim, baseband, n)
    T2, extra = _input_stage(plan, sim, imp, comps, baseband, x, t, fs, sps)
    fields, ase = _branch(i, plan, sim, imp, comps, T2, fs, t, bias1)
    if ase is not None:
        extra_power = ase if extra_power is None else extra_power + ase
    I = photocurrent(fields["g"], sim, imp, comps.pd, fs, seed=sim.seed + 7 * (i + 1),
                     extra_power=extra_power, amp_seed=sim.seed + 11 * (i + 1))
    return ChannelResult(t, fs, fields, I, band_report(plan, comps, sim.Rs, sim.alpha_eff, imp),
                         _meta(i, plan, sim, bias1, extra))


def run_all(plan, sim, imp, comps, baseband=None, bias1="null", n=None, keep_fields=True, indices=None):
    """Ветки плана (по умолчанию все), считаются последовательно; общий вход (T2) — один раз.

    При неидеальном WDM добавляет помехи соседей (некогерентно). keep_fields=False: вместо
    полных оптических полей в ChannelResult.fields лежат прореженные спектры (экономия памяти).
    """
    sps, fs, n, x, t, baseband = _setup(plan, sim, baseband, n)
    T2, extra = _input_stage(plan, sim, imp, comps, baseband, x, t, fs, sps)
    idx = list(range(plan.N)) if indices is None else list(indices)
    offs = plan.offsets_GHz()
    leak = not comps.wdm.is_ideal and comps.wdm_isolation_dB is not None
    rep = band_report(plan, comps, sim.Rs, sim.alpha_eff, imp)
    iso = 10 ** (-(comps.wdm_isolation_dB or 0) / 10)

    def pd(i, gpow, ase, xt=None):
        # ФД зависит только от мощности: поле g не нужно, достаточно |E_g|^2 (float)
        if ase is not None:
            xt = ase if xt is None else xt + ase
        return photocurrent(np.sqrt(gpow), sim, imp, comps.pd, fs, seed=sim.seed + 7 * (i + 1),
                            extra_power=xt, amp_seed=sim.seed + 11 * (i + 1))

    def reduce(i, fields):
        """Поля ветки для результата: полные (keep_fields) либо прореженные спектры + фрагменты."""
        meta = _meta(i, plan, sim, bias1, extra)
        if keep_fields:
            return fields, meta
        nfr = 8 * sps    # фрагмент полей для вкладки Time (первые 8 символов)
        meta["frag"] = {k: v[:nfr] for k, v in fields.items()}
        return _thin_fields(fields, fs), meta

    out = []
    if not leak:
        for i in idx:
            fields, ase = _branch(i, plan, sim, imp, comps, T2, fs, t, bias1)
            f_out, meta = reduce(i, fields)
            out.append(ChannelResult(t, fs, f_out, pd(i, np.abs(fields["g"]) ** 2, ase), rep, meta))
        return out
    # утечка WDM: ветки считаются последовательно; от каждой хранится только |E_g|^2 (и ASE-мощность)
    gpow, ases, reduced = {}, {}, {}
    for i in range(plan.N):
        fields, ases[i] = _branch(i, plan, sim, imp, comps, T2, fs, t, bias1)
        gpow[i] = np.abs(fields["g"]) ** 2
        if i in idx:
            reduced[i] = reduce(i, fields)
        del fields
    for i in idx:
        xt = sum(iso * gpow[j] for j in range(plan.N)
                 if j != i and abs(abs(offs[i] - offs[j]) - plan.wdm_spacing_GHz) < 1e-9)
        f_out, meta = reduced[i]
        out.append(ChannelResult(t, fs, f_out, pd(i, gpow[i], ases[i], None if np.isscalar(xt) else xt), rep, meta))
    return out


def simulate(plan, sim, imp, comps, i=0, bias1="null"):
    """Прогон ветки i с модулированным входом: возвращает (ChannelResult, MultiBaseband, fs).

    MultiBaseband проксирует несущую 0 (bits/symbols) и хранит композит x; эталон приёмника
    несущей k — tx.ref(k).
    """
    from .signals import make_multicarrier_baseband

    sps = resolve_sps(plan, sim)
    tx = make_multicarrier_baseband(plan, sim, sps)
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
    fi, fc = plan.f_lo_GHz[i], plan.fc_of(i)
    allf = [0.0 if k == "DC" else line_freq_GHz(k, fi, fc) for k in LINE_COEFFS]
    n = coherent_length(fs, allf, n_target) or n_target
    res = run_channel(i, plan, sim, imp, comps, bias1=bias1, n=n)
    freqs = [0.0 if k == "DC" else line_freq_GHz(k, fi, fc) for k in names]
    amps = line_amplitudes(res.I, fs, freqs) / (sim.R * _dbm_to_W(sim.P_dBm) / 4)
    return {k: (f, float(a)) for k, f, a in zip(names, freqs, amps)}
