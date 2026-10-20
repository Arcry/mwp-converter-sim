"""Выходной полосовой фильтр (BPF) канала и таблица подавления паразитных линий."""
import numpy as np
from scipy import signal

from .response import bpf_att_dB, bpf_band
from .spectra import band_power


def bpf_sos(fs, f0_GHz, bw_GHz, order=4, kind="butter", ripple_dB=1.0, stop_dB=40.0):
    """Цифровой аналог полосового фильтра Баттерворта или Чебышёва I (sos)."""
    nyq = fs / 2e9
    lo = max(f0_GHz - bw_GHz / 2, 1e-3 * nyq)
    hi = min(f0_GHz + bw_GHz / 2, 0.999 * nyq)
    wn = [lo / nyq, hi / nyq]
    if kind == "cheby1":
        return signal.cheby1(order, ripple_dB, wn, btype="band", output="sos")
    if kind == "ellip":
        return signal.ellip(order, ripple_dB, stop_dB, wn, btype="band", output="sos")
    return signal.butter(order, wn, btype="band", output="sos")


def bandpass(I, fs, f0_GHz, bw_GHz, order=4, kind="butter", ripple_dB=1.0, stop_dB=40.0):
    """Каузальная фильтрация (sosfilt): учитывает групповую задержку и фазовые искажения."""
    return signal.sosfilt(bpf_sos(fs, f0_GHz, bw_GHz, order, kind, ripple_dB, stop_dB), I)


def bpf_for(plan, sim, bpf, i):
    """(центр, полоса) BPF канала i, ГГц."""
    return bpf_band(plan, sim.Rs, sim.alpha_eff, bpf, i)


def apply_bpf(I, fs, plan, sim, bpf, i):
    """Фототок канала i после BPF."""
    f0, bw = bpf_for(plan, sim, bpf, i)
    return bandpass(I, fs, f0, bw, bpf.order, bpf.kind, bpf.ripple_dB, bpf.stop_dB)


def bpf_response(fs, plan, sim, bpf, i, f_GHz):
    """Комплексная АЧХ цифрового BPF канала i на частотах f_GHz."""
    f0, bw = bpf_for(plan, sim, bpf, i)
    sos = bpf_sos(fs, f0, bw, bpf.order, bpf.kind, bpf.ripple_dB, bpf.stop_dB)
    _, h = signal.sosfreqz(sos, worN=np.asarray(f_GHz, float) * 1e9, fs=fs)
    return h


def group_delay_ns(fs, plan, sim, bpf, i, f_GHz):
    """Групповая задержка BPF, нс (численное дифференцирование фазы)."""
    f = np.asarray(f_GHz, float)
    h = bpf_response(fs, plan, sim, bpf, i, f)
    ph = np.unwrap(np.angle(h))
    return -np.gradient(ph, 2 * np.pi * f * 1e9) * 1e9


def suppression_table(I, I_bpf, fs, plan, sim, i, nperseg=2 ** 14):
    """Паразитные линии в пределах ±f_out от выхода: уровень до и после BPF, dBc к полезной.

    Мощность линии — интеграл PSD в окне (1+alpha_eff)*Rs вокруг её частоты.
    Строки: line, f_GHz, before_dBc, after_dBc, suppression_dB.
    """
    f, p0 = signal.welch(I, fs=fs, nperseg=min(nperseg, len(I)))
    _, p1 = signal.welch(I_bpf, fs=fs, nperseg=min(nperseg, len(I)))
    f = f / 1e9
    bw = (1 + sim.alpha_eff) * sim.Rs
    fo = plan.line_info(i)[0]
    ref0, ref1 = band_power(f, p0, fo, bw), band_power(f, p1, fo, bw)
    rows = []
    for name, fl in sorted(plan.spurs(i).items(), key=lambda kv: kv[1]):
        if abs(fl - fo) > fo or fl >= fs / 2e9 or abs(fl - fo) < 1e-9:
            continue
        b = 10 * np.log10(band_power(f, p0, fl, bw) / ref0)
        a = 10 * np.log10(band_power(f, p1, fl, bw) / ref1)
        rows.append(dict(line=name, f_GHz=round(fl, 3), before_dBc=round(b, 1), after_dBc=round(a, 1),
                         suppression_dB=round(b - a, 1)))
    return rows


def required_selectivity(f0_GHz, bw_GHz, f_adj_GHz):
    """Нормированная расстройка прототипа x = |(f^2 - f0^2)/(f*B)| для соседа на частоте f_adj."""
    return abs((f_adj_GHz ** 2 - f0_GHz ** 2) / (f_adj_GHz * bw_GHz))


def required_order(kind, f0_GHz, bw_GHz, f_adj_GHz, A_req_dB=30.0, ripple_dB=1.0, stop_dB=40.0, nmax=20):
    """Наименьший порядок прототипа, дающий подавление >= A_req_dB на частоте соседа.

    Для ellip нужно stop_dB >= A_req_dB. Возвращает None, если порядка <= nmax не хватает.
    """
    for n in range(1, nmax + 1):
        if bpf_att_dB(f_adj_GHz, f0_GHz, bw_GHz, n, kind, ripple_dB, stop_dB) >= A_req_dB:
            return n
    return None


def required_Q(f0_GHz, bw_GHz):
    """Добротность Q = f_out / B."""
    return f0_GHz / bw_GHz
