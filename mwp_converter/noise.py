"""Источники шума этапа 1: входной шум (C/N, LNA, Фрис), EDFA с ASE, шум усилителя после ФД."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
K_B = 1.380649e-23


def nf_cascade(nf1_dB, g1_dB, nf2_dB):
    """Коэффициент шума каскада двух ступеней (формула Фрииса), дБ: F = F1 + (F2-1)/G1."""
    f1, f2, g1 = 10 ** (nf1_dB / 10), 10 ** (nf2_dB / 10), 10 ** (g1_dB / 10)
    return float(10 * np.log10(f1 + (f2 - 1) / g1))


def input_noise(x, fs, sps, imp, sim, seed, a_max2=1.0, span_GHz=0.0):
    """Огибающая на входе МЦМ2 с шумом: x + n (C/N на входе LNA в полосе Rs), затем входной BPF.

    C/N_eff = C/N - NF_total (Фрис по LNA и драйверу); дисперсия комплексного белого шума на
    отсчёт sigma^2 = SpS / 10^(C/N_eff/10) при mean|x|^2 = 1. Входной BPF — Баттерворт 4-го
    порядка (фазово-нулевой, sosfiltfilt), срез bw_factor*(1+alpha_eff)*Rs/2 + span/2. При K > 1
    C/N задаётся по самой сильной несущей (a_max2 — её мощность в композите), span — ширина
    полосы между крайними несущими (ГГц).
    """
    nf = nf_cascade(imp.nf_lna_dB, imp.g_lna_dB, imp.nf_driver_dB)
    cn_eff = imp.input_cn_dB - nf
    sigma2 = sps * a_max2 / 10 ** (cn_eff / 10)
    rng = np.random.default_rng(seed)
    n = np.sqrt(sigma2 / 2) * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
    cut = imp.input_bw_factor * (1 + sim.alpha_eff) * sim.Rs * 1e9 / 2 + span_GHz * 1e9 / 2
    sos = butter(4, cut / (fs / 2), output="sos")
    return sosfiltfilt(sos, x + n)


def n_ase(G_dB, NF_dB, lambda_nm=1550.0):
    """Спектральная плотность ASE одной поляризации, Вт/Гц: (G-1) n_sp h nu, n_sp = (GF-1)/(2(G-1))."""
    G, F = 10 ** (G_dB / 10), 10 ** (NF_dB / 10)
    # (G-1) n_sp = (G F - 1)/2: без деления на G-1 (корректно и при G = 1)
    return (G * F - 1) / 2 * H_PLANCK * C_LIGHT / (lambda_nm * 1e-9)


def edfa_field(E, G_dB, NF_dB, fs, seed, lambda_nm=1550.0, ase=True):
    """EDFA: возвращает (E_out, n_perp).

    E_out = sqrt(G)*E + n_par; n_par, n_perp ~ CN(0, N_ASE*fs) (две поляризации). При ase=False
    E_out = sqrt(G)*E ровно, n_perp = 0. n_perp в сигнальной поляризации не интерферирует, поэтому
    добавляется к фототоку по мощности (extra_power).
    """
    g = np.sqrt(10 ** (G_dB / 10))
    if not ase:
        return g * E, np.zeros(len(E), complex)
    sigma2 = n_ase(G_dB, NF_dB, lambda_nm) * fs
    rng = np.random.default_rng(seed)

    def cn():
        return np.sqrt(sigma2 / 2) * (rng.normal(size=len(E)) + 1j * rng.normal(size=len(E)))

    return g * E + cn(), cn()


def amp_current_noise(n, fs, RL, NF_dB, Tc, seed):
    """Белый ток шума усилителя после ФД: sigma^2 = (F-1)*4kT*(fs/2)/RL (на отсчёт)."""
    F = 10 ** (NF_dB / 10)
    sigma2 = (F - 1) * 4 * K_B * (Tc + 273.15) * (fs / 2) / RL
    return np.sqrt(sigma2) * np.random.default_rng(seed).normal(size=n)


# ---------- этап 2: фазовый шум гетеродина, драйверы, дрейф смещения ----------
LO_PRESETS = {
    "synth_typical": [[1e3, -75], [1e4, -85], [1e5, -95], [1e6, -115], [1e7, -135], [1e8, -145]],
    # ориентировочная маска DVB-S2 (approx.)
    "dvbs2_mask": [[1e3, -73], [1e4, -93], [1e5, -103], [1e6, -114], [1e7, -120], [1e8, -125]],
}


def lo_table(imp):
    """Таблица L(f) [[f_Hz, dBc/Hz], ...] по профилю imp.lo_profile (пресет или 'table')."""
    return imp.lo_table if imp.lo_profile == "table" else LO_PRESETS[imp.lo_profile]


def interp_L(table, f):
    """L(f), дБн/Гц: линейная в дБ по lg f; за краями таблицы — крайние значения."""
    t = np.asarray(table, float)
    return np.interp(np.log10(np.maximum(f, 1e-30)), np.log10(t[:, 0]), t[:, 1])


def phase_noise(table, n, fs, seed):
    """Реализация фазового шума phi(t), рад, по таблице L(f).

    S_phi = 2*10^(L/10) (односторонняя СПМ фазы). Спектр вещественного белого шума умножается на
    sqrt(S_phi*fs/2), затем irfft. Разрешение по отстройке fs/n (ниже — не разрешается), постоянная
    составляющая обнулена.
    """
    f = np.fft.rfftfreq(n, 1 / fs)
    S = 2 * 10 ** (interp_L(table, f) / 10)
    X = np.fft.rfft(np.random.default_rng(seed).normal(size=n)) * np.sqrt(S * fs / 2)
    X[0] = 0
    return np.fft.irfft(X, n=n)


def driver_poly(u, A_nom, oip3_dBm, hd2_dBc, RL=50.0):
    """Нелинейность драйвера: y = u + a2 u^2 + a3 u^3, затем вычитается среднее.

    A_IP3 = sqrt(2 RL 10^((OIP3-30)/10)), a3 = -4/(3 A_IP3^2); a2 = 2*10^(HD2/20)/A_nom,
    где A_nom — номинальная амплитуда драйва (2 beta Vpi / pi).
    """
    a_ip3 = np.sqrt(2 * RL * 10 ** ((oip3_dBm - 30) / 10))
    a3 = -4 / (3 * a_ip3 ** 2)
    a2 = 2 * 10 ** (hd2_dBc / 20) / A_nom
    y = u + a2 * u ** 2 + a3 * u ** 3
    return y - y.mean()


def ou_process(sigma, tau_s, n, fs, seed):
    """Процесс Орнштейна-Уленбека (доля Vpi): rho = exp(-dt/tau), b0 ~ N(0, sigma^2), стационарный."""
    from scipy.signal import lfilter

    rho = np.exp(-1 / (fs * tau_s))
    rng = np.random.default_rng(seed)
    b0 = sigma * rng.normal()
    w = sigma * np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    w[0] = 0.0
    return lfilter([1.0], [1.0, -rho], w, zi=[b0])[0]
