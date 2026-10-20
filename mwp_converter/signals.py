"""PSK+RRC baseband, РЧ-драйв, приёмник линии и метрики."""
from collections import namedtuple

import numpy as np
from optic.comm.metrics import theoryBER
from optic.comm.modulation import demodulateGray, modulateGray
from optic.dsp.core import firFilter, pnorm, pulseShape
from optic.utils import parameters
from scipy.signal import butter, sosfiltfilt

from .model import beta_to_drive

Baseband = namedtuple("Baseband", "bits symbols x")


class MultiBaseband:
    """Композит входной полосы из K несущих (общий драйв МЦМ2).

    carriers — Baseband каждой несущей (без сдвига частоты), a — амплитудные веса (sum a_k^2 = 1,
    если не обнулялись), f_GHz — частоты несущих, f_mid — центр полосы (ГГц), x — композит
    sum a_k x_k exp(j 2 pi (f_ck - f_mid) t). Прокси bits/symbols — несущая 0 (совместимость).
    """

    def __init__(self, carriers, a, f_GHz, f_mid, x, M):
        self.carriers, self.a, self.f_GHz, self.f_mid, self.x, self.M = carriers, a, list(f_GHz), f_mid, x, list(M)

    bits = property(lambda self: self.carriers[0].bits)
    symbols = property(lambda self: self.carriers[0].symbols)

    def ref(self, k):
        """Baseband несущей k (эталон приёмника: символы и огибающая без сдвига и веса)."""
        return self.carriers[k]


def _rrc(SpS, alpha):
    p = parameters()
    p.pulseType = "rrc"
    p.SpS = SpS
    p.nFilterTaps = 16 * SpS
    p.rollOff = alpha
    return pulseShape(p)


def _mf(SpS, alpha, pulse):
    """Согласованный фильтр: RRC или интегрирование на символе (boxcar) для rect."""
    return _rrc(SpS, alpha) if pulse == "rrc" else np.ones(SpS) / SpS


def make_psk_baseband(M, n_sym, SpS, alpha, seed, pulse="rrc"):
    """PSK-огибающая (pulse: rrc | rect). Возвращает (bits, symbols, x), mean|x|^2 = 1.

    rect — прямоугольные импульсы без формирующего фильтра (как в статье), спектр sinc^2.
    """
    rng = np.random.default_rng(seed)
    k = int(np.log2(M))
    bits = rng.integers(0, 2, n_sym * k)
    symbols = pnorm(modulateGray(bits, M, "psk"))
    if pulse == "rect":
        return Baseband(bits, symbols, np.repeat(symbols, SpS))
    up = np.zeros(n_sym * SpS, complex)
    up[::SpS] = symbols
    x = pnorm(firFilter(_rrc(SpS, alpha), up))
    return Baseband(bits, symbols, x)


def carrier_weights(carriers):
    """Амплитудные веса несущих a_k = sqrt(10^(L_k/10) / sum_j 10^(L_j/10)) (sum a_k^2 = 1)."""
    lv = np.array([10 ** (c.level_dB / 10) for c in carriers])
    return np.sqrt(lv / lv.sum())


def make_multicarrier_baseband(plan, sim, sps, cw=False, n=None, zero=None):
    """Композит из K несущих плана: x = sum a_k x_k exp(j 2 pi (f_ck - f_mid) t), без перенормировки.

    a_k = sqrt(10^(L_k/10) / sum_j 10^(L_j/10)). У каждой несущей свой seed (sim.seed +
    seed_offset) и порядок PSK (Carrier.M или sim.M). cw=True — немодулированные тона (x_k = 1,
    n отсчётов). zero=k — обнулить a_k без перенормировки остальных (прогон 2 для C/I), zero='all' — все a_k = 0
    (остаются только линии гетеродина: прогон 0).
    При K = 1 композит совпадает с make_psk_baseband.
    """
    cs = plan.carriers
    a = carrier_weights(cs)
    f_mid = plan.f_mid()
    fs = sps * sim.Rs * 1e9
    bbs, Ms = [], []
    for c in cs:
        M = c.M or sim.M
        Ms.append(M)
        if cw:
            bbs.append(Baseband(None, None, np.ones(n, complex)))
        else:
            bbs.append(make_psk_baseband(M, sim.n_sym, sps, sim.alpha, sim.seed + c.seed_offset, sim.pulse))
    n_s = len(bbs[0].x)
    t = np.arange(n_s) / fs
    aw = a.copy()
    if zero == "all":
        aw[:] = 0.0
    elif zero is not None:
        aw[zero] = 0.0
    x = np.zeros(n_s, complex)
    for ak, c, bb in zip(aw, cs, bbs):
        x = x + ak * bb.x * np.exp(2j * np.pi * (c.f_GHz - f_mid) * 1e9 * t)
    return MultiBaseband(bbs, aw, [c.f_GHz for c in cs], f_mid, x, Ms)


def drive_stats(mb, sim):
    """Справочные характеристики драйва МЦМ2: beta_k = beta_m a_k, PAPR композита, beta_peak, IBO.

    IBO = 20 lg((pi/2)/beta_m) — запас до beta = pi/2 (справочно).
    """
    p = np.abs(mb.x) ** 2
    if p.mean() == 0:
        return dict(beta_k=[0.0] * len(mb.a), papr_dB=0.0, beta_peak=0.0,
                    ibo_dB=float(20 * np.log10((np.pi / 2) / sim.beta_m)))
    return dict(beta_k=[sim.beta_m * float(ak) for ak in mb.a],
                papr_dB=float(10 * np.log10(p.max() / p.mean())),
                beta_peak=float(sim.beta_m * np.sqrt(p.max())),
                ibo_dB=float(20 * np.log10((np.pi / 2) / sim.beta_m)))


def carrier_M(tx, sim, k=0):
    """Порядок PSK несущей k (для MultiBaseband — свой, иначе sim.M)."""
    return tx.M[k] if hasattr(tx, "M") else sim.M


def rf_drive(x, fc, beta_m, Vpi, t):
    """u2 = u0m * Im{x exp(j 2 pi fc t)}, u0m = 2 beta_m Vpi / pi (beta_m — по RMS огибающей).

    fc в Гц. Пик RRC-QPSK примерно в 1,6-1,8 раза выше RMS.
    """
    return beta_to_drive(beta_m, Vpi) * np.imag(x * np.exp(2j * np.pi * fc * t))


def receive_line(I, fs, f_line_GHz, sideband, sim, tx, trim=32, carrier=0, eq_taps=None):
    """Приёмник линии: IQ-смешение, ФНЧ, RRC, сопряжение, задержка, децимация.

    sideband: '+' (копия), '-' (сопряжение), 'x3' (утроенная фаза).
    tx — Baseband передатчика. Возвращает (rx, s_ref): символы линии (до data-aided
    нормировки) и опорные символы (для 'x3' s_ref = s^3/|s^3|).
    """
    if isinstance(tx, MultiBaseband):
        tx = tx.ref(carrier)
    n = len(I)
    t = np.arange(n) / fs
    SpS = int(round(fs / (sim.Rs * 1e9)))
    r = I * np.exp(-2j * np.pi * f_line_GHz * 1e9 * t)
    cut = (4.5 if sideband == "x3" else 1.5 if sim.pulse == "rrc" else 3.0) * sim.Rs * 1e9
    r = sosfiltfilt(butter(6, cut / (fs / 2), output="sos"), r)
    h = _mf(SpS, sim.alpha, sim.pulse)
    if sideband != "x3":
        r = firFilter(h, r)
    if sideband == "-":
        r = np.conj(r)
    if sideband == "x3":
        ref_wave = tx.x ** 3
        s_ref_all = tx.symbols ** 3 / np.abs(tx.symbols ** 3)
    else:
        ref_wave = firFilter(h, tx.x)
        s_ref_all = tx.symbols
    corr = np.fft.ifft(np.fft.fft(r) * np.conj(np.fft.fft(ref_wave)))
    lags = np.arange(-2 * SpS, 2 * SpS + 1)
    d = int(lags[np.argmax(np.abs(corr[lags % n]))])
    ns = len(tx.symbols)
    # RRC: пик согласованного фильтра на k*SpS; boxcar (rect): в конце окна символа, k*SpS + SpS/2
    idx = np.arange(ns) * SpS + d + (SpS // 2 if sim.pulse == "rect" else 0)
    keep = (idx >= 0) & (idx < n)
    keep[:trim] = False
    keep[ns - trim:] = False
    ik = idx[keep]
    if eq_taps:
        # матрица T/2-отсчётов вокруг момента каждого символа (для эквалайзера): столбцы j = -c..c
        h, c = SpS // 2, (eq_taps - 1) // 2
        A = np.stack([r[ik + (j - c) * h] for j in range(eq_taps)], axis=1)
        return r[ik], s_ref_all[keep], A
    return r[ik], s_ref_all[keep]


def equalize(A, s_ref):
    """Data-aided T/2-эквалайзер: МНК-подбор отводов по известным символам.

    A — матрица отсчётов (символы x отводы). Обучение и оценка разнесены по половинам записи
    (обучение на одной половине, применение к другой, затем наоборот), чтобы EVM не был занижен
    переобучением. Возвращает выровненные символы.
    """
    K = len(s_ref)
    h = K // 2
    z = np.empty(K, complex)
    for tr, te in ((slice(0, h), slice(h, K)), (slice(h, K), slice(0, h))):
        w = np.linalg.lstsq(A[tr], s_ref[tr], rcond=None)[0]
        z[te] = A[te] @ w
    return z


def _norm_evm(rx, s_ref, block):
    g = np.sum(rx * np.conj(s_ref)) / np.sum(np.abs(s_ref) ** 2)
    rxn = rx / g
    if block:
        rxn = rxn.copy()
        for a in range(0, len(rxn), block):
            seg = slice(a, a + block)
            rxn[seg] *= np.exp(-1j * np.angle(np.sum(rxn[seg] * np.conj(s_ref[seg]))))
    return rxn, np.sum(np.abs(rxn - s_ref) ** 2) / np.sum(np.abs(s_ref) ** 2)


def measure(I, fs, f_line_GHz, sideband, sim, tx, carrier=0):
    """receive_line + metrics с учётом sim.rx_block и эквалайзера (sim.equalizer != 'none').

    Всегда возвращает EVM без эквалайзера (evm_pct); при включённом эквалайзере — ещё evm_eq_pct.
    """
    M = carrier_M(tx, sim, carrier)
    if sim.equalizer != "none":
        rx, s, A = receive_line(I, fs, f_line_GHz, sideband, sim, tx, carrier=carrier, eq_taps=sim.eq_taps)
        return metrics(rx, s, M, sim.rx_block, eq=A)
    rx, s = receive_line(I, fs, f_line_GHz, sideband, sim, tx, carrier=carrier)
    return metrics(rx, s, M, sim.rx_block)


def metrics(rx, s_ref, M, block=None, eq=None):
    """EVM, SNR по EVM, BER Монте-Карло и BER по EVM.

    Data-aided: g = sum(rx conj(s)) / sum|s|^2; rx -> rx/g. Если задан block (символов), то
    затем для каждого блока поворот на -arg sum(rx conj(s)) (амплитуда остаётся глобальной).
    Если передана матрица eq (receive_line(..., eq_taps=N)), дополнительно считается эквалайзер:
    evm_eq_pct, snr_eq_dB, ber_eq (evm_eq_pct = None без eq).
    Возвращает dict evm_pct, snr_evm_dB, ber_mc, n_err, n_bits, ber_evm, rx_norm.
    """
    g = np.sum(rx * np.conj(s_ref)) / np.sum(np.abs(s_ref) ** 2)
    rxn = rx / g
    if block:
        rxn = rxn.copy()
        for a in range(0, len(rxn), block):
            seg = slice(a, a + block)
            rxn[seg] *= np.exp(-1j * np.angle(np.sum(rxn[seg] * np.conj(s_ref[seg]))))
    evm2 = np.sum(np.abs(rxn - s_ref) ** 2) / np.sum(np.abs(s_ref) ** 2)
    snr = -10 * np.log10(max(evm2, 1e-30))
    b_rx = demodulateGray(rxn, M, "psk")
    b_ref = demodulateGray(s_ref, M, "psk")
    n_err = int(np.sum(b_rx != b_ref))
    k = int(np.log2(M))
    out = dict(evm_pct=100 * np.sqrt(evm2), snr_evm_dB=snr, ber_mc=n_err / len(b_ref),
               n_err=n_err, n_bits=len(b_ref),
               ber_evm=float(theoryBER(M, snr - 10 * np.log10(k), "psk")), rx_norm=rxn,
               evm_eq_pct=None, snr_eq_dB=None, ber_eq=None)
    if eq is not None:
        rz, e2 = _norm_evm(equalize(eq, s_ref), s_ref, block)
        out.update(evm_eq_pct=100 * np.sqrt(e2), snr_eq_dB=-10 * np.log10(max(e2, 1e-30)),
                   ber_eq=float(np.mean(demodulateGray(rz, M, "psk") != b_ref)), rx_eq_norm=rz)
    return out
