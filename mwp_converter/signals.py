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


def rf_drive(x, fc, beta_m, Vpi, t):
    """u2 = u0m * Im{x exp(j 2 pi fc t)}, u0m = 2 beta_m Vpi / pi (beta_m — по RMS огибающей).

    fc в Гц. Пик RRC-QPSK примерно в 1,6-1,8 раза выше RMS.
    """
    return beta_to_drive(beta_m, Vpi) * np.imag(x * np.exp(2j * np.pi * fc * t))


def receive_line(I, fs, f_line_GHz, sideband, sim, tx, trim=32):
    """Приёмник линии: IQ-смешение, ФНЧ, RRC, сопряжение, задержка, децимация.

    sideband: '+' (копия), '-' (сопряжение), 'x3' (утроенная фаза).
    tx — Baseband передатчика. Возвращает (rx, s_ref): символы линии (до data-aided
    нормировки) и опорные символы (для 'x3' s_ref = s^3/|s^3|).
    """
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
    return r[idx[keep]], s_ref_all[keep]


def metrics(rx, s_ref, M, block=None):
    """EVM, SNR по EVM, BER Монте-Карло и BER по EVM.

    Data-aided: g = sum(rx conj(s)) / sum|s|^2; rx -> rx/g. Если задан block (символов), то
    затем для каждого блока поворот на -arg sum(rx conj(s)) (амплитуда остаётся глобальной).
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
    return dict(evm_pct=100 * np.sqrt(evm2), snr_evm_dB=snr, ber_mc=n_err / len(b_ref),
                n_err=n_err, n_bits=len(b_ref),
                ber_evm=float(theoryBER(M, snr - 10 * np.log10(k), "psk")), rx_norm=rxn)
