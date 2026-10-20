"""Спектры: Welch-PSD, амплитуды линий, оптический спектр."""
from fractions import Fraction
from math import gcd

import numpy as np
from scipy import signal
from scipy.fft import next_fast_len


def rf_psd(I, fs, nperseg=2 ** 13):
    """Welch PSD фототока. Возвращает (f_GHz, дБ относительно максимума)."""
    f, p = signal.welch(I, fs=fs, nperseg=min(nperseg, len(I)), return_onesided=True)
    return f / 1e9, 10 * np.log10(p / p.max() + 1e-300)


def rf_psd_abs(x, fs, nperseg=2 ** 13):
    """Welch PSD в линейных единицах: (f_GHz, p)."""
    f, p = signal.welch(x, fs=fs, nperseg=min(nperseg, len(x)), return_onesided=True)
    return f / 1e9, p


def band_power(f, p, f0, bw):
    """Мощность (сумма PSD по бинам) в окне f0 ± bw/2; f, f0, bw — в ГГц."""
    m = np.abs(f - f0) <= bw / 2
    return float(p[m].sum()) if m.any() else 1e-300


def coherent_length(fs, freqs_GHz, n_target, nmax=2 ** 20):
    """Длина записи N ~ n_target, при которой все частоты попадают точно в бины.

    N кратна знаменателю f/fs (частоты округляются до 1e-4 ГГц). Если N > nmax —
    возвращается None (тогда надо брать окно flat-top).
    """
    q = 1
    for f in freqs_GHz:
        fr = Fraction(int(round(f * 1e4)), int(round(fs / 1e5)))
        q = q * fr.denominator // gcd(q, fr.denominator)
        if q > nmax:
            return None
    k = max(1, round(n_target / q))
    k = next_fast_len(k)
    n = q * k
    return n if n <= nmax else q


def line_amplitudes(I, fs, freqs_GHz):
    """Амплитуды линий фототока (А): 2|X_k|/N, DC — |X_0|/N.

    Если все частоты попадают точно в бины (длину подбирают coherent_length) — окно
    прямоугольное; иначе — flat-top, погрешность амплитуды < 0,01 дБ (разброс
    вершины окна). Делить результат на R*P0/4 — снаружи.
    """
    n = len(I)
    f = np.asarray(freqs_GHz, float) * 1e9
    k = f * n / fs
    coherent = np.allclose(k, np.round(k), atol=1e-6, rtol=0)
    if coherent:
        w = np.ones(n)
    else:
        w = signal.windows.flattop(n, sym=False)
    X = np.fft.rfft(I * w)
    s = w.sum()
    out = []
    for kk, ff in zip(k, f):
        j = int(round(kk))
        j = min(max(j, 0), len(X) - 1)
        if not coherent:
            lo, hi = max(j - 1, 0), min(j + 2, len(X))
            a = np.abs(X[lo:hi]).max()
        else:
            a = abs(X[j])
        out.append(a / s if ff < 1e-3 else 2 * a / s)
    return np.array(out)


def optical_spectrum(E, fs):
    """Оптический спектр: (отстройка ГГц, дБ относительно максимума)."""
    X = np.fft.fftshift(np.fft.fft(E))
    f = np.fft.fftshift(np.fft.fftfreq(len(E), 1 / fs)) / 1e9
    p = np.abs(X) ** 2
    return f, 10 * np.log10(p / p.max() + 1e-300)
