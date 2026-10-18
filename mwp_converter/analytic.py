"""Аналитические амплитуды линий фототока: (13') (МЦМ1 в нуле) и (16) (МЦМ1 в квадратуре).

Амплитуды — в единицах R*E0^2/4; DC — значение постоянной составляющей.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import jv

from .response import effective_betas

# имя линии -> (k_i, k_c): частота |k_i*f_i + k_c*f_c|
LINE_COEFFS = {
    "DC": (0, 0), "fi": (1, 0), "2fi": (2, 0), "3fi": (3, 0), "fc": (0, 1), "3fc": (0, 3),
    "fi+fc": (1, 1), "fi-fc": (1, -1), "2fi+fc": (2, 1), "2fi-fc": (2, -1),
    "3fi+fc": (3, 1), "3fi-fc": (3, -1), "2fi+3fc": (2, 3), "2fi-3fc": (2, -3),
    "4fi+fc": (4, 1), "4fi-fc": (4, -1),
}


def line_freq_GHz(name, fi_GHz, fc_GHz):
    """Частота линии по её имени (положительная)."""
    ki, kc = LINE_COEFFS[name]
    return abs(ki * fi_GHz + kc * fc_GHz)


def lines_null(beta1, beta_m):
    """Амплитуды по (13'), МЦМ1 в нуле."""
    a, b = 2 * beta1, 2 * beta_m
    d0 = 1 - jv(0, a)
    j2, j4 = jv(2, a), jv(4, a)
    m1, m3 = jv(1, b), jv(3, b)
    return {
        "DC": d0, "2fi": 2 * j2, "fc": 2 * d0 * m1,
        "2fi+fc": 2 * j2 * m1, "2fi-fc": 2 * j2 * m1,
        "3fc": 2 * d0 * m3,
        "2fi+3fc": 2 * j2 * m3, "2fi-3fc": 2 * j2 * m3,
        "4fi+fc": 2 * jv(4, a) * m1, "4fi-fc": 2 * j4 * m1,
    }


def lines_quad(beta1, beta_m):
    """Амплитуды по (16), МЦМ1 в квадратуре."""
    a, b = 2 * beta1, 2 * beta_m
    j1, j3 = jv(1, a), jv(3, a)
    m1 = jv(1, b)
    return {
        "DC": 1.0, "fi": 2 * j1, "fc": 2 * m1, "3fi": 2 * j3, "3fc": 2 * jv(3, b),
        "fi+fc": 2 * j1 * m1, "fi-fc": 2 * j1 * m1,
        "3fi+fc": 2 * j3 * m1, "3fi-fc": 2 * j3 * m1,
    }


def beta1_opt():
    """beta1, максимизирующий J2(2 beta1): ~1.527."""
    r = minimize_scalar(lambda b: -jv(2, 2 * b), bounds=(0.5, 2.5), method="bounded",
                        options=dict(xatol=1e-10))
    return float(r.x)


def expected_lines(i, plan, sim, comps, bias="null"):
    """Ожидаемые амплитуды линий канала i с учётом полос.

    beta_эфф = beta*|H_мцм| (effective_betas), линия умножается на |H_pd(f_line)|.
    Возвращает (dict имя -> (f_GHz, амплитуда), approx), approx=True, если WDM не ideal.
    """
    b1, bm = effective_betas(plan, comps, sim.beta1, sim.beta_m)[i]
    amps = (lines_null if bias == "null" else lines_quad)(b1, bm)
    fi, fc = plan.f_lo_GHz[i], plan.fc_GHz
    out = {}
    for name, a in amps.items():
        f = 0.0 if name == "DC" else line_freq_GHz(name, fi, fc)
        h = np.abs(comps.pd.H(np.array([f * 1e9])))[0]
        out[name] = (f, float(a * h))
    return out, not comps.wdm.is_ideal


def bias_error_ratio(beta1, bias_err):
    """Отношение линии f_i±f_c к 2f_i±f_c при уходе смещения MZM1 (bias_err — доля Vpi).

    В конвенции OptiCommPy фаза смещается на delta = pi*bias_err/2, T1 = sin^2(x + delta):
    f_i±f_c = 2|sin 2delta| J1(2 beta1) J1(2 beta_m), 2f_i±f_c = 2 cos 2delta J2(2 beta1) J1(2 beta_m),
    отношение = |tan(pi*bias_err)| J1(2 beta1)/J2(2 beta1).
    """
    return abs(np.tan(np.pi * bias_err)) * jv(1, 2 * beta1) / jv(2, 2 * beta1)
