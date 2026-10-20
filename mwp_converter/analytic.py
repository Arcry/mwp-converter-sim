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
    fi, fc = plan.f_lo_GHz[i], plan.fc_of(i)
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


# ---------- многотоновая аналитика (МЦМ1 в нуле, K тонов на МЦМ2) ----------
def multitone_products(beta_i, betas, fcs):
    """Продукты МЦМ2 на K тонах и их амплитуды линии на выходе (ед. R*E0^2/4).

    Разложение Якоби-Ангера по каждому тону; |T2|^2 = 1/2[1 - sin(2 sum beta_k sin th_k)].
    Амплитуда линии на выходе ветки = J2(2 beta_i) * A, где A = 2 prod_k J_{n_k}(2 beta_k):
      - тон f_ck:               2 J1(2b_k) prod_{j!=k} J0 (линия на 2f_i±f_ck и сам тон: [1-J0(2b_i)]*A);
      - IM3 2f_ca - f_cb:       2 J2(2b_a) J1(2b_b) prod_{others} J0;
      - тройной f_ca+f_cb-f_cc: 2 J1(2b_a) J1(2b_b) J1(2b_c) prod_{others} J0.
    Возвращает list dict(name, kind, p_GHz, amp), где p — частота продукта во входной полосе, amp —
    амплитуда каждой из линий 2f_i±p (kind: 'tone' | 'im3' | 'tri').
    """
    K = len(betas)
    b2 = [2 * b for b in betas]
    j0 = [jv(0, x) for x in b2]
    j1 = [jv(1, x) for x in b2]
    j2 = [jv(2, x) for x in b2]
    g = 2 * jv(2, 2 * beta_i)

    def others(excl):
        return float(np.prod([j0[m] for m in range(K) if m not in excl]))

    out = []
    for k in range(K):
        out.append(dict(name=f"tone(c{k + 1})", kind="tone", p_GHz=fcs[k], amp=abs(g * j1[k] * others({k}))))
    for a in range(K):
        for b in range(K):
            if a != b:
                out.append(dict(name=f"im3(2c{a + 1}-c{b + 1})", kind="im3", p_GHz=2 * fcs[a] - fcs[b],
                                amp=abs(g * j2[a] * j1[b] * others({a, b}))))
    for a in range(K):
        for b in range(a + 1, K):
            for c in range(K):
                if c not in (a, b):
                    out.append(dict(name=f"im3(c{a + 1}+c{b + 1}-c{c + 1})", kind="tri",
                                    p_GHz=fcs[a] + fcs[b] - fcs[c],
                                    amp=abs(g * j1[a] * j1[b] * j1[c] * others({a, b, c}))))
    return out


def lines_multitone(beta_i, betas, k, fcs, f_i, sideband="lower"):
    """Амплитуды линий ветки (МЦМ1 в нуле) для несущей k: имя -> (f_ГГц, амплитуда), ед. R*E0^2/4.

    Ключи: 'DC', '2fi', 'fc' (прямое просачивание несущей k), 'out' (полезная линия по боковой
    маршрута) и по одной записи на каждый продукт IM3/тройной (частота — после переноса).
    При K = 1 'out' совпадает с lines_null['2fi±fc'], 'fc' — с lines_null['fc'].
    """
    prods = multitone_products(beta_i, betas, fcs)
    tone = next(p for p in prods if p["name"] == f"tone(c{k + 1})")

    def xfer(p):
        return abs(2 * f_i - p) if sideband == "lower" else 2 * f_i + p

    d0 = 1 - jv(0, 2 * beta_i)
    lines = {"DC": (0.0, d0), "2fi": (2 * f_i, 2 * jv(2, 2 * beta_i)),
             "fc": (fcs[k], d0 * tone["amp"] / jv(2, 2 * beta_i)),
             "out": (xfer(fcs[k]), tone["amp"])}
    for p in prods:
        if p["kind"] != "tone":
            lines[p["name"]] = (xfer(p["p_GHz"]), p["amp"])
    return lines


def ci_multitone(beta_i, betas, k, fcs, f_i, sideband="lower", tol_GHz=1e-6):
    """C/I по многотоновой аналитике: полезная линия k против IM3/тройных продуктов в её слоте.

    Продукты складываются по мощности (фазы не учитываются). Возвращает dict(ci_dB, C, I, n).
    """
    lines = lines_multitone(beta_i, betas, k, fcs, f_i, sideband)
    fo, C = lines["out"]
    I2 = [a ** 2 for n, (f, a) in lines.items() if n.startswith("im3(") and abs(f - fo) < tol_GHz]
    I = float(np.sqrt(sum(I2)))
    return dict(ci_dB=float(20 * np.log10(C / I)) if I > 0 else float("inf"), C=float(C), I=I, n=len(I2))
