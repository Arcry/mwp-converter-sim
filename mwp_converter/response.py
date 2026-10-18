"""АЧХ компонентов (ideal/rc/skin/brickwall/supergauss/csv), применение к сигналам, отчёт о полосах."""
import io
import warnings
from dataclasses import dataclass

import numpy as np

KINDS = ("ideal", "rc", "skin", "brickwall", "supergauss", "csv")


@dataclass
class Response:
    """АЧХ компонента.

    kind: ideal | rc | skin | brickwall | supergauss (только WDM) | csv.
    f3dB_GHz: частота по -3 дБ (для supergauss — полная ширина полосы B по -3 дБ).
    order: порядок (rc, supergauss).
    csv_path / csv_text: таблица `freq_GHz,S21_dB[,phase_deg]` (файл или текст).
    Фаза у skin нулевая (отмечено в README).
    """

    kind: str = "ideal"
    f3dB_GHz: float | None = None
    order: int = 1
    csv_path: str | None = None
    csv_text: str | None = None

    def _table(self):
        if self.csv_text is not None:
            src = io.StringIO(self.csv_text)
        else:
            src = open(self.csv_path)
        with src:
            data = np.genfromtxt(src, delimiter=",", names=True)
        names = data.dtype.names
        f = np.asarray(data[names[0]], float)
        s21 = np.asarray(data[names[1]], float)
        ph = np.asarray(data[names[2]], float) if len(names) > 2 else None
        return f, s21, ph

    def H(self, f_Hz):
        """Комплексная передаточная функция на частотах f_Hz (знак частоты не важен)."""
        fa = np.abs(np.asarray(f_Hz, float)) / 1e9
        k = self.kind
        if k == "ideal":
            return np.ones(fa.shape, complex)
        if k == "csv":
            f, s21, ph = self._table()
            if np.any(fa > f.max() * (1 + 1e-12)):
                warnings.warn(f"CSV-АЧХ: частоты выше {f.max()} ГГц — взято последнее значение таблицы")
            mag = 10 ** (np.interp(fa, f, s21) / 20)
            phase = np.deg2rad(np.interp(fa, f, ph)) if ph is not None else 0.0
            return mag * np.exp(1j * phase)
        if self.f3dB_GHz is None:
            raise ValueError(f"Response '{k}': нужна f3dB_GHz")
        f3 = self.f3dB_GHz
        if k == "rc":
            return 1.0 / (1 + 1j * fa / f3) ** self.order
        if k == "skin":
            return (10 ** (-3 * np.sqrt(fa / f3) / 20)).astype(complex)
        if k == "brickwall":
            warnings.warn("brickwall нефизичен: идеальный прямоугольный срез")
            return (fa <= f3).astype(complex)
        if k == "supergauss":
            return np.exp(-np.log(2) / 2 * (2 * fa / f3) ** (2 * self.order)).astype(complex)
        raise ValueError(f"Неизвестный вид АЧХ: {k}")

    @property
    def is_ideal(self):
        return self.kind == "ideal"


@dataclass
class Bpf:
    """Выходной полосовой фильтр канала (перед усилителем).

    kind: butter | cheby1 (ripple_dB). order — порядок прототипа (4). Ширина полосы:
    bw_GHz, а если не задана — bw_factor*(1+alpha_eff)*Rs. center: по одному элементу на канал
    ('auto' — центр следует за выходной линией, либо число, ГГц); если элементов меньше, чем
    каналов, используется последний.
    """

    kind: str = "butter"
    order: int = 4
    bw_factor: float = 2.0
    bw_GHz: float | None = None
    ripple_dB: float = 1.0
    center: tuple = ("auto",)


def bpf_band(plan, rs_GHz, alpha_eff, bpf, i):
    """(центр, полоса) BPF канала i, ГГц."""
    c = bpf.center[min(i, len(bpf.center) - 1)]
    f0 = plan.line_info(i)[0] if c == "auto" else float(c)
    bw = bpf.bw_GHz if bpf.bw_GHz else bpf.bw_factor * (1 + alpha_eff) * rs_GHz
    return f0, bw


def bpf_att_dB(f, f0, bw, order):
    """Оценка ослабления Баттерворта (полосовое преобразование), дБ; -3 дБ на краях полосы."""
    x = (f * f - f0 * f0) / (max(f, 1e-9) * bw)
    return float(10 * np.log10(1 + x ** (2 * order)))


def apply(x, fs, response):
    """БПФ -> умножение на H(f) -> ОБПФ. Вещественный сигнал — rfft, комплексная огибающая — fft."""
    if response is None or response.is_ideal:
        return np.array(x, copy=True)
    n = len(x)
    if np.iscomplexobj(x):
        f = np.fft.fftfreq(n, 1 / fs)
        return np.fft.ifft(np.fft.fft(x) * response.H(f))
    f = np.fft.rfftfreq(n, 1 / fs)
    return np.fft.irfft(np.fft.rfft(x) * response.H(f), n=n)


def level_of(att_dB):
    """ok <= 1 дБ; warn 1..3 дБ; out >= 3 дБ (ровно 3 дБ — край полосы — считается out)."""
    if att_dB <= 1.0:
        return "ok"
    if att_dB < 3.0 - 1e-9:
        return "warn"
    return "out"


def _att(resp, f_Hz, passes=1):
    h = np.abs(resp.H(np.array([f_Hz])))[0]
    return float(-20 * passes * np.log10(max(h, 1e-12)))


def band_report(plan, comps, rs_GHz=1.0, alpha=0.35, imp=None):
    """Ослабление |H| (дБ) компонентов на значимых частотах каждого канала.

    WDM — ослабление двух проходов фильтра (MUX+DEMUX, в модели фильтр применяется дважды) на
    оптических боковых своего канала.
    Запись: component, channel, line, f_GHz, att_dB, level. В конец добавлены записи
    plan.check() (component="plan") о паразитных линиях у выхода.
    """
    rep = []
    fc = plan.fc_GHz
    bw = (1 + alpha) * rs_GHz

    def add(comp, ch, line, f, resp, passes=1):
        a = _att(resp, f * 1e9, passes)
        rep.append(dict(component=comp, channel=ch, line=line, f_GHz=float(abs(f)),
                        att_dB=a, level=level_of(a)))

    for i, fi in enumerate(plan.f_lo_GHz):
        add("mzm1_rf", i + 1, "fi", fi, comps.mzm1_rf)
        add("mzm2_rf", i + 1, "fc", fc, comps.mzm2_rf)
        add("mzm2_rf", i + 1, "fc-edge", fc - bw / 2, comps.mzm2_rf)
        add("mzm2_rf", i + 1, "fc+edge", fc + bw / 2, comps.mzm2_rf)
        for name, f in (("2fi+fc", 2 * fi + fc), ("2fi-fc", 2 * fi - fc), ("fc", fc), ("3fc", 3 * fc)):
            add("pd", i + 1, name, f, comps.pd)
        for name, f in (("fi", fi), ("fi+fc", fi + fc), ("fi-fc", fi - fc), ("3fi", 3 * fi)):
            add("wdm", i + 1, f"optical {name} [MUX+DEMUX]", f, comps.wdm, passes=2)
        if not comps.wdm.is_ideal and comps.wdm_isolation_dB is not None:
            rep.append(dict(component="wdm", channel=i + 1, line="neighbour leakage",
                            f_GHz=float(plan.wdm_spacing_GHz), att_dB=float(comps.wdm_isolation_dB),
                            level="ok"))
        if (not comps.wdm.is_ideal and not comps.pd.is_ideal and comps.pd.f3dB_GHz is not None
                and plan.wdm_spacing_GHz - bw < comps.pd.f3dB_GHz):
            rep.append(dict(component="wdm", channel=i + 1,
                            line="WDM beat inside PD band: interference model is inexact",
                            f_GHz=float(plan.wdm_spacing_GHz), att_dB=0.0, level="warn"))
    if imp is not None and getattr(imp, "edfa", False) and getattr(imp, "edfa_ase", True) and comps.wdm.is_ideal:
        rep.append(dict(component="edfa", channel=0, line="ASE", f_GHz=0.0, att_dB=0.0, level="warn",
                        msg="EDFA ASE is not filtered: WDM is ideal (full-band ASE reaches the photodiode)"))
    chk = plan.check(rs_GHz, alpha)
    rep += chk
    flagged = {(r["channel"], r["line"]) for r in chk}
    for i in range(len(plan.f_lo_GHz)):
        f0, bw = bpf_band(plan, rs_GHz, alpha, comps.bpf, i)
        fo = plan.line_info(i)[0]
        inside = abs(fo - f0) <= bw / 2 + 1e-9
        att = bpf_att_dB(fo, f0, bw, comps.bpf.order)
        rep.append(dict(component="bpf", channel=i + 1, line="output line", f_GHz=float(fo), att_dB=att,
                        level="ok" if inside else "out",
                        **({} if inside else dict(msg=f"channel {i + 1}: output line {fo:g} GHz outside "
                                                      f"BPF {f0:g}±{bw / 2:g} GHz"))))
        for name, f in plan.spurs(i).items():
            if abs(f - f0) <= bw / 2 and (i + 1, name) not in flagged and abs(f - fo) > 1e-9:
                rep.append(dict(component="bpf", channel=i + 1, line=name, f_GHz=float(f), att_dB=0.0,
                                level="out", msg=f"channel {i + 1}: spurious {name} = {f:g} GHz inside "
                                                 f"BPF {f0:g}±{bw / 2:g} GHz"))
    return rep


def band_report_for_output(plan, comps, rs_GHz=1.0, alpha=0.35, channel=None, imp=None):
    """Записи band_report только для цепочки полезной (выходной) линии.

    mzm1_rf на f_i; mzm2_rf на f_c и краях полосы; pd на самой выходной линии |2f_i∓f_c|;
    wdm (MUX+DEMUX) на боковых, формирующих выход: lower (|2f_i-f_c|) — ±f_i и ±|f_i-f_c|;
    upper (2f_i+f_c) — ±f_i и ±(f_i+f_c); записи plan.check() (совпадения с выходом).
    channel — номер канала с 1 (None — все каналы).
    """
    out_line = "2fi-fc" if plan.direction == "lower" else "2fi+fc"
    keep = {"mzm1_rf": ("fi",), "mzm2_rf": ("fc", "fc-edge", "fc+edge"), "pd": (out_line,),
            "wdm": ("optical fi",
                    "optical fi-fc" if plan.direction == "lower" else "optical fi+fc"), "plan": None, "bpf": None}
    rep = []
    for r in band_report(plan, comps, rs_GHz, alpha, imp):
        if r["component"] == "edfa":
            continue
        if channel is not None and r["channel"] != channel:
            continue
        lines = keep.get(r["component"], ())
        if lines is None or r["line"].replace(" [MUX+DEMUX]", "") in lines:
            rep.append(r)
    return rep


def worst_level(rep):
    """Худший уровень (ok < warn < out) в списке записей."""
    lv = {r["level"] for r in rep}
    return "out" if "out" in lv else "warn" if "warn" in lv else "ok"


def effective_betas(plan, comps, beta1, beta_m):
    """Эффективные (beta1*|H_mzm1(fi)|, beta_m*|H_mzm2(fc)|) для каждого канала."""
    hm = np.abs(comps.mzm2_rf.H(np.array([plan.fc_GHz * 1e9])))[0]
    out = []
    for fi in plan.f_lo_GHz:
        h1 = np.abs(comps.mzm1_rf.H(np.array([fi * 1e9])))[0]
        out.append((beta1 * h1, beta_m * hm))
    return out
