"""АЧХ компонентов по точкам схемы и сквозное ослабление цепочки полезной линии."""
import numpy as np

from .bpf import bpf_for, bpf_response, group_delay_ns
from .response import Response

STAGES = ["MZM1 RF", "MUX", "MZM2 RF", "DEMUX", "PD", "BPF"]
# какие ступени цепочки входят в «сквозную АЧХ до точки» (включая показанный компонент)
UPTO = {"a": [0], "b": [0], "c": [0], "d": [0, 1], "m(t)": [2], "e": [0, 1, 2], "g": [0, 1, 2, 3],
        "h": [0, 1, 2, 3], "i": [0, 1, 2, 3], "k": [0, 1, 2, 3, 4], "l": [0, 1, 2, 3, 4],
        "m": [0, 1, 2, 3, 4], "BPF₁": [0, 1, 2, 3, 4, 5], "BPF₂": [0, 1, 2, 3, 4, 5],
        "BPF₃": [0, 1, 2, 3, 4, 5]}


UPTO_TYPE = {"a": [0], "d": [0, 1], "m(t)": [2], "e": [0, 1, 2], "g": [0, 1, 2, 3], "k": [0, 1, 2, 3, 4],
             "BPF": [0, 1, 2, 3, 4, 5]}
# старые имена точек (ветки 1-3) -> тип; новый вид — «тип + номер ветки»
LEGACY_POINTS = {"a": ("a", 0), "b": ("a", 1), "c": ("a", 2), "d": ("d", 0), "m(t)": ("m(t)", 0), "e": ("e", 0),
                 "g": ("g", 0), "h": ("g", 1), "i": ("g", 2), "k": ("k", 0), "l": ("k", 1), "m": ("k", 2),
                 "BPF₁": ("BPF", 0), "BPF₂": ("BPF", 1), "BPF₃": ("BPF", 2)}


def _ptype(point):
    """Тип точки: a | d | m(t) | e | g | k | BPF (принимает и старые имена b, c, h, i, l, m, BPF₁₋₃)."""
    return LEGACY_POINTS[point][0] if point in LEGACY_POINTS else point


def _db(resp: Response, f_GHz):
    h = np.abs(resp.H(np.atleast_1d(np.asarray(f_GHz, float)) * 1e9))
    return 20 * np.log10(np.maximum(h, 1e-6))


def useful_sidebands(plan, i):
    """Оптические боковые (ГГц, отстройка от несущей канала), формирующие выход: ±f_i и ±|f_i∓f_c|."""
    fi, fc = plan.f_lo_GHz[i], plan.fc_of(i)
    s = abs(fi - fc) if plan.routes[i].sideband == "lower" else fi + fc
    return [fi, s]


def stage_table(plan, sim, comps, fs, i):
    """Ослабление (дБ, положительное) каждой ступени цепочки полезной линии канала i."""
    fi, fc = plan.f_lo_GHz[i], plan.fc_of(i)
    fo = plan.line_info(i)[0]
    bw = (1 + sim.alpha_eff) * sim.Rs
    sb = useful_sidebands(plan, i)
    wdm = float(max(-_db(comps.wdm, s)[0] for s in sb))
    rows = [dict(stage="MZM1 RF", f_GHz=fi, att_dB=float(-_db(comps.mzm1_rf, fi)[0])),
            dict(stage="MUX", f_GHz=max(sb), att_dB=wdm),
            dict(stage="MZM2 RF", f_GHz=fc, att_dB=float(max(-_db(comps.mzm2_rf, [fc - bw / 2, fc, fc + bw / 2])))),
            dict(stage="DEMUX", f_GHz=max(sb), att_dB=wdm),
            dict(stage="PD", f_GHz=fo, att_dB=float(-_db(comps.pd, fo)[0])),
            dict(stage="BPF", f_GHz=fo,
                 att_dB=float(-20 * np.log10(abs(bpf_response(fs, plan, sim, comps.bpf, i, [fo])[0]) + 1e-12)))]
    return rows


def chain_attenuation(point, plan, sim, comps, fs, i):
    """Сквозное ослабление до точки: строки stage, f_GHz, att_dB, cum_dB (по ступеням из UPTO)."""
    rows = stage_table(plan, sim, comps, fs, i)
    cum = 0.0
    out = []
    for k in UPTO_TYPE[_ptype(point)]:
        cum += rows[k]["att_dB"]
        out.append(dict(rows[k], cum_dB=cum))
    return out


def response_view(point, plan, sim, comps, fs, i):
    """Данные графика АЧХ компонента точки: dict(title, xlabel, curves, marks, band, table, gd)."""
    fi, fc = plan.f_lo_GHz[i], plan.fc_of(i)
    fo = plan.line_info(i)[0]
    bw = (1 + sim.alpha_eff) * sim.Rs
    f = np.linspace(0, 150, 750)
    t = _ptype(point)
    v = dict(xlabel="Frequency, GHz", curves=[], marks=[], band=None, table=[], gd=None)

    def row(name, fq, resp):
        v["table"].append(dict(line=name, f_GHz=round(float(fq), 3), att_dB=round(float(-_db(resp, fq)[0]), 2)))

    if t == "a":
        r = comps.mzm1_rf
        v["title"] = f"MZM1 RF input, channel {i + 1}"
        v["curves"] = [("MZM1 RF |H|", f, _db(r, f))]
        for k in (1, 2, 3):
            v["marks"].append((k * fi, f"{k}f{i + 1}"))
            row(f"{k}f{i + 1}", k * fi, r)
    elif t in ("d", "g"):
        r = comps.wdm
        offs = plan.offsets_GHz()
        sel = range(len(offs)) if t == "d" else [i]
        lo, hi = min(offs) - 90, max(offs) + 90
        if t != "d":
            lo, hi = -90, 90
        x = np.linspace(lo, hi, 1200)
        v["xlabel"] = "Optical offset, GHz"
        v["title"] = "MUX (all channels, shifted by WDM spacing)" if t == "d" else f"DEMUX, channel {i + 1}"
        for j in sel:
            sh = offs[j] if t == "d" else 0.0
            v["curves"].append((f"ch{j + 1} WDM |H|", x, _db(r, x - sh)))
            fj = plan.f_lo_GHz[j]
            useful = useful_sidebands(plan, j)
            for s in (-1, 1):
                for k, q in ((fj, "f"), (abs(fj - fc), "|f-fc|"), (fj + fc, "f+fc")):
                    mk = f"{'+' if s > 0 else '-'}{q}"
                    is_u = any(abs(k - u) < 1e-9 for u in useful)
                    v["marks"].append((sh + s * k, ("★" if is_u else "") + mk))
                    if s > 0:
                        row(f"ch{j + 1} ±{q}{' (useful)' if is_u else ''}", k, r)
    elif t in ("m(t)", "e"):
        r = comps.mzm2_rf
        v["title"] = "MZM2 RF input"
        v["curves"] = [("MZM2 RF |H|", f, _db(r, f))]
        lo_, hi_ = plan.band_GHz(bw)
        v["band"] = (lo_, hi_)
        for nm, fq in (("fc", fc), ("fc-edge", lo_), ("fc+edge", hi_)):
            v["marks"].append((fq, nm))
            row(nm, fq, r)
    elif t == "k":
        r = comps.pd
        v["title"] = f"Photodiode, channel {i + 1}"
        v["curves"] = [("PD |H|", f, _db(r, f))]
        for nm, fq in (("output", fo), ("fc", fc), ("2fi", 2 * fi), ("3fc", 3 * fc)):
            v["marks"].append((fq, nm))
            row(nm, fq, r)
    else:  # BPF
        f0, bwf = bpf_for(plan, sim, comps.bpf, i)
        x = np.linspace(max(f0 - 6 * bwf, 0.05), f0 + 6 * bwf, 1500)
        h = bpf_response(fs, plan, sim, comps.bpf, i, x)
        v["title"] = f"BPF{i + 1}: {f0:g} GHz / {bwf:g} GHz ({comps.bpf.kind}, order {comps.bpf.order})"
        v["curves"] = [("BPF |H|", x, 20 * np.log10(np.maximum(np.abs(h), 1e-6)))]
        v["gd"] = (x, group_delay_ns(fs, plan, sim, comps.bpf, i, x))
        v["band"] = (f0 - bwf / 2, f0 + bwf / 2)
        v["marks"].append((fo, "output"))
        pts = [("output", fo)] + sorted(((n, fq) for n, fq in plan.spurs(i).items()
                                         if x[0] <= fq <= x[-1]), key=lambda t: t[1])
        for n, fq in pts:
            if n != "output":
                v["marks"].append((fq, n))
            hh = bpf_response(fs, plan, sim, comps.bpf, i, [fq])[0]
            v["table"].append(dict(line=n, f_GHz=round(float(fq), 3),
                                   att_dB=round(float(-20 * np.log10(abs(hh) + 1e-12)), 2)))
    return v
