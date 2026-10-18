"""Интерактивная страница: схема преобразователя и графики в любой её точке.

Запуск: .venv/bin/streamlit run app.py
Физики здесь нет: всё считает пакет mwp_converter; результаты кэшируются.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from mwp_converter.analytic import expected_lines
from mwp_converter.bpf import apply_bpf, bpf_for, bpf_response, suppression_table
from mwp_converter.budget import noise_budget
from mwp_converter.chain import STAGES, chain_attenuation, response_view
from mwp_converter.model import measure_lines, run_all
from mwp_converter.params import (Components, Config, Impairments, Plan, fs_GHz, load_config,
                                  resolve_sps)
from mwp_converter.plots import _thin
from mwp_converter.response import Bpf, Response, band_report, band_report_for_output
from mwp_converter.signals import make_psk_baseband, metrics, receive_line
from scipy.signal import hilbert

from mwp_converter.spectra import optical_spectrum, rf_psd, rf_psd_abs

CFG_DIR = Path(__file__).parent / "configs"
POINTS = ["a", "b", "c", "d", "m(t)", "e", "g", "h", "i", "k", "l", "m", "BPF₁", "BPF₂", "BPF₃"]
CH = {"a": 0, "b": 1, "c": 2, "g": 0, "h": 1, "i": 2, "k": 0, "l": 1, "m": 2,
      "BPF₁": 0, "BPF₂": 1, "BPF₃": 2}
LINES = {"2fᵢ+f_c": "upper", "2fᵢ−f_c": "lower", "f_c": "fc", "3f_c": "3fc"}
COMP_NAME = {"edfa": "EDFA", "mzm1_rf": "MZM1", "mzm2_rf": "MZM2", "pd": "PD", "wdm": "WDM", "plan": "Plan", "bpf": "BPF"}
KINDS = ["ideal", "rc", "skin", "brickwall", "supergauss", "csv"]

st.set_page_config(page_title="MWP converter", layout="wide")


# ---------- состояние и конфиг ----------
def init_state(cfg, tag):
    """При смене конфига перезаписывает значения виджетов значениями конфига."""
    if st.session_state.get("_cfg_tag") == tag:
        return
    st.session_state["_cfg_tag"] = tag
    p, s, c = cfg.plan, cfg.sim, cfg.comps
    vals = dict(fc=p.fc_GHz, f1=p.f_lo_GHz[0], f2=p.f_lo_GHz[1], f3=p.f_lo_GHz[2],
                wdm_sp=p.wdm_spacing_GHz, rs=s.Rs, direction=p.direction, beta1=s.beta1,
                beta_m=s.beta_m, bias_err=cfg.imp.bias_err * 100, er=cfg.imp.er_dB or 30.0,
                rin=cfg.imp.rin_dBHz if cfg.imp.rin_dBHz is not None else -150.0,
                rx_pw=cfg.imp.rx_power_dBm if cfg.imp.rx_power_dBm is not None else -20.0,
                lw=cfg.imp.linewidth_Hz or 1e6, wdm_iso=c.wdm_isolation_dB or 25.0)
    im = cfg.imp
    vals.update(n_loss=any(x > 0 for x in im.loss_dB), loss1=im.loss_dB[0], loss2=im.loss_dB[1], loss3=im.loss_dB[2],
                loss4=im.loss_dB[3], n_input=im.input_cn_dB is not None, cn=im.input_cn_dB or 20.0,
                nf_lna=im.nf_lna_dB, g_lna=im.g_lna_dB, nf_drv=im.nf_driver_dB,
                n_laser=im.rin_dBHz is not None or im.linewidth_Hz > 0, n_er=im.er_dB is not None,
                n_pd=im.pd_noise, n_amp=im.amp_nf_dB is not None, amp_nf=im.amp_nf_dB or 3.0,
                n_lo=im.lo_pn, lo_prof=im.lo_profile if im.lo_profile != "table" else "synth_typical", n_drv=im.drivers,
                oip3=im.oip3_dBm, hd2lo=im.hd2_lo_dBc, hd2sig=im.hd2_sig_dBc, n_drift=im.bias_drift,
                drift_sig=im.drift_sigma * 100, drift_tau=im.drift_tau_s, rx_block=s.rx_block or 0,
                n_edfa=im.edfa, edfa_g=im.edfa_G_dB, edfa_nf=im.edfa_NF_dB, fix_rx=im.rx_power_dBm is not None)
    b = c.bpf
    vals.update(pulse=s.pulse, bpf_kind=b.kind, bpf_order=b.order, bpf_bwf=b.bw_factor, bpf_ripple=b.ripple_dB,
                **{f"bpf_c{j + 1}": str(b.center[min(j, len(b.center) - 1)]) for j in range(3)})
    for k, r in (("mzm1", c.mzm1_rf), ("mzm2", c.mzm2_rf), ("pd", c.pd), ("wdm", c.wdm)):
        vals[f"{k}_kind"], vals[f"{k}_f3"], vals[f"{k}_ord"] = r.kind, r.f3dB_GHz or 40.0, r.order
    st.session_state.update(vals)
    st.session_state["_defaults"] = vals


def num(sb, label, key, **kw):
    """number_input условного блока: значение по умолчанию из конфига (состояние виджета, не
    отрисованного на прошлом прогоне, Streamlit сбрасывает)."""
    st.session_state.setdefault(key, st.session_state["_defaults"][key])
    return sb.number_input(label, key=key, **kw)


def sel(sb, label, key, options):
    """selectbox условного блока (значение по умолчанию из конфига)."""
    st.session_state.setdefault(key, st.session_state["_defaults"][key])
    return sb.selectbox(label, options, key=key)


def read_config():
    st.sidebar.header("Config")
    choice = st.sidebar.selectbox("Config", ["default", "realistic", "upload"], index=1, key="cfg_choice")
    if choice == "upload":
        up = st.sidebar.file_uploader("TOML config", type="toml", key="toml_up")
        cfg = load_config(text=up.getvalue().decode()) if up else load_config(CFG_DIR / "default.toml")
        tag = f"upload:{up.name if up else ''}"
    else:
        cfg, tag = load_config(CFG_DIR / f"{choice}.toml"), choice
    init_state(cfg, tag)
    return cfg


def read_widgets(cfg):
    sb = st.sidebar
    sb.header("Frequencies, GHz")
    fc = sb.number_input("f_c", min_value=0.1, step=0.5, key="fc")
    f = [sb.number_input(f"f₍{i + 1}₎ (LO)", min_value=0.1, step=0.5, key=f"f{i + 1}") for i in range(3)]
    sp = sb.number_input("WDM spacing", min_value=10.0, step=10.0, key="wdm_sp")
    rs = sb.number_input("Symbol rate, GBd", min_value=0.1, step=0.1, key="rs")
    direction = sb.selectbox("Direction", ["lower", "upper"], key="direction")
    plan = Plan(fc, tuple(f), direction, sp)
    outs = ", ".join(f"{o:g}" for o in plan.out_freqs())
    sb.header("Modulators / signal")
    b1 = sb.slider("β₁", 0.1, 3.0, step=0.05, key="beta1")
    bm = sb.slider("β_m", 0.05, 1.5, step=0.05, key="beta_m")
    bias = sb.selectbox("MZM1 bias", ["null", "quad"], key="bias1")
    M = {"BPSK": 2, "QPSK": 4, "8PSK": 8}[sb.selectbox("Modulation", ["BPSK", "QPSK", "8PSK"], index=1)]
    ns = sb.select_slider("Symbols (log2)", options=[10, 11, 12, 13], value=11)
    pulse = sb.selectbox("Pulse shape", ["rrc", "rect"], key="pulse")
    sim = replace(cfg.sim, Rs=rs, beta1=b1, beta_m=bm, M=M, n_sym=2 ** ns, pulse=pulse)
    sb.caption(f"Outputs |2fᵢ∓f_c|: {outs} GHz; fs = {fs_GHz(plan, sim):g} GHz "
               f"(SpS {resolve_sps(plan, sim)})")
    sb.header("Component bandwidths")
    resp = {}
    for k, label in (("mzm1", "MZM1 RF"), ("mzm2", "MZM2 RF"), ("pd", "PD"), ("wdm", "WDM")):
        kind = sb.selectbox(f"{label} type", KINDS, key=f"{k}_kind")
        f3 = sb.number_input(f"{label} f3dB (WDM: B3dB), GHz", min_value=0.1, key=f"{k}_f3")
        order = sb.number_input(f"{label} order", min_value=1, max_value=8, step=1, key=f"{k}_ord")
        text = None
        if kind == "csv":
            up = sb.file_uploader(f"{label} S21 CSV", type="csv", key=f"{k}_csv")
            text = up.getvalue().decode() if up else Path(CFG_DIR / "example_s21.csv").read_text()
        resp[k] = Response(kind, f3, int(order), csv_text=text)
    iso = sb.number_input("WDM isolation, dB", key="wdm_iso")
    sb.header("Output BPF")
    centers = []
    for j in range(3):
        txt = sb.text_input(f"BPF{'₁₂₃'[j]} center: auto or GHz", key=f"bpf_c{j + 1}")
        try:
            centers.append(float(txt))
        except ValueError:
            centers.append("auto")
    bpf = Bpf(kind=sb.selectbox("BPF type", ["butter", "cheby1"], key="bpf_kind"),
              order=int(sb.number_input("BPF order", min_value=1, max_value=8, step=1, key="bpf_order")),
              bw_factor=sb.number_input("BPF width factor × (1+α)R_s", min_value=0.2, step=0.1, key="bpf_bwf"),
              ripple_dB=sb.number_input("BPF ripple (cheby1), dB", min_value=0.01, key="bpf_ripple"),
              center=tuple(centers))
    comps = Components(resp["mzm1"], resp["mzm2"], resp["pd"], resp["wdm"], iso, bpf)
    sb.header("Noise sources")
    ci = cfg.imp
    kw = {"bias_err": sb.slider("MZM1 bias error ΔV_b/V_π, %", -10.0, 10.0, step=0.5, key="bias_err") / 100}
    if sb.checkbox("Insertion losses (MZM1 / MUX / MZM2 / DEMUX)", key="n_loss"):
        kw["loss_dB"] = tuple(num(sb, lbl + " loss, dB", f"loss{j + 1}", min_value=0.0, step=0.5)
                              for j, lbl in enumerate(["MZM1", "MUX", "MZM2", "DEMUX"]))
    if sb.checkbox("Input noise (C/N at LNA input in R_s band)", key="n_input"):
        kw.update(input_cn_dB=num(sb, "C/N, dB", "cn"), input_bw_factor=ci.input_bw_factor,
                  nf_lna_dB=num(sb, "LNA NF, dB", "nf_lna"),
                  g_lna_dB=num(sb, "LNA gain, dB", "g_lna"),
                  nf_driver_dB=num(sb, "Driver NF, dB", "nf_drv"))
    if sb.checkbox("Laser RIN and linewidth", key="n_laser"):
        kw.update(rin_dBHz=num(sb, "RIN, dB/Hz", "rin"),
                  linewidth_Hz=num(sb, "Laser linewidth, Hz", "lw"))
    if sb.checkbox("Finite extinction ratio", key="n_er"):
        kw["er_dB"] = num(sb, "ER, dB", "er")
    if sb.checkbox("PD shot + thermal noise", key="n_pd"):
        kw["pd_noise"] = True
    if sb.checkbox("Amplifier after PD", key="n_amp"):
        kw["amp_nf_dB"] = num(sb, "Amplifier NF, dB", "amp_nf")
    if sb.checkbox("EDFA after MZM2 (ASE)", key="n_edfa"):
        kw.update(edfa=True, edfa_G_dB=num(sb, "EDFA gain, dB", "edfa_g"),
                  edfa_NF_dB=num(sb, "EDFA NF, dB", "edfa_nf"), edfa_lambda_nm=ci.edfa_lambda_nm)
    if sb.checkbox("LO phase noise", key="n_lo"):
        kw.update(lo_pn=True, lo_profile=sel(sb, "LO phase noise profile", "lo_prof", ["synth_typical", "dvbs2_mask"]))
        sb.caption("dvbs2_mask is an approximate mask")
    if sb.checkbox("Driver nonlinearity (OIP3, HD2)", key="n_drv"):
        kw.update(drivers=True, oip3_dBm=num(sb, "OIP3, dBm", "oip3"), hd2_lo_dBc=num(sb, "HD2 LO driver, dBc", "hd2lo"),
                  hd2_sig_dBc=num(sb, "HD2 signal driver, dBc", "hd2sig"))
    if sb.checkbox("MZM1 bias drift (Ornstein-Uhlenbeck, one draw)", key="n_drift"):
        kw.update(bias_drift=True, drift_sigma=num(sb, "Drift sigma, % of V_π", "drift_sig", min_value=0.0) / 100,
                  drift_tau_s=num(sb, "Drift tau, s", "drift_tau", min_value=1e-3),
                  drift_ens=int(sb.number_input("Draw (ensemble index)", min_value=0, step=1, value=0, key="drift_ens")))
    if sb.checkbox("Fix received optical power", key="fix_rx"):
        kw["rx_power_dBm"] = num(sb, "Received power, dBm", "rx_pw")
    rxb = int(sb.number_input("Receiver phase tracking block, symbols (0 = off)", min_value=0, step=64, key="rx_block"))
    imp = Impairments(**kw)
    sim = replace(sim, rx_block=rxb or None)
    return Config(cfg.name, plan, sim, imp, comps, cfg.sweeps, cfg.scenarios), bias


# ---------- кэшируемые расчёты ----------
@st.cache_resource(show_spinner="Simulating channels...")
def compute(key, _cfg, bias):
    plan, sim = _cfg.plan, _cfg.sim
    sps = resolve_sps(plan, sim)
    tx = make_psk_baseband(sim.M, sim.n_sym, sps, sim.alpha, sim.seed, sim.pulse)
    res = run_all(plan, sim, _cfg.imp, _cfg.comps, baseband=tx, bias1=bias)
    return res, tx, res[0].fs


@st.cache_resource(show_spinner="BPF...")
def bpf_out(key, _res, _cfg, i):
    return apply_bpf(_res[i].I, _res[i].fs, _cfg.plan, _cfg.sim, _cfg.comps.bpf, i)


@st.cache_data(show_spinner="Receiver after BPF...")
def rx_metrics_bpf(key, _res, _tx, _cfg, i):
    plan, sim = _cfg.plan, _cfg.sim
    f, sb = plan.line_info(i)
    rx, s_ref = receive_line(bpf_out(key, _res, _cfg, i), _res[i].fs, f, sb, sim, _tx)
    m = metrics(rx, s_ref, sim.M, sim.rx_block)
    z = m.pop("rx_norm")
    return f, sb, m, z[:: max(1, len(z) // 3000)]


@st.cache_data(show_spinner="Suppression table...")
def suppression(key, _res, _cfg, i):
    return suppression_table(_res[i].I, bpf_out(key, _res, _cfg, i), _res[i].fs, _cfg.plan, _cfg.sim, i)


@st.cache_data(show_spinner="CW line table...")
def line_table(key, _cfg, bias, i):
    ana, approx = expected_lines(i, _cfg.plan, _cfg.sim, _cfg.comps, bias)
    names = [n for n in ana if n != "DC"]
    meas = measure_lines(i, _cfg.plan, _cfg.sim, _cfg.comps, names, bias1=bias)
    rows = []
    for n in names:
        f, a = meas[n]
        e = ana[n][1]
        rows.append({"line": n, "f, GHz": round(f, 3), "model": round(a, 5), "analytic (13′)/(16)": round(e, 5),
                     "diff, dB": round(20 * np.log10((a + 1e-12) / (e + 1e-12)), 3)})
    return rows, approx


@st.cache_data(show_spinner="Receiver...")
def rx_metrics(key, _res, _tx, _cfg, i, which):
    plan, sim = _cfg.plan, _cfg.sim
    f, sb = plan.line_info(i, which)
    rx, s_ref = receive_line(_res[i].I, _res[i].fs, f, sb, sim, _tx)
    m = metrics(rx, s_ref, sim.M, sim.rx_block)
    z = m.pop("rx_norm")
    step = max(1, len(z) // 3000)
    return f, sb, m, z[::step]


# ---------- графики ----------
def spec_fig(f, p, title, xrange=None, marks=(), mask=None, ylabel="dB rel. max"):
    xr = xrange or (float(f.min()), float(f.max()))
    ff, pp = _thin(np.asarray(f), np.asarray(p), xr)
    fig = go.Figure(go.Scatter(x=ff, y=pp, mode="lines", line=dict(width=1)))
    for x, name in marks:
        if xr[0] <= x <= xr[1]:
            fig.add_vline(x=x, line=dict(dash="dot", color="gray", width=1), annotation_text=name,
                          annotation_font_size=9, annotation_textangle=-90)
    if mask:
        fig.add_vrect(x0=mask[0], x1=mask[1], fillcolor="green", opacity=0.25, line_width=0)
    fig.update_layout(title=title, xaxis_title="GHz", yaxis_title=ylabel, height=420,
                      yaxis_range=[-110, 10], xaxis_range=list(xr), margin=dict(t=40))
    return fig


def time_fig(t, y, title, ylabel):
    k = max(1, len(t) // 20000)
    fig = go.Figure(go.Scatter(x=t[::k] * 1e9, y=y[::k], mode="lines", line=dict(width=1)))
    fig.update_layout(title=title, xaxis_title="t, ns", yaxis_title=ylabel, height=380, margin=dict(t=40))
    return fig


def const_fig(z, title):
    fig = go.Figure(go.Scatter(x=z.real, y=z.imag, mode="markers", marker=dict(size=3, opacity=0.5)))
    fig.update_layout(title=title, xaxis=dict(range=[-2, 2], title="I"), yaxis=dict(range=[-2, 2], title="Q",
                      scaleanchor="x"), height=480, width=480, margin=dict(t=50))
    return fig


def metric_text(m, f, sb):
    ber = "BER = 0" if m["n_err"] == 0 else f"BER = {m['ber_mc']:.2e}"
    inv = ", conjugated" if sb == "-" else ""
    return (f"{f:g} GHz{inv}: EVM = {m['evm_pct']:.2f} %, SNR(EVM) = {m['snr_evm_dB']:.1f} dB, "
            f"{ber} ({m['n_err']} errors / {m['n_bits']} bits), BER from EVM = {m['ber_evm']:.2e}")


def bpf_fill(rep, i, selected):
    """Цвет блока BPF: красный при out по этому каналу, иначе голубой (выбран) / белый."""
    if any(r["component"] == "bpf" and r["channel"] == i + 1 and r["level"] == "out" for r in rep):
        return "tomato"
    return "lightblue" if selected else "white"


def scheme_dot(plan, point, rep, sim, comps, imp):
    """Схема рис. 1 статьи (DOT): точки на рёбрах, выбранная подсвечена, компоненты красятся по band_report."""
    rank = {"ok": 0, "warn": 1, "out": 2}
    worst = {}
    for r in rep:
        c = r["component"]
        if rank[r["level"]] > rank[worst.get(c, "ok")]:
            worst[c] = r["level"]
    color = {"ok": "white", "warn": "gold", "out": "tomato"}

    def node(n, label, comp=None, shape="box"):
        return f'{n} [label="{label}", shape={shape}, style=filled, fillcolor={color[worst.get(comp, "ok")]}];'

    def edge(u, v, p=""):
        sel = p == point
        col = "red" if sel else "black"
        return f'{u} -> {v} [label="{p}", color={col}, fontcolor={col}, penwidth={3 if sel else 1}];'

    pts = (("a", "g", "k"), ("b", "h", "l"), ("c", "i", "m"))
    lines = ["digraph G { rankdir=LR; node [fontsize=10]; edge [fontsize=10];",
             node("MUX", "OMP (WDM MUX)", "wdm"), node("M2", f"MZM2 (f_c={plan.fc_GHz:g} GHz)", "mzm2_rf"),
             node("DMX", "ODMP (WDM DEMUX)", "wdm"), node("AMP", "AMP", shape="triangle"),
             edge("MUX", "M2", "d"),
             node("IN", f"Input PM signal\\nf_c = {plan.fc_GHz:g} GHz (LNA/BPF)"
                  + (f"\\nC/N = {imp.input_cn_dB:g} dB" if imp.input_cn_dB is not None else "")),
             edge("IN", "M2", "m(t)")]
    if imp.edfa:
        lines += [node("EDFA", f"EDFA\\nG = {imp.edfa_G_dB:g} dB / NF = {imp.edfa_NF_dB:g} dB", "edfa", "ellipse"),
                  edge("M2", "EDFA", "e"), edge("EDFA", "DMX")]
    else:
        lines += [edge("M2", "DMX", "e")]
    for i, (a, g, k) in enumerate(pts):
        fi = plan.f_lo_GHz[i]
        sub = "₁₂₃"[i]
        bpf_sel = point == f"BPF{sub}"
        f0, bw = bpf_for(plan, sim, comps.bpf, i)
        lines += [node(f"LD{i}", f"LD{i + 1}"), node(f"M1_{i}", f"MZM1 ({fi:g} GHz)", "mzm1_rf"),
                  node(f"PD{i}", f"PD{i + 1}", "pd"),
                  f'BPF{i} [label="BPF{sub}\\n{f0:g} GHz / {bw:g} GHz", shape=box, style=filled, '
                  f'fillcolor={bpf_fill(rep, i, bpf_sel)}];',
                  edge(f"LD{i}", f"M1_{i}"), edge(f"M1_{i}", "MUX", a), edge("DMX", f"PD{i}", g),
                  edge(f"PD{i}", f"BPF{i}", k), edge(f"BPF{i}", "AMP")]
    return "\n".join(lines + ["}"])


# ---------- страница ----------
cfg0 = read_config()
cfg, bias = read_widgets(cfg0)
st.title("Multi-band microwave-photonic frequency converter")
plan, sim = cfg.plan, cfg.sim
rep = band_report(plan, cfg.comps, sim.Rs, sim.alpha_eff, cfg.imp)
useful = band_report_for_output(plan, cfg.comps, sim.Rs, sim.alpha_eff)


def rec_text(r):
    return r.get("msg") or (f"{COMP_NAME[r['component']]}: ch{r['channel']} line {r['line']} = {r['f_GHz']:g} GHz "
                            f"attenuated by {r['att_dB']:.1f} dB")


for r in useful:
    if r["level"] == "out":
        st.error(rec_text(r))
n_warn = sum(r["level"] == "warn" for r in rep)
n_out = sum(r["level"] == "out" for r in rep)
u_out = sum(r["level"] == "out" for r in useful)
u_warn = sum(r["level"] == "warn" for r in useful)
with st.expander(f"Band check: {n_warn} warn, {n_out} out (useful chain: {u_warn} warn, {u_out} out)",
                 expanded=u_out > 0):
    for r in rep:
        if r["level"] != "ok":
            st.write(f"**{r['level']}** — {rec_text(r)}")

point = st.radio("Point", POINTS, horizontal=True, key="point")
st.graphviz_chart(scheme_dot(plan, point, useful, sim, cfg.comps, cfg.imp))
if all(c == "auto" for c in cfg.comps.bpf.center):
    st.caption("BPF retuned to " + ", ".join(f"{bpf_for(plan, sim, cfg.comps.bpf, j)[0]:g}" for j in range(3)) + " GHz (center = auto)")

key = repr(cfg) + bias
res, tx, fs = compute(key, cfg, bias)
i = CH.get(point, 0)
tabs = st.tabs(["Spectrum", "Time", "Constellation", "Responses", "Noise budget"])
fc = plan.fc_GHz
optical = point in ("a", "b", "c", "d", "e", "g", "h", "i")
rf = point in ("k", "l", "m")
is_bpf = point.startswith("BPF")
bwsig = (1 + sim.alpha_eff) * sim.Rs


def add_bpf_fill(fig, xr):
    """АЧХ BPF канала i полупрозрачной заливкой поверх спектра (дБ, нижняя граница -110)."""
    x = np.linspace(max(xr[0], 0.05), min(xr[1], fs / 2e9 * 0.999), 1500)
    y = np.maximum(20 * np.log10(np.abs(bpf_response(fs, plan, sim, cfg.comps.bpf, i, x)) + 1e-9), -110)
    fig.add_trace(go.Scatter(x=x, y=np.full_like(x, -110.0), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=0), fill="tonexty", name="BPF |H|",
                             fillcolor="rgba(0,160,0,0.18)"))


def psd_fig(curves, title, xr, marks=(), band=None, with_bpf=False):
    """curves: [(имя, f, dB, цвет|None)]; band — маска полосы приёма."""
    fig = go.Figure()
    for name, f, p, color in curves:
        ff, pp = _thin(np.asarray(f), np.asarray(p), xr)
        fig.add_trace(go.Scatter(x=ff, y=pp, mode="lines", name=name,
                                 line=dict(width=1, **({"color": color} if color else {}))))
    if with_bpf:
        add_bpf_fill(fig, xr)
    for x, name in marks:
        if xr[0] <= x <= xr[1]:
            fig.add_vline(x=x, line=dict(dash="dot", color="gray", width=1), annotation_text=name,
                          annotation_font_size=9, annotation_textangle=-90)
    if band:
        fig.add_vrect(x0=band[0], x1=band[1], fillcolor="green", opacity=0.2, line_width=0)
    fig.update_layout(title=title, xaxis_title="GHz", yaxis_title="dB rel. max", height=400,
                      yaxis_range=[-110, 10], xaxis_range=list(xr), margin=dict(t=40))
    return fig


with tabs[0]:
    if optical:
        xl = max(plan.f_lo_GHz) + fc + 10
        if point in ("d", "e"):
            offs = plan.offsets_GHz()
            fig = go.Figure()
            for j, r in enumerate(res):
                f, p = optical_spectrum(r.fields[point], fs)
                ff, pp = _thin(f + offs[j], p, (min(offs) - xl, max(offs) + xl))
                fig.add_trace(go.Scatter(x=ff, y=pp, mode="lines", name=f"ch{j + 1} (shift {offs[j]:+g})",
                                         line=dict(width=1)))
            fig.update_layout(title=f"Point {point}: group spectrum, channels shifted by WDM spacing",
                              xaxis_title="Offset from optical carrier, GHz", yaxis_title="dB rel. max",
                              yaxis_range=[-110, 10], height=420, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fk = "a" if point in "abc" else "g"
            f, p = optical_spectrum(res[i].fields[fk], fs)
            fi = plan.f_lo_GHz[i]
            marks = [(s * fi, f"{s:+d}f{i + 1}") for s in (-3, -1, 1, 3)]
            if fk == "g":
                marks += [(s * fi + q * fc, "") for s in (-1, 1) for q in (-1, 1)]
            st.plotly_chart(spec_fig(f, p, f"Point {point}: optical spectrum (offset from ω_λ{i + 1})",
                                     (-xl, xl), marks, ylabel="dB rel. max"), use_container_width=True)
    elif point == "m(t)":
        meta = res[0].meta
        f, p_in = rf_psd_abs(meta["u2_in"], fs, 2 ** 15)
        ref = p_in.max()
        curves = [("input", f, 10 * np.log10(p_in / ref + 1e-300), None)]
        if not cfg.comps.mzm2_rf.is_ideal:
            curves.append(("after MZM2 RF response", f,
                           10 * np.log10(rf_psd_abs(meta["u2_mzm"], fs, 2 ** 15)[1] / ref + 1e-300), None))
        for ttl, xr in (("full band", (0.0, fs / 2e9)), ("zoom f_c ± 3R_s", (fc - 3 * sim.Rs, fc + 3 * sim.Rs))):
            st.plotly_chart(psd_fig(curves, f"m(t): input PM signal PSD, {ttl}", xr,
                                    band=(fc - bwsig / 2, fc + bwsig / 2)), use_container_width=True)
    elif rf or is_bpf:
        f, p = rf_psd(res[i].I, fs)
        fo = plan.line_info(i)[0]
        band = (fo - bwsig / 2, fo + bwsig / 2)
        xmax = min(fs / 2e9, 1.1 * (2 * plan.f_lo_GHz[i] + 3 * fc))
        if rf:
            ana, approx = expected_lines(i, plan, sim, cfg.comps, bias)
            marks = [(v[0], n) for n, v in ana.items() if n != "DC"]
            st.plotly_chart(psd_fig([("photocurrent", f, p, None)], f"Point {point}: photocurrent PSD "
                                    "(green: receive band; shaded: BPF |H|)", (0, xmax), marks, band, True),
                            use_container_width=True)
            rows, approx = line_table(key, cfg, bias, i)
            st.caption("Line amplitudes, units of R·P₀/4 (noise-free CW run, no bias error)"
                       + (" — approx (WDM not ideal)" if approx else ""))
            st.dataframe(rows)
        else:
            Ib = bpf_out(key, res, cfg, i)
            _, pb = rf_psd(Ib, fs)
            pb = pb + 10 * np.log10(rf_psd_abs(Ib, fs)[1].max() / rf_psd_abs(res[i].I, fs)[1].max())
            f0, bwf = bpf_for(plan, sim, cfg.comps.bpf, i)
            spur = [(v, n) for n, v in plan.spurs(i).items()]
            marks = [(v, n.replace("fi", f"f{i + 1}")) for v, n in spur]
            marks.append((fo, "output"))
            curves = [("before BPF", f, p, "gray"), ("after BPF", f, pb, "#1f77b4")]
            st.plotly_chart(psd_fig(curves, f"BPF{'₁₂₃'[i]}: spectrum before / after, full band "
                                    f"(BPF {f0:g} GHz / {bwf:g} GHz)", (0, xmax), marks, band, True),
                            use_container_width=True)
            st.plotly_chart(psd_fig(curves, "zoom ±5 GHz around the output", (max(fo - 5, 0), fo + 5), marks,
                                    band, True), use_container_width=True)
            st.caption("Suppression of spurious lines within ±f_out of the output (power in the signal band, "
                       "dBc to the useful line):")
            st.dataframe([{"line": r["line"].replace("fi", f"f{i + 1}"), "f, GHz": r["f_GHz"],
                           "before BPF, dBc": r["before_dBc"], "after BPF, dBc": r["after_dBc"],
                           "suppression, dB": r["suppression_dB"]} for r in suppression(key, res, cfg, i)])
    else:
        st.info("Select a point to see its spectrum.")

with tabs[1]:
    if point in ("d", "e"):
        i = 0
    if point == "m(t)":
        meta = res[0].meta
        n = 8 * resolve_sps(plan, sim)
        st.plotly_chart(time_fig(res[0].t[:n], meta["u2_in"][:n], "m(t): u2(t) (MZM2 drive)", "u2, V"),
                        use_container_width=True)
        st.plotly_chart(time_fig(res[0].t[:n], np.abs(meta["x_in"][:n]), "m(t): envelope |x_in(t)| (with input noise, if enabled)", "|x|"),
                        use_container_width=True)
    elif optical:
        fk = {"a": "a", "b": "a", "c": "a", "d": "d", "e": "e"}.get(point, "g")
        n = int(5 * fs / (plan.f_lo_GHz[i] * 1e9))
        st.plotly_chart(time_fig(res[i].t[:n], np.abs(res[i].fields[fk][:n]) ** 2, f"Point {point}: |E|²",
                                 "|E|², W"), use_container_width=True)
    elif is_bpf:
        Ib = bpf_out(key, res, cfg, i)
        n = 8 * resolve_sps(plan, sim)
        t = res[i].t[:n] * 1e9
        k = max(1, n // 20000)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[::k], y=res[i].I[:n:k], mode="lines", name="before BPF",
                                 line=dict(width=1, color="lightgray")))
        fig.add_trace(go.Scatter(x=t[::k], y=Ib[:n:k], mode="lines", name="after BPF", line=dict(width=1)))
        fig.update_layout(title=f"BPF{'₁₂₃'[i]}: RF signal at the amplifier input", xaxis_title="t, ns",
                          yaxis_title="I, A", height=380, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
        env = go.Figure()
        env.add_trace(go.Scatter(x=t[::k], y=np.abs(hilbert(res[i].I[:n]))[::k], mode="lines", name="before BPF",
                                 line=dict(width=1, color="gray")))
        env.add_trace(go.Scatter(x=t[::k], y=np.abs(hilbert(Ib[:n]))[::k], mode="lines", name="after BPF",
                                 line=dict(width=1)))
        env.update_layout(title="Envelope", xaxis_title="t, ns", yaxis_title="|I_a|, A", height=340,
                          margin=dict(t=40))
        st.plotly_chart(env, use_container_width=True)
    elif rf:
        n = int(5 * fs / (plan.f_lo_GHz[i] * 1e9))
        st.plotly_chart(time_fig(res[i].t[:n], res[i].I[:n], f"Point {point}: photocurrent", "I, A"),
                        use_container_width=True)

with tabs[2]:
    if point == "m(t)":
        meta = res[0].meta
        sps = resolve_sps(plan, sim)
        sy, xs = meta["symbols"], meta["x_in"][::sps]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs.real, y=xs.imag, mode="markers", name=f"x_in after {sim.pulse} (symbol instants)",
                                 marker=dict(size=3, opacity=0.4)))
        fig.add_trace(go.Scatter(x=sy.real, y=sy.imag, mode="markers", name="transmitted symbols",
                                 marker=dict(size=9, symbol="x", color="red")))
        fig.update_layout(title="m(t): input constellation", xaxis=dict(range=[-2, 2], title="I"),
                          yaxis=dict(range=[-2, 2], title="Q", scaleanchor="x"), height=480, width=520)
        st.plotly_chart(fig, use_container_width=False)
    elif is_bpf:
        f, sb, m, z = rx_metrics_bpf(key, res, tx, cfg, i)
        _, _, m0, _ = rx_metrics(key, res, tx, cfg, i, None)
        st.write("After BPF — " + metric_text(m, f, sb))
        st.write("Without BPF — " + metric_text(m0, f, sb))
        st.plotly_chart(const_fig(z, f"{point} after BPF: {f:g} GHz"), use_container_width=False)
    elif rf:
        label = st.radio("Output line", list(LINES), horizontal=True, key="line_sel")
        f, sb, m, z = rx_metrics(key, res, tx, cfg, i, LINES[label])
        st.write(metric_text(m, f, sb))
        st.plotly_chart(const_fig(z, f"{point}: {f:g} GHz"), use_container_width=False)
    else:
        st.info("Constellation is shown for the input m(t), photocurrent points (k, l, m) and after BPF.")

with tabs[3]:
    ri = 0 if point in ("d", "e", "m(t)") else i
    v = response_view(point, plan, sim, cfg.comps, fs, ri)
    fig = go.Figure()
    for name, x, y in v["curves"]:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    if v["gd"] is not None:
        fig.add_trace(go.Scatter(x=v["gd"][0], y=v["gd"][1], mode="lines", name="group delay, ns", yaxis="y2",
                                 line=dict(dash="dash")))
        fig.update_layout(yaxis2=dict(title="group delay, ns", overlaying="y", side="right"))
    for x, name in v["marks"]:
        fig.add_vline(x=x, line=dict(dash="dot", color="gray", width=1), annotation_text=name,
                      annotation_font_size=9, annotation_textangle=-90)
    if v["band"]:
        fig.add_vrect(x0=v["band"][0], x1=v["band"][1], fillcolor="green", opacity=0.15, line_width=0)
    fig.update_layout(title=f"Point {point}: {v['title']}", xaxis_title=v["xlabel"], yaxis_title="|H|, dB",
                      yaxis_range=[-40, 3], height=440, margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe([{"line": r["line"], "f, GHz": r["f_GHz"], "attenuation, dB": r["att_dB"]} for r in v["table"]])
    if st.checkbox("Cumulative response up to this point", key="chain_resp"):
        ch = chain_attenuation(point, plan, sim, cfg.comps, fs, ri)
        st.dataframe([{"stage": r["stage"], "f, GHz": round(r["f_GHz"], 3), "attenuation, dB": round(r["att_dB"], 2),
                       "cumulative, dB": round(r["cum_dB"], 2)} for r in ch])
        cb = go.Figure(go.Bar(x=[r["stage"] for r in ch], y=[-r["cum_dB"] for r in ch]))
        cb.update_layout(title=f"Useful-line chain, channel {ri + 1}: cumulative |H|, dB "
                         f"(stages: {', '.join(STAGES)}; MZM RF losses act on modulation drive)",
                         yaxis_title="dB", height=320, margin=dict(t=50))
        st.plotly_chart(cb, use_container_width=True)


@st.cache_data(show_spinner="Computing noise budget (several simulations)...")
def budget_rows(key, _cfg, i, cn):
    return noise_budget(_cfg, i, cn)


with tabs[4]:
    st.caption("Contribution of each noise source (stage 1) to the output SNR at the selected channel's output line. "
               "Takes several simulations, so it is computed on demand.")
    cn_b = st.number_input("C/N at the input for the budget, dB", key="budget_cn", value=20.0)
    if st.button("Compute noise budget", key="budget_btn"):
        rows = budget_rows(key, cfg, i, cn_b)
        st.dataframe([{"source": r["name"], "EVM, %": round(r["evm_pct"], 2), "SNR alone, dB": round(r["snr_dB"], 1),
                       "Δ(1/SNR)": f"{r['delta_inv']:.2e}"} for r in rows])
        bf = go.Figure(go.Bar(x=[r["name"] for r in rows], y=[r["snr_dB"] for r in rows]))
        bf.update_layout(title=f"Channel {i + 1}: SNR if the source acts alone (dB)", height=380, margin=dict(t=50))
        st.plotly_chart(bf, use_container_width=True)
