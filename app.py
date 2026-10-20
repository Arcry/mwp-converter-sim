"""Интерактивная страница: схема преобразователя (N веток, K несущих) и графики в любой её точке.

Запуск: .venv/bin/streamlit run app.py
Физики здесь нет: всё считает пакет mwp_converter; результаты кэшируются.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import hilbert

from mwp_converter.analytic import expected_lines
from mwp_converter.bpf import apply_bpf, bpf_for, bpf_response, suppression_table
from mwp_converter.budget import noise_budget
from mwp_converter.chain import STAGES, chain_attenuation, response_view
from mwp_converter.interference import ci_runs, drive_sweep, npr
from mwp_converter.model import measure_lines, run_all
from mwp_converter.params import (Carrier, Components, Config, Impairments, Plan, Route, load_config,
                                  mem_estimate_GB, resolve_sps)
from mwp_converter.plots import _thin
from mwp_converter.response import (Bpf, OpticalFilter, Response, band_report, band_report_for_output)
from mwp_converter.signals import make_multicarrier_baseband, measure
from mwp_converter.spectra import rf_psd, rf_psd_abs

CFG_DIR = Path(__file__).parent / "configs"
CONFIGS = ["default", "realistic", "router", "router_realistic", "upload"]
PTYPES = ["a", "d", "m(t)", "e", "g", "k", "BPF"]
PNAME = {"a": "a — after MZM1", "d": "d — after MUX", "m(t)": "m(t) — MZM2 input", "e": "e — after MZM2",
         "g": "g — after DEMUX", "k": "k — photodiode", "BPF": "BPF — amplifier input"}
COMP_NAME = {"edfa": "EDFA", "mzm1_rf": "MZM1", "mzm2_rf": "MZM2", "pd": "PD", "wdm": "WDM", "plan": "Plan",
             "bpf": "BPF", "ofilt": "Optical filter"}
KINDS = ["ideal", "rc", "skin", "brickwall", "supergauss", "csv"]
ROUTE_COLS = ["carrier", "f_out_GHz", "sideband", "lo", "bpf_center", "bpf_bw_GHz"]
_INFO = {}            # контейнер сводки в боковой панели
MAX_DRAWN = 8          # при N > 8 схема рисует выбранную ветку с соседями

st.set_page_config(page_title="MWP converter", layout="wide")


# ---------- состояние и конфиг ----------
def base_tables(cfg):
    """Исходные таблицы Carriers и Routes из конфига (для st.data_editor)."""
    p, b = cfg.plan, cfg.comps.bpf
    cdf = pd.DataFrame([dict(f_GHz=c.f_GHz, level_dB=c.level_dB, M=c.M or cfg.sim.M) for c in p.carriers])
    rows = []
    for i, r in enumerate(p.routes):
        fck = p.fc_of(i)
        c = b.center[min(i, len(b.center) - 1)]
        bw = b.bw_GHz[min(i, len(b.bw_GHz) - 1)] if isinstance(b.bw_GHz, (tuple, list)) else (b.bw_GHz or 0.0)
        rows.append(dict(carrier=r.carrier + 1, f_out_GHz=float(p.line_info(i)[0]), sideband=r.sideband,
                         lo="high" if (r.sideband == "lower" and 2 * r.f_lo_GHz > fck) else "low",
                         bpf_center=str(c), bpf_bw_GHz=float(bw)))
    return cdf, pd.DataFrame(rows, columns=ROUTE_COLS)


def init_state(cfg, tag):
    """При смене конфига перезаписывает значения виджетов значениями конфига."""
    if st.session_state.get("_cfg_tag") == tag:
        return
    st.session_state["_cfg_tag"] = tag
    for k in ("carriers_ed", "routes_ed", "branch", "ci_results", "npr_results", "sweep_result"):
        st.session_state.pop(k, None)
    s, c, im, b = cfg.sim, cfg.comps, cfg.imp, cfg.comps.bpf
    cdf, rdf = base_tables(cfg)
    st.session_state["_carriers_base"], st.session_state["_routes_base"] = cdf, rdf
    vals = dict(wdm_sp=cfg.plan.wdm_spacing_GHz, rs=s.Rs, beta1=s.beta1, beta_m=s.beta_m,
                bias_err=im.bias_err * 100, er=im.er_dB or 30.0,
                rin=im.rin_dBHz if im.rin_dBHz is not None else -150.0,
                rx_pw=im.rx_power_dBm if im.rx_power_dBm is not None else -20.0,
                lw=im.linewidth_Hz or 1e6, wdm_iso=c.wdm_isolation_dB or 25.0,
                n_sym_log2=9 if cfg.plan.K > 1 else 11, pulse=s.pulse, bias1="null",
                n_loss=any(x > 0 for x in im.loss_dB), loss1=im.loss_dB[0], loss2=im.loss_dB[1],
                loss3=im.loss_dB[2], loss4=im.loss_dB[3], n_input=im.input_cn_dB is not None,
                cn=im.input_cn_dB or 20.0, nf_lna=im.nf_lna_dB, g_lna=im.g_lna_dB, nf_drv=im.nf_driver_dB,
                n_laser=im.rin_dBHz is not None or im.linewidth_Hz > 0, n_er=im.er_dB is not None,
                n_pd=im.pd_noise, n_amp=im.amp_nf_dB is not None, amp_nf=im.amp_nf_dB or 3.0,
                n_lo=im.lo_pn, lo_prof=im.lo_profile if im.lo_profile != "table" else "synth_typical",
                n_drv=im.drivers, oip3=im.oip3_dBm, hd2lo=im.hd2_lo_dBc, hd2sig=im.hd2_sig_dBc,
                n_drift=im.bias_drift, drift_sig=im.drift_sigma * 100, drift_tau=im.drift_tau_s,
                rx_block=s.rx_block or 0, n_edfa=im.edfa, edfa_g=im.edfa_G_dB, edfa_nf=im.edfa_NF_dB,
                fix_rx=im.rx_power_dBm is not None, bpf_kind=b.kind, bpf_order=b.order, bpf_bwf=b.bw_factor,
                bpf_ripple=b.ripple_dB, bpf_stop=b.stop_dB, of_kind=c.ofilt.kind, of_fwhm=c.ofilt.fwhm_GHz,
                of_order=c.ofilt.order, eq_mode=s.equalizer, eq_taps=s.eq_taps)
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
    choice = st.sidebar.selectbox("Config", CONFIGS, index=1, key="cfg_choice")
    if choice == "upload":
        up = st.sidebar.file_uploader("TOML config", type="toml", key="toml_up")
        cfg = load_config(text=up.getvalue().decode()) if up else load_config(CFG_DIR / "default.toml")
        tag = f"upload:{up.name if up else ''}"
    else:
        cfg, tag = load_config(CFG_DIR / f"{choice}.toml"), choice
    init_state(cfg, tag)
    return cfg


def read_plan(cfg):
    """Таблицы Carriers/Routes (st.data_editor) -> Plan и параметры BPF по веткам."""
    sb = st.sidebar
    sb.header("Carriers (input band)")
    cdf = sb.data_editor(st.session_state["_carriers_base"], key="carriers_ed", num_rows="dynamic",
                         column_config={"M": st.column_config.SelectboxColumn("M", options=[2, 4, 8])},
                         hide_index=True)
    sb.header("Routes (branches)")
    rdf = sb.data_editor(st.session_state["_routes_base"], key="routes_ed", num_rows="dynamic",
                         column_config={"sideband": st.column_config.SelectboxColumn("sideband", options=["lower", "upper"]),
                                        "lo": st.column_config.SelectboxColumn("lo", options=["low", "high"])},
                         hide_index=True)
    sb.caption("carrier: number from 1; lo: LO below (low) or above (high) f_c/2 (high = spectrum inverted); "
               "bpf_center: auto or GHz; bpf_bw_GHz: 0 = default width.")
    box = sb.container()           # вычисляемые f_i / инверсия и сводка fs, SpS, память, β_k, PAPR, IBO (заполняются ниже)
    _INFO["box"] = box
    try:
        cdf = cdf.dropna(subset=["f_GHz"])
        carriers = tuple(Carrier(float(r.f_GHz), float(r.level_dB), int(r.M), k) for k, r in enumerate(cdf.itertuples()))
        routes, centers, bws = [], [], []
        for r in rdf.dropna(subset=["f_out_GHz", "carrier"]).itertuples():
            routes.append(Route.from_output(carriers, int(r.carrier) - 1, float(r.f_out_GHz), r.sideband, r.lo))
            try:
                centers.append(float(r.bpf_center))
            except (TypeError, ValueError):
                centers.append("auto")
            bws.append(float(r.bpf_bw_GHz or 0.0))
        plan = Plan(carriers=carriers, routes=tuple(routes), wdm_spacing_GHz=num(sb, "WDM spacing, GHz", "wdm_sp",
                                                                                  min_value=10.0, step=10.0))
    except (ValueError, IndexError, TypeError) as e:
        st.error(f"Invalid carriers/routes table: {e}")
        st.stop()
    box.dataframe([dict(branch=i + 1, carrier=plan.routes[i].carrier + 1, f_i_GHz=round(plan.f_lo_GHz[i], 4),
                        f_out_GHz=round(plan.line_info(i)[0], 4),
                        inverted="yes" if plan.line_info(i)[1] == "-" else "no") for i in range(plan.N)],
                  hide_index=True)
    return plan, centers, bws


def read_widgets(cfg):
    sb = st.sidebar
    plan, centers, bws = read_plan(cfg)
    sb.header("Modulators / signal")
    rs = sb.number_input("Symbol rate, GBd", min_value=0.01, step=0.05, key="rs")
    b1 = sb.slider("β₁", 0.1, 3.0, step=0.05, key="beta1")
    bm = sb.slider("β_m", 0.05, 1.5, step=0.05, key="beta_m")
    bias = sb.selectbox("MZM1 bias", ["null", "quad"], key="bias1")
    ns = sb.select_slider("Symbols (log2)", options=[9, 10, 11, 12, 13], key="n_sym_log2")
    pulse = sb.selectbox("Pulse shape", ["rrc", "rect"], key="pulse")
    sim = replace(cfg.sim, Rs=rs, beta1=b1, beta_m=bm, n_sym=2 ** ns, pulse=pulse)
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
    kind = sb.selectbox("BPF type", ["butter", "cheby1", "ellip"], key="bpf_kind")
    order = int(sb.number_input("BPF order", min_value=1, max_value=12, step=1, key="bpf_order"))
    bwf = sb.number_input("BPF width factor × (1+α)R_s (if width = 0 in the table)", min_value=0.2, step=0.1, key="bpf_bwf")
    ripple = sb.number_input("BPF ripple (cheby1/ellip), dB", min_value=0.01, key="bpf_ripple")
    stop = sb.number_input("BPF stopband (ellip), dB", min_value=1.0, key="bpf_stop")
    dflt = bwf * (1 + sim.alpha_eff) * sim.Rs
    bpf = Bpf(kind=kind, order=order, bw_factor=bwf, ripple_dB=ripple, stop_dB=stop, center=tuple(centers),
              bw_GHz=tuple(b or dflt for b in bws) if any(bws) else None)
    sb.header("Optical filter before PD")
    of_kind = sb.selectbox("Optical filter", ["none", "lorentz", "supergauss"], key="of_kind")
    ofilt = OpticalFilter(of_kind, num(sb, "FWHM, GHz", "of_fwhm", min_value=0.01), int(num(sb, "supergauss order", "of_order",
                                                                                          min_value=1, step=1)))
    sb.header("Receiver equalizer")
    eq = sb.selectbox("Equalizer", ["none", "lms"], key="eq_mode")
    sim = replace(sim, equalizer=eq, eq_taps=int(sb.number_input("Equalizer taps (T/2)", min_value=3, step=2, key="eq_taps")))
    comps = Components(resp["mzm1"], resp["mzm2"], resp["pd"], resp["wdm"], iso, bpf, ofilt)
    imp = read_noise(cfg)
    rxb = int(sb.number_input("Receiver phase tracking block, symbols (0 = off)", min_value=0, step=64, key="rx_block"))
    sim = replace(sim, rx_block=rxb or None)
    return Config(cfg.name, plan, sim, imp, comps, cfg.sweeps, cfg.scenarios), bias


def read_noise(cfg):
    sb = st.sidebar
    sb.header("Noise sources")
    ci = cfg.imp
    kw = {"bias_err": sb.slider("MZM1 bias error ΔV_b/V_π, %", -10.0, 10.0, step=0.5, key="bias_err") / 100}
    if sb.checkbox("Insertion losses (MZM1 / MUX / MZM2 / DEMUX)", key="n_loss"):
        kw["loss_dB"] = tuple(num(sb, lbl + " loss, dB", f"loss{j + 1}", min_value=0.0, step=0.5)
                              for j, lbl in enumerate(["MZM1", "MUX", "MZM2", "DEMUX"]))
    if sb.checkbox("Input noise (C/N at LNA input in R_s band)", key="n_input"):
        kw.update(input_cn_dB=num(sb, "C/N, dB", "cn"), input_bw_factor=ci.input_bw_factor,
                  nf_lna_dB=num(sb, "LNA NF, dB", "nf_lna"), g_lna_dB=num(sb, "LNA gain, dB", "g_lna"),
                  nf_driver_dB=num(sb, "Driver NF, dB", "nf_drv"))
    if sb.checkbox("Laser RIN and linewidth", key="n_laser"):
        kw.update(rin_dBHz=num(sb, "RIN, dB/Hz", "rin"), linewidth_Hz=num(sb, "Laser linewidth, Hz", "lw"))
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
    return Impairments(**kw)


# ---------- кэшируемые расчёты ----------
@st.cache_resource(show_spinner="Simulating branches...")
def compute(key, _cfg, bias):
    plan, sim = _cfg.plan, _cfg.sim
    tx = make_multicarrier_baseband(plan, sim, resolve_sps(plan, sim))
    res = run_all(plan, sim, _cfg.imp, _cfg.comps, baseband=tx, bias1=bias, keep_fields=False)
    return res, tx, res[0].fs


@st.cache_resource(show_spinner="BPF...")
def bpf_out(key, _res, _cfg, i):
    return apply_bpf(_res[i].I, _res[i].fs, _cfg.plan, _cfg.sim, _cfg.comps.bpf, i)


def line_for(plan, i, j):
    """(частота ГГц, sideband) несущей j на выходе ветки i."""
    fi, fcj = plan.f_lo_GHz[i], plan.carriers[j].f_GHz
    if plan.routes[i].sideband == "upper":
        return 2 * fi + fcj, "+"
    return abs(2 * fi - fcj), ("-" if 2 * fi > fcj else "+")


@st.cache_data(show_spinner="Receiver...")
def rx_metrics(key, _res, _tx, _cfg, i, j, after_bpf):
    plan, sim = _cfg.plan, _cfg.sim
    f, sb = line_for(plan, i, j)
    I = bpf_out(key, _res, _cfg, i) if after_bpf else _res[i].I
    m = measure(I, _res[i].fs, f, sb, sim, _tx, carrier=j)
    z = m.pop("rx_norm")
    ze = m.pop("rx_eq_norm", None)
    step = max(1, len(z) // 3000)
    return f, sb, m, z[::step], None if ze is None else ze[::step]


@st.cache_data(show_spinner="Branch metrics...")
def branch_table(key, _res, _tx, _cfg):
    rows = []
    for i in range(_cfg.plan.N):
        k = _cfg.plan.routes[i].carrier
        _, _, m, _, _ = rx_metrics(key, _res, _tx, _cfg, i, k, True)
        rows.append(dict(i=i, carrier=k, evm=m["evm_pct"], evm_eq=m["evm_eq_pct"], ber=m["ber_mc"], n_err=m["n_err"]))
    return rows


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


@st.cache_data(show_spinner="Computing C/I (3-4 simulations)...")
def ci_branch(key, _cfg, i):
    r = ci_runs(_cfg, i)
    return {k: (float(v) if np.isscalar(v) and v is not None else v) for k, v in r.items() if k != "rx_norm"}


@st.cache_data(show_spinner="Computing NPR...")
def npr_branch(key, _cfg, i):
    return npr(_cfg, i)


@st.cache_data(show_spinner="Drive sweep (several simulations)...")
def sweep_branch(key, _cfg, i, betas):
    return drive_sweep(_cfg, i, list(betas), with_npr=False)


@st.cache_data(show_spinner="Computing noise budget (several simulations)...")
def budget_rows(key, _cfg, i, cn):
    return noise_budget(_cfg, i, cn)


# ---------- графики ----------
def time_fig(t, y, title, ylabel):
    k = max(1, len(t) // 20000)
    fig = go.Figure(go.Scatter(x=t[::k] * 1e9, y=y[::k], mode="lines", line=dict(width=1)))
    fig.update_layout(title=title, xaxis_title="t, ns", yaxis_title=ylabel, height=380, margin=dict(t=40))
    return fig


def const_fig(traces, title):
    fig = go.Figure()
    for z, name, kw in traces:
        fig.add_trace(go.Scatter(x=z.real, y=z.imag, mode="markers", name=name, marker={"size": 3, "opacity": 0.5, **kw}))
    fig.update_layout(title=title, xaxis=dict(range=[-2, 2], title="I"), yaxis=dict(range=[-2, 2], title="Q", scaleanchor="x"),
                      height=480, width=520, margin=dict(t=50))
    return fig


def metric_text(m, f, sb, j):
    ber = "BER = 0" if m["n_err"] == 0 else f"BER = {m['ber_mc']:.2e}"
    inv = ", conjugated" if sb == "-" else ""
    eq = f", EVM with equalizer = {m['evm_eq_pct']:.2f} %" if m.get("evm_eq_pct") is not None else ""
    return (f"carrier {j + 1} at {f:g} GHz{inv}: EVM = {m['evm_pct']:.2f} %{eq}, SNR(EVM) = {m['snr_evm_dB']:.1f} dB, "
            f"{ber} ({m['n_err']} errors / {m['n_bits']} bits), BER from EVM = {m['ber_evm']:.2e}")


def rec_text(r):
    return r.get("msg") or (f"{COMP_NAME[r['component']]}: ch{r['channel']} line {r['line']} = {r['f_GHz']:g} GHz "
                            f"attenuated by {r['att_dB']:.1f} dB")


def bpf_fill(rep, i, selected):
    """Цвет блока BPF: красный при out по этому каналу, иначе голубой (выбран) / белый."""
    if any(r["component"] == "bpf" and r["channel"] == i + 1 and r["level"] == "out" for r in rep):
        return "tomato"
    return "lightblue" if selected else "white"


def scheme_dot(plan, ptype, branch, rep, sim, comps, imp):
    """Схема рис. 1 статьи (DOT) на N веток; при N > 8 — выбранная ветка с соседями и узел «… more»."""
    rank = {"ok": 0, "warn": 1, "out": 2}
    worst = {}
    for r in rep:
        if rank[r["level"]] > rank[worst.get(r["component"], "ok")]:
            worst[r["component"]] = r["level"]
    color = {"ok": "white", "warn": "gold", "out": "tomato"}

    def node(n, label, comp=None, shape="box"):
        return f'{n} [label="{label}", shape={shape}, style=filled, fillcolor={color[worst.get(comp, "ok")]}];'

    def edge(u, v, p="", sel=False):
        col = "red" if sel else "black"
        return f'{u} -> {v} [label="{p}", color={col}, fontcolor={col}, penwidth={3 if sel else 1}];'

    fcs = [c.f_GHz for c in plan.carriers]
    inp = f"Input band: {plan.K} carrier(s)\\n{min(fcs):g}–{max(fcs):g} GHz (LNA/BPF)"
    if imp.input_cn_dB is not None:
        inp += f"\\nC/N = {imp.input_cn_dB:g} dB"
    lines = ["digraph G { rankdir=LR; node [fontsize=10]; edge [fontsize=10];",
             node("MUX", "OMP (WDM MUX)", "wdm"), node("M2", f"MZM2 (f_mid={plan.f_mid():g} GHz)", "mzm2_rf"),
             node("DMX", "ODMP (WDM DEMUX)", "wdm"), node("AMP", "AMP", shape="triangle"),
             edge("MUX", "M2", "d", ptype == "d"), node("IN", inp), edge("IN", "M2", "m(t)", ptype == "m(t)")]
    if imp.edfa:
        lines += [node("EDFA", f"EDFA\\nG = {imp.edfa_G_dB:g} dB / NF = {imp.edfa_NF_dB:g} dB", "edfa", "ellipse"),
                  edge("M2", "EDFA", "e", ptype == "e"), edge("EDFA", "DMX")]
    else:
        lines += [edge("M2", "DMX", "e", ptype == "e")]
    shown = list(range(plan.N))
    if plan.N > MAX_DRAWN:
        shown = [j for j in range(branch - 1, branch + 2) if 0 <= j < plan.N]
        lines.append(f'MORE [label="… {plan.N - len(shown)} more branches", shape=plaintext];')
        lines.append("MORE -> AMP [style=dotted];")
    for i in shown:
        fi = plan.f_lo_GHz[i]
        sel_b = i == branch - 1
        f0, bw = bpf_for(plan, sim, comps.bpf, i)
        pdl = f"PD{i + 1}" + ("\\n+ optical filter" if not comps.ofilt.is_none else "")
        lines += [node(f"LD{i}", f"LD{i + 1}"),
                  node(f"M1_{i}", f"MZM1 ({fi:g} GHz)\\nc{plan.routes[i].carrier + 1} → {plan.line_info(i)[0]:g} GHz", "mzm1_rf"),
                  node(f"PD{i}", pdl, "pd"),
                  f'BPF{i} [label="BPF{i + 1}\\n{f0:g} GHz / {bw:g} GHz", shape=box, style=filled, '
                  f'fillcolor={bpf_fill(rep, i, sel_b and ptype == "BPF")}];',
                  edge(f"LD{i}", f"M1_{i}"), edge(f"M1_{i}", "MUX", f"a{i + 1}", sel_b and ptype == "a"),
                  edge("DMX", f"PD{i}", f"g{i + 1}", sel_b and ptype == "g"),
                  edge(f"PD{i}", f"BPF{i}", f"k{i + 1}", sel_b and ptype == "k"), edge(f"BPF{i}", "AMP")]
    return "\n".join(lines + ["}"])


def routing_matrix(plan):
    """Матрица «несущие × ветки»: частота выхода каждой несущей через каждую ветку (★ — выбранная несущая)."""
    data = {}
    for i in range(plan.N):
        col = []
        for j in range(plan.K):
            f, sb = line_for(plan, i, j)
            own = plan.routes[i].carrier == j
            col.append(f"{'★ ' if own else ''}{f:.3f}{' (inv)' if sb == '-' else ''}")
        data[f"b{i + 1} (f_i={plan.f_lo_GHz[i]:.3f})"] = col
    return pd.DataFrame(data, index=[f"c{j + 1} ({c.f_GHz:g} GHz)" for j, c in enumerate(plan.carriers)])


# ---------- страница ----------
cfg0 = read_config()
cfg, bias = read_widgets(cfg0)
st.title("Multi-band microwave-photonic frequency converter / carrier router")
plan, sim = cfg.plan, cfg.sim
rep = band_report(plan, cfg.comps, sim.Rs, sim.alpha_eff, cfg.imp)
useful = band_report_for_output(plan, cfg.comps, sim.Rs, sim.alpha_eff)
key = repr(cfg) + bias
res, tx, fs = compute(key, cfg, bias)

meta0 = res[0].meta
sps = resolve_sps(plan, sim)
_INFO["box"].caption(f"fs = {fs / 1e9:g} GHz (SpS {sps}), memory ≈ {mem_estimate_GB(plan, sim):.2f} GB; β_k = "
                     f"{', '.join(f'{b:.3f}' for b in meta0.get('beta_k', [sim.beta_m]))}; PAPR = {meta0.get('papr_dB', float('nan')):.1f} dB; "
                     f"IBO ref. = {meta0.get('ibo_dB', float('nan')):.1f} dB")
st.caption(f"fs = {fs / 1e9:g} GHz (SpS {sps}), symbols = {sim.n_sym}, estimated memory = {mem_estimate_GB(plan, sim):.2f} GB; "
           f"K = {plan.K} carrier(s), N = {plan.N} branch(es); drive: β_k = "
           f"{', '.join(f'{b:.3f}' for b in meta0.get('beta_k', [sim.beta_m]))}, PAPR = {meta0.get('papr_dB', float('nan')):.1f} dB, "
           f"IBO ref. = {meta0.get('ibo_dB', float('nan')):.1f} dB"
           + ("; K > 1: 2^9 symbols by default to keep the page ≤ 20 s (raise in the sidebar for better statistics)"
              if plan.K > 1 and sim.n_sym <= 2 ** 9 else ""))
for r in useful:
    if r["level"] == "out":
        st.error(rec_text(r))
n_warn = sum(r["level"] == "warn" for r in rep)
n_out = sum(r["level"] == "out" for r in rep)
u_out = sum(r["level"] == "out" for r in useful)
u_warn = sum(r["level"] == "warn" for r in useful)
with st.expander(f"Band check: {n_warn} warn, {n_out} out (useful chain: {u_warn} warn, {u_out} out)", expanded=u_out > 0):
    for r in rep:
        if r["level"] != "ok":
            st.write(f"**{r['level']}** — {rec_text(r)}")
rows_tbl = [dict(branch=i + 1, carrier=plan.routes[i].carrier + 1, f_carrier_GHz=plan.fc_of(i), f_i_GHz=round(plan.f_lo_GHz[i], 4),
                 f_out_GHz=round(plan.line_info(i)[0], 4), inverted="yes" if plan.line_info(i)[1] == "-" else "no",
                 BPF_center_GHz=round(bpf_for(plan, sim, cfg.comps.bpf, i)[0], 4),
                 BPF_bw_GHz=round(bpf_for(plan, sim, cfg.comps.bpf, i)[1], 4)) for i in range(plan.N)]
with st.expander("Branches (computed from the routes table)", expanded=False):
    st.dataframe(rows_tbl)
    st.write("Routing matrix: output frequency (GHz) of each carrier through each branch (★ = selected carrier, others are images)")
    st.dataframe(routing_matrix(plan))

pc1, pc2 = st.columns([4, 1])
ptype = pc1.radio("Point type", PTYPES, horizontal=True, key="ptype", format_func=lambda p: PNAME[p])
if st.session_state.get("branch", 1) > plan.N:
    st.session_state["branch"] = 1
branch = pc2.selectbox("Branch", list(range(1, plan.N + 1)), key="branch")
i = branch - 1
k_own = plan.routes[i].carrier
st.graphviz_chart(scheme_dot(plan, ptype, branch, rep, sim, cfg.comps, cfg.imp))
if all(c == "auto" for c in cfg.comps.bpf.center):
    st.caption("BPF retuned to " + ", ".join(f"{bpf_for(plan, sim, cfg.comps.bpf, j)[0]:g}" for j in range(min(plan.N, 8)))
               + " GHz (center = auto)")

tabs = st.tabs(["Spectrum", "Time", "Constellation", "Responses", "Interference", "Noise budget"])
bwsig = (1 + sim.alpha_eff) * sim.Rs
fo = plan.line_info(i)[0]
spacing = abs(plan.carriers[1].f_GHz - plan.carriers[0].f_GHz) if plan.K > 1 else 5 * bwsig


def add_bpf_fill(fig, xr):
    """АЧХ BPF ветки i полупрозрачной заливкой поверх спектра (дБ, нижняя граница -110)."""
    x = np.linspace(max(xr[0], 0.05), min(xr[1], fs / 2e9 * 0.999), 1500)
    y = np.maximum(20 * np.log10(np.abs(bpf_response(fs, plan, sim, cfg.comps.bpf, i, x)) + 1e-9), -110)
    fig.add_trace(go.Scatter(x=x, y=np.full_like(x, -110.0), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=0), fill="tonexty", name="BPF |H|", fillcolor="rgba(0,160,0,0.18)"))


def psd_fig(curves, title, xr, marks=(), band=None, with_bpf=False, ylabel="dB rel. max", xlabel="GHz"):
    """curves: [(имя, f, dB, цвет|None)]; band — маска полосы приёма."""
    fig = go.Figure()
    for name, f, p, color in curves:
        ff, pp = _thin(np.asarray(f), np.asarray(p), xr)
        fig.add_trace(go.Scatter(x=ff, y=pp, mode="lines", name=name, line=dict(width=1, **({"color": color} if color else {}))))
    if with_bpf:
        add_bpf_fill(fig, xr)
    for x, name in marks:
        if xr[0] <= x <= xr[1]:
            fig.add_vline(x=x, line=dict(dash="dot", color="gray", width=1), annotation_text=name,
                          annotation_font_size=9, annotation_textangle=-90)
    if band:
        fig.add_vrect(x0=band[0], x1=band[1], fillcolor="green", opacity=0.2, line_width=0)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, height=400, yaxis_range=[-110, 10],
                      xaxis_range=list(xr), margin=dict(t=40))
    return fig


def spur_marks(i, xr=None):
    """Подписи паразитных линий ветки (образы, IM3 — сгруппированы по частоте)."""
    seen = {}
    for name, fq in plan.spurs(i).items():
        if name.endswith("@+"):
            continue
        lab = "IM3" if name.startswith("im3(") else name.replace("fi", f"f{i + 1}")
        seen.setdefault(round(fq, 4), set()).add(lab)
    return [(f, ", ".join(sorted(v))) for f, v in seen.items() if xr is None or xr[0] <= f <= xr[1]]


def optical_centers(i):
    """Оптические боковые ветки: +f_i и боковые несущих (f_ck - f_i для lower, -(f_i+f_ck) для upper)."""
    fi = plan.f_lo_GHz[i]
    sidx = [(fi, "+f_i")]
    for j, c in enumerate(plan.carriers):
        sidx.append(((c.f_GHz - fi) if plan.routes[i].sideband == "lower" else -(c.f_GHz + fi), f"c{j + 1}"))
    return sidx


with tabs[0]:
    if ptype in ("a", "g"):
        f, p = res[i].fields[ptype]
        xl = max(plan.f_lo_GHz) + max(c.f_GHz for c in plan.carriers) + 10
        zoom = st.radio("Range", ["full", "around the useful sidebands"], horizontal=True, key="opt_zoom")
        marks = [(s * plan.f_lo_GHz[i], f"{s:+d}f_i") for s in (-3, -1, 1, 3)]
        xr = (-xl, xl)
        if ptype == "g":
            marks = [(c, n) for c, n in optical_centers(i)]
            if zoom != "full":
                cs = [c for c, _ in optical_centers(i)[1:]]
                xr = (min(cs) - 1.0, max(cs) + 1.0)
        st.plotly_chart(psd_fig([("optical", f, p, None)], f"Branch {i + 1}, point {ptype}: optical spectrum (offset from the laser)", xr, marks,
                                xlabel="Optical offset, GHz"), use_container_width=True)
    elif ptype in ("d", "e"):
        offs = plan.offsets_GHz()
        xl = max(plan.f_lo_GHz) + max(c.f_GHz for c in plan.carriers) + 10
        curves = [(f"branch {j + 1} (shift {offs[j]:+g})", res[j].fields[ptype][0] + offs[j], res[j].fields[ptype][1], None)
                  for j in range(min(plan.N, MAX_DRAWN))]
        st.plotly_chart(psd_fig(curves, f"Point {ptype}: group spectrum, branches shifted by the WDM spacing",
                                (min(offs[:MAX_DRAWN]) - xl, max(offs[:MAX_DRAWN]) + xl), xlabel="Optical offset, GHz"),
                        use_container_width=True)
    elif ptype == "m(t)":
        meta = res[0].meta
        f, p_in = rf_psd_abs(meta["u2_in"], fs, min(16 * sps, len(meta["u2_in"])))
        ref = p_in.max()
        curves = [("input u2 (with noise, if enabled)", f, 10 * np.log10(p_in / ref + 1e-300), None)]
        if not cfg.comps.mzm2_rf.is_ideal:
            curves.append(("after MZM2 RF response", f, 10 * np.log10(
                rf_psd_abs(meta["u2_mzm"], fs, min(16 * sps, len(meta["u2_in"])))[1] / ref + 1e-300), None))
        lo, hi = plan.band_GHz(bwsig)
        marks = [(c.f_GHz, f"c{j + 1}") for j, c in enumerate(plan.carriers)]
        for ttl, xr in (("full band", (0.0, fs / 2e9)), ("zoom on the input band", (lo - 2 * bwsig, hi + 2 * bwsig))):
            st.plotly_chart(psd_fig(curves, f"m(t): input band PSD, {ttl}", xr, marks, band=(lo, hi)), use_container_width=True)
    else:   # k, BPF
        f, p = rf_psd(res[i].I, fs, nperseg=min(16 * sps, len(res[i].I)))
        band = (fo - bwsig / 2, fo + bwsig / 2)
        xmax = min(fs / 2e9, 1.1 * (2 * plan.f_lo_GHz[i] + 3 * max(c.f_GHz for c in plan.carriers)))
        zr = (max(fo - 3 * spacing, 0.0), fo + 3 * spacing)
        marks = [(fo, "output")] + spur_marks(i)
        if ptype == "k":
            if plan.K == 1:
                ana, approx = expected_lines(i, plan, sim, cfg.comps, bias)
                marks = [(v[0], n) for n, v in ana.items() if n != "DC"]
            st.plotly_chart(psd_fig([("photocurrent", f, p, None)], f"Branch {i + 1}, point k: photocurrent PSD "
                                    "(green: receive band; shaded: BPF |H|)", (0, xmax), marks, band, True), use_container_width=True)
            if plan.K > 1:
                st.plotly_chart(psd_fig([("photocurrent", f, p, None)], "zoom around the output (images and IM3 marked)", zr,
                                        marks, band, True), use_container_width=True)
            else:
                rows, approx = line_table(key, cfg, bias, i)
                st.caption("Line amplitudes, units of R·P₀/4 (noise-free CW run, no bias error)"
                           + (" — approx (WDM not ideal)" if approx else ""))
                st.dataframe(rows)
        else:
            Ib = bpf_out(key, res, cfg, i)
            _, pb = rf_psd(Ib, fs, nperseg=min(16 * sps, len(Ib)))
            pb = pb + 10 * np.log10(rf_psd_abs(Ib, fs)[1].max() / rf_psd_abs(res[i].I, fs)[1].max())
            f0, bwf = bpf_for(plan, sim, cfg.comps.bpf, i)
            curves = [("before BPF", f, p, "gray"), ("after BPF", f, pb, "#1f77b4")]
            st.plotly_chart(psd_fig(curves, f"BPF{i + 1}: spectrum before / after, full band (BPF {f0:g} GHz / {bwf:g} GHz)",
                                    (0, xmax), marks, band, True), use_container_width=True)
            st.plotly_chart(psd_fig(curves, "zoom around the output (neighbour images and IM3 marked)", zr, marks, band, True),
                            use_container_width=True)
            st.caption("Suppression of spurious lines (incl. neighbour images and IM3) within ±f_out of the output "
                       "(power in the signal band, dBc to the useful line):")
            st.dataframe([{"line": r["line"], "f, GHz": r["f_GHz"], "before BPF, dBc": r["before_dBc"],
                           "after BPF, dBc": r["after_dBc"], "suppression, dB": r["suppression_dB"]}
                          for r in suppression(key, res, cfg, i)])

with tabs[1]:
    if ptype == "m(t)":
        meta = res[0].meta
        n = min(8 * sps, len(meta["u2_in"]))
        st.plotly_chart(time_fig(res[0].t[:n], meta["u2_in"][:n], "m(t): u2(t) (MZM2 drive)", "u2, V"), use_container_width=True)
        st.plotly_chart(time_fig(res[0].t[:n], np.abs(meta["x_in"][:n]), "m(t): envelope |x_in(t)| of the composite (with input noise, if enabled)",
                                 "|x|"), use_container_width=True)
    elif ptype in ("a", "d", "e", "g"):
        fr = res[i].meta["frag"][ptype]
        n = min(int(5 * fs / (plan.f_lo_GHz[i] * 1e9)), len(fr))
        st.plotly_chart(time_fig(res[i].t[:n], np.abs(fr[:n]) ** 2, f"Branch {i + 1}, point {ptype}: |E|² (fragment)", "|E|², W"),
                        use_container_width=True)
    elif ptype == "BPF":
        Ib = bpf_out(key, res, cfg, i)
        n = min(8 * sps, len(Ib))
        t = res[i].t[:n] * 1e9
        k = max(1, n // 20000)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[::k], y=res[i].I[:n:k], mode="lines", name="before BPF", line=dict(width=1, color="lightgray")))
        fig.add_trace(go.Scatter(x=t[::k], y=Ib[:n:k], mode="lines", name="after BPF", line=dict(width=1)))
        fig.update_layout(title=f"BPF{i + 1}: RF signal at the amplifier input", xaxis_title="t, ns", yaxis_title="I, A",
                          height=380, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
        env = go.Figure()
        env.add_trace(go.Scatter(x=t[::k], y=np.abs(hilbert(res[i].I[:n]))[::k], mode="lines", name="before BPF",
                                 line=dict(width=1, color="gray")))
        env.add_trace(go.Scatter(x=t[::k], y=np.abs(hilbert(Ib[:n]))[::k], mode="lines", name="after BPF", line=dict(width=1)))
        env.update_layout(title="Envelope", xaxis_title="t, ns", yaxis_title="|I_a|, A", height=340, margin=dict(t=40))
        st.plotly_chart(env, use_container_width=True)
    else:
        n = min(int(5 * fs / (plan.f_lo_GHz[i] * 1e9)), len(res[i].I))
        st.plotly_chart(time_fig(res[i].t[:n], res[i].I[:n], f"Branch {i + 1}, point k: photocurrent", "I, A"), use_container_width=True)

with tabs[2]:
    jopts = list(range(1, plan.K + 1))
    jsel = st.selectbox("Carrier to demodulate", jopts, index=k_own, key=f"carrier_sel_{branch}_{plan.K}",
                        format_func=lambda j: f"c{j} ({plan.carriers[j - 1].f_GHz:g} GHz)" + (" — routed" if j - 1 == k_own else " — image"))
    j = jsel - 1
    if ptype == "m(t)":
        ref_bb = tx.ref(j)
        sy = ref_bb.symbols
        xs = ref_bb.x[::sps]
        z_in = res[0].meta["x_in"][::sps] if plan.K == 1 else None
        traces = [(xs, f"carrier {j + 1} after {sim.pulse} (clean)", {})]
        if z_in is not None:
            traces = [(z_in, "x_in after pulse shaping (symbol instants)", {})]
        traces.append((sy, "transmitted symbols", dict(size=9, symbol="x", color="red", opacity=1.0)))
        st.plotly_chart(const_fig(traces, f"m(t): carrier {j + 1} constellation"), use_container_width=False)
    elif ptype in ("k", "BPF"):
        f, sb, m, z, ze = rx_metrics(key, res, tx, cfg, i, j, ptype == "BPF")
        st.write(("After BPF — " if ptype == "BPF" else "") + metric_text(m, f, sb, j))
        if ptype == "BPF":
            _, _, m0, _, _ = rx_metrics(key, res, tx, cfg, i, j, False)
            st.write("Without BPF — " + metric_text(m0, f, sb, j))
        traces = [(z, "without equalizer", {})]
        if ze is not None:
            traces.append((ze, "with equalizer", {}))
        st.plotly_chart(const_fig(traces, f"branch {i + 1}, {ptype}: carrier {j + 1} at {f:g} GHz"), use_container_width=False)
    else:
        st.info("Constellation is shown for the input m(t), the photodiode (k) and after the BPF.")

with tabs[3]:
    ri = 0 if ptype in ("d", "e", "m(t)") else i
    v = response_view(ptype, plan, sim, cfg.comps, fs, ri)
    fig = go.Figure()
    for name, x, y in v["curves"]:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    if v["gd"] is not None:
        fig.add_trace(go.Scatter(x=v["gd"][0], y=v["gd"][1], mode="lines", name="group delay, ns", yaxis="y2", line=dict(dash="dash")))
        fig.update_layout(yaxis2=dict(title="group delay, ns", overlaying="y", side="right"))
    for x, name in v["marks"]:
        fig.add_vline(x=x, line=dict(dash="dot", color="gray", width=1), annotation_text=name,
                      annotation_font_size=9, annotation_textangle=-90)
    if v["band"]:
        fig.add_vrect(x0=v["band"][0], x1=v["band"][1], fillcolor="green", opacity=0.15, line_width=0)
    fig.update_layout(title=f"Point {ptype}: {v['title']}", xaxis_title=v["xlabel"], yaxis_title="|H|, dB",
                      yaxis_range=[-40, 3], height=440, margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe([{"line": r["line"], "f, GHz": r["f_GHz"], "attenuation, dB": r["att_dB"]} for r in v["table"]])
    if st.checkbox("Cumulative response up to this point", key="chain_resp"):
        ch = chain_attenuation(ptype, plan, sim, cfg.comps, fs, ri)
        st.dataframe([{"stage": r["stage"], "f, GHz": round(r["f_GHz"], 3), "attenuation, dB": round(r["att_dB"], 2),
                       "cumulative, dB": round(r["cum_dB"], 2)} for r in ch])
        cb = go.Figure(go.Bar(x=[r["stage"] for r in ch], y=[-r["cum_dB"] for r in ch]))
        cb.update_layout(title=f"Useful-line chain, branch {ri + 1}: cumulative |H|, dB (stages: {', '.join(STAGES)}; "
                         "MZM RF losses act on the modulation drive)", yaxis_title="dB", height=320, margin=dict(t=50))
        st.plotly_chart(cb, use_container_width=True)

with tabs[4]:
    ci_store = st.session_state.setdefault("ci_results", {})
    npr_store = st.session_state.setdefault("npr_results", {})
    bt = branch_table(key, res, tx, cfg)
    rows = []
    for b in bt:
        c = ci_store.get((key, b["i"]))
        rows.append({"branch": b["i"] + 1, "carrier": b["carrier"] + 1, "f_out, GHz": round(plan.line_info(b["i"])[0], 3),
                     "EVM, %": round(b["evm"], 2), "EVM with eq., %": None if b["evm_eq"] is None else round(b["evm_eq"], 2),
                     "BER": f"{b['ber']:.1e}", "C/I, dB": None if c is None else round(c["ci_dB"], 1),
                     "C/ACI, dB": None if c is None else round(c["ci_aci_dB"], 1),
                     "C/IMD, dB": None if c is None else round(c["ci_imd_dB"], 1),
                     "C/LO-spur, dB": None if c is None else round(c["ci_lo_dB"], 1),
                     "NPR, dB": npr_store.get((key, b["i"]))})
    st.dataframe(rows)
    st.caption("EVM/BER are from the main run (after the BPF). C/I (runs 0-3), NPR and the drive sweep are computed on demand "
               "for the selected branch; they need K ≥ 2 carriers.")
    multi = plan.K > 1
    b1, b2, b3 = st.columns(3)
    if b1.button("Compute C/I (ACI/IMD)", key="ci_btn", disabled=not multi):
        ci_store[(key, i)] = ci_branch(key, cfg, i)
        st.rerun()
    if b2.button("Compute NPR", key="npr_btn", disabled=not multi):
        npr_store[(key, i)] = round(float(npr_branch(key, cfg, i)), 1)
        st.rerun()
    betas_txt = b3.text_input("Drive sweep β_m values", value="0.1, 0.33, 1.0", key="sweep_betas")
    if b3.button("Drive sweep", key="sweep_btn", disabled=not multi):
        try:
            betas = tuple(float(x) for x in betas_txt.split(",") if x.strip())
            st.session_state["sweep_result"] = (key, i, *sweep_branch(key, cfg, i, betas))
        except ValueError:
            st.error("β_m values must be comma-separated numbers")
    sw = st.session_state.get("sweep_result")
    if sw and sw[0] == key and sw[1] == i:
        srows, best = sw[2], sw[3]
        st.dataframe([{"β_m": r["beta"], "EVM, %": round(r["evm_pct"], 2), "C/(N+I), dB": round(r["cni_dB"], 1),
                       "C/ACI, dB": round(r["ci_aci_dB"], 1), "C/IMD, dB": round(r["ci_imd_dB"], 1)} for r in srows])
        sf = go.Figure(go.Scatter(x=[r["beta"] for r in srows], y=[r["cni_dB"] for r in srows], mode="lines+markers"))
        sf.add_vline(x=best, line=dict(dash="dot"), annotation_text=f"β* = {best:g}")
        sf.update_layout(title=f"Branch {i + 1}: C/(N+I) vs β_m", xaxis_title="β_m", yaxis_title="dB", height=340, margin=dict(t=50))
        st.plotly_chart(sf, use_container_width=True)

with tabs[5]:
    st.caption(f"Contribution of each noise source to the output SNR on branch {branch} (the selected one). "
               "Takes several simulations, so it is computed on demand.")
    cn_b = st.number_input("C/N at the input for the budget, dB", key="budget_cn", value=20.0)
    if st.button("Compute noise budget", key="budget_btn"):
        rows = budget_rows(key, cfg, i, cn_b)
        st.dataframe([{"source": r["name"], "EVM, %": round(r["evm_pct"], 2), "SNR alone, dB": round(r["snr_dB"], 1),
                       "Δ(1/SNR)": f"{r['delta_inv']:.2e}"} for r in rows])
        bf = go.Figure(go.Bar(x=[r["name"] for r in rows], y=[r["snr_dB"] for r in rows]))
        bf.update_layout(title=f"Branch {branch}: SNR if the source acts alone (dB)", height=380, margin=dict(t=50))
        st.plotly_chart(bf, use_container_width=True)
