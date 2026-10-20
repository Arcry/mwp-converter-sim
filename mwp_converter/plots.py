"""Рисунки fig1..fig8 (+fig3b). Каждая функция возвращает (список (имя, Figure), строки EVM/BER)."""
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analytic import beta1_opt, line_freq_GHz, lines_null
from .model import measure_lines, run_all, simulate
from .params import Components, Impairments, Plan, load_config, quick as quick_cfg, resolve_sps, scenario_plans
from .response import band_report_for_output, worst_level
from .budget import noise_budget
from .bpf import apply_bpf, bpf_for, suppression_table
from .signals import carrier_weights, make_multicarrier_baseband, measure, metrics, receive_line
from scipy.signal import hilbert
from .spectra import optical_spectrum, rf_psd, rf_psd_abs

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------- вспомогательное ----------
def realistic_parts(cfg=None):
    """(Impairments, Components) из configs/realistic.toml."""
    r = load_config(CONFIG_DIR / "realistic.toml")
    return r.imp, r.comps


def ideal_parts():
    return Impairments(), Components()


def _sub(plan, flo, direction=None):
    return Plan(plan.fc_GHz, tuple(flo), direction or plan.direction, plan.wdm_spacing_GHz)


def _thin(f, p, xlim, nmax=20000):
    m = (f >= xlim[0]) & (f <= xlim[1])
    f, p = f[m], p[m]
    if len(f) > nmax:
        k = int(np.ceil(len(f) / nmax))
        n = len(f) // k * k
        f, p = f[:n].reshape(-1, k).mean(1), p[:n].reshape(-1, k).max(1)
    return f, p


def _spec(ax, f, p, xlim, **kw):
    ff, pp = _thin(f, p, xlim)
    ax.plot(ff, pp, lw=0.8, **kw)
    ax.set_xlim(xlim)


def _mask(ax, f0, sim, label=None):
    bw = (1 + sim.alpha_eff) * sim.Rs
    ax.axvspan(f0 - bw / 2, f0 + bw / 2, color="tab:green", alpha=0.25, label=label)


def _label(ax, f, p, xlim, items, dy=3, rot=90):
    """Подписи линий items = [(freq_GHz, text)] над спектром (f, p)."""
    for fr, txt in items:
        if not (xlim[0] <= fr <= xlim[1]):
            continue
        m = np.abs(f - fr) < 0.4
        y = p[m].max() if m.any() else -90
        ax.annotate(txt, (fr, y), xytext=(0, 4), textcoords="offset points", rotation=rot,
                    ha="center", va="bottom", fontsize=7)


def _rf_items(plan, i, names=None):
    fi, fc = plan.f_lo_GHz[i], plan.fc_GHz
    lab = f"{i + 1}".translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    names = names or ["fc", "2fi", "2fi+fc", "2fi-fc", "3fc", "2fi+3fc", "2fi-3fc", "4fi+fc", "4fi-fc"]
    seen = {}
    for n in names:
        f = line_freq_GHz(n, fi, fc)
        seen.setdefault(round(f, 6), []).append(n.replace("fi", "f" + lab))
    return [(f, "/".join(v) + f" {f:g}") for f, v in seen.items()]


def _psd_axes(ax, res, fs, plan, i, sim, xmax=None, names=None, mask=True, **kw):
    f, p = rf_psd(res.I, fs)
    xmax = xmax or min(fs / 2e9, 1.1 * (2 * plan.f_lo_GHz[i] + 3 * plan.fc_GHz))
    _spec(ax, f, p, (0, xmax), **kw)
    ax.set_ylim(-110, 10)
    ax.set_xlabel("Frequency, GHz")
    ax.set_ylabel("PSD, dB rel. max")
    if mask:
        _mask(ax, plan.line_info(i)[0], sim)
    _label(ax, f, p, (0, xmax), _rf_items(plan, i, names))
    return f, p


def _rx(res, tx, fs, plan, sim, i, which=None):
    f, sb = plan.line_info(i, which)
    return f, sb, measure(res.I, fs, f, sb, sim, tx, carrier=plan.routes[i].carrier)


def _const(ax, m, title):
    z = m["rx_norm"]
    ax.scatter(z.real, z.imag, s=2, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")


def _row(rows, fig, label, f, m):
    rows.append((fig, label, f, m["evm_pct"], m["ber_mc"], m["n_err"], m["ber_evm"]))


def _sub_i(s):
    return s.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def _line_title(plan, i, which, f, sb):
    lab = _sub_i(str(i + 1))
    nm = {"upper": f"2f{lab}+fc", "lower": f"|2f{lab}-fc|", "fc": "fc", "3fc": "3fc"}[which]
    inv = ", conjugated (inverted)" if sb == "-" else ""
    return f"{f:g} GHz ({nm}){inv}"


# ---------- рисунки ----------
def fig1(cfg, quick):
    plan, sim, imp, comps = cfg.plan, cfg.sim, cfg.imp, cfg.comps
    rows = []
    sps = resolve_sps(plan, sim)
    from .signals import make_psk_baseband

    tx = make_psk_baseband(sim.M, sim.n_sym, sps, sim.alpha, sim.seed, sim.pulse)
    res = run_all(plan, sim, imp, comps, baseband=tx)
    fs = res[0].fs
    fig, ax = plt.subplots(3, 2, figsize=(12, 12))
    xl = plan.f_lo_GHz[0] + plan.fc_GHz + 10
    f, p = optical_spectrum(res[0].fields["a"], fs)
    _spec(ax[0, 0], f, p, (-xl, xl))
    ax[0, 0].set_title("(a) Optical spectrum after MZM1, channel 1")
    ax[0, 0].set_ylim(-90, 5)
    offs = plan.offsets_GHz()
    for k, r in enumerate(res):
        f, p = optical_spectrum(r.fields["d"], fs)
        _spec(ax[0, 1], f + offs[k], p, (min(offs) - xl, max(offs) + xl), color=COLORS[k],
              label=f"ch{k + 1}: f={plan.f_lo_GHz[k]:g} GHz (shift {offs[k]:+g})")
    ax[0, 1].set_title("(b) Group spectrum at point d (channels shifted by WDM spacing)")
    ax[0, 1].set_ylim(-90, 5)
    ax[0, 1].legend(fontsize=7)
    f, p = optical_spectrum(res[0].fields["e"], fs)
    _spec(ax[1, 0], f, p, (-xl, xl))
    ax[1, 0].set_title("(c) Optical spectrum after MZM2, channel 1")
    ax[1, 0].set_ylim(-90, 5)
    _psd_axes(ax[1, 1], res[0], fs, plan, 0, sim, label="photocurrent")
    ax[1, 1].set_title("(d) RF photocurrent, channel 1 (green: receive band)")
    u_in, u_m = res[0].meta["u2_in"], res[0].meta["u2_mzm"]
    f, p_in = rf_psd_abs(u_in, fs, 2 ** 15)
    ref = p_in.max()
    _, p_m = rf_psd_abs(u_m, fs, 2 ** 15)
    fc = plan.fc_GHz
    for a, xr in ((ax[2, 0], (0, fs / 2e9)), (ax[2, 1], (fc - 3 * sim.Rs, fc + 3 * sim.Rs))):
        _spec(a, f, 10 * np.log10(p_in / ref + 1e-300), xr, label="input u2")
        if not comps.mzm2_rf.is_ideal:
            _spec(a, f, 10 * np.log10(p_m / ref + 1e-300), xr, label="after MZM2 RF response")
        a.set_ylim(-100, 5)
        a.set_xlabel("Frequency, GHz")
        a.set_ylabel("PSD, dB rel. input max")
        a.legend(fontsize=7)
    _mask(ax[2, 1], fc, sim)
    ax[2, 0].set_title("(e) Input PM signal u2 (MZM2 drive), full band")
    ax[2, 1].set_title("(f) Input PM signal u2, zoom fc ± 3Rs (green: (1+α)Rs)")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("Offset from optical carrier, GHz")
        a.set_ylabel("Power, dB rel. max")
    fig.suptitle(f"fc={plan.fc_GHz:g} GHz, f1={plan.f_lo_GHz[0]:g} GHz, beta1={sim.beta1:g}, "
                 f"beta_m={sim.beta_m:g}; fs={fs / 1e9:g} GHz", fontsize=9)
    fig.tight_layout()
    return [("fig1_points_spectra", fig)], rows


def fig2(cfg, quick):
    plan, sim, imp, comps = cfg.plan, cfg.sim, cfg.imp, cfg.comps
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    for a, bias in zip(ax, ("null", "quad")):
        res, tx, fs = simulate(plan, sim, imp, comps, 0, bias)
        names = None if bias == "null" else ["fi", "fi+fc", "fi-fc", "3fi", "3fi+fc", "3fi-fc", "fc", "3fc"]
        _psd_axes(a, res, fs, plan, 0, sim, names=names)
        a.set_title(f"MZM1 biased at {'null' if bias == 'null' else 'quadrature'}")
    ax[1].text(0.99, 0.9, "Lines fi, fi±fc, 3fi±fc (paper Fig. 4) appear only at quadrature",
               transform=ax[1].transAxes, ha="right", fontsize=8)
    fig.tight_layout()
    return [("fig2_bias_comparison", fig)], []


def fig3(cfg, quick):
    plan, sim, imp, comps = cfg.plan, cfg.sim, cfg.imp, cfg.comps
    rows = []
    res, tx, fs = simulate(plan, sim, imp, comps, 0)
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    for a, which in zip(ax.ravel(), ("upper", "lower", "fc", "3fc")):
        f, sb, m = _rx(res, tx, fs, plan, sim, 0, which)
        _const(a, m, f"{_line_title(plan, 0, which, f, sb)}\nEVM={m['evm_pct']:.1f}%")
        _row(rows, "fig3", which, f, m)
    fig.tight_layout()
    fig_b, axb = plt.subplots(1, 3, figsize=(12, 4.2))
    for a, M in zip(axb, (2, 4, 8)):
        s = replace(sim, M=M)
        r, tx, fs = simulate(plan, s, imp, comps, 0)
        f, sb, m = _rx(r, tx, fs, plan, s, 0, "3fc")
        _const(a, m, f"{M}-PSK, 3fc = {f:g} GHz, phase x3\nEVM={m['evm_pct']:.1f}%")
        _row(rows, "fig3b", f"M={M}", f, m)
    fig_b.tight_layout()
    return [("fig3_constellations", fig), ("fig3b_triple_phase", fig_b)], rows


def _sweep_lines(cfg, names, key, values, other):
    """Измеренные амплитуды names при свипе sim.<key>=values; возвращает dict имя -> array."""
    plan = _sub(cfg.plan, [cfg.sweeps.get("fig4_flo_GHz", 9.0)])
    out = {n: [] for n in names}
    for v in values:
        sim = replace(cfg.sim, **{key: v, **other})
        m = measure_lines(0, plan, sim, cfg.comps, names, imp=Impairments())
        for n in names:
            out[n].append(m[n][1])
    return plan, {n: np.array(v) for n, v in out.items()}


def fig4(cfg, quick):
    sw = cfg.sweeps
    npt = sw.get("n_beta_quick", 8) if quick else sw.get("n_beta", 30)
    plan = _sub(cfg.plan, [sw.get("fig4_flo_GHz", 9.0)])
    fi, fc = plan.f_lo_GHz[0], plan.fc_GHz
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = [("beta1", np.linspace(sw["beta1_min"], sw["beta1_max"], npt), dict(beta_m=0.5),
               ["2fi", "2fi+fc", "fc", "4fi+fc"], "$\\beta_1$ ($\\beta_m=0.5$)"),
              ("beta_m", np.linspace(sw["beta_m_min"], sw["beta_m_max"], npt), dict(beta1=1.0),
               ["fc", "2fi+fc", "3fc", "2fi+3fc"], "$\\beta_m$ ($\\beta_1=1$)")]
    for a, (key, vals, other, names, xl) in zip(ax, panels):
        _, meas = _sweep_lines(replace(cfg, plan=plan), names, key, vals, other)
        dense = np.linspace(vals.min(), vals.max(), 300)
        for k, n in enumerate(names):
            ana = np.array([lines_null(**{"beta1": 1.0, "beta_m": 0.5, **other, key: d})[n] for d in dense])
            a.plot(dense, 20 * np.log10(np.abs(ana) + 1e-9), color=COLORS[k], lw=1,
                   label=f"{n.replace('fi', 'f1')} = {line_freq_GHz(n, fi, fc):g} GHz")
            a.plot(vals, 20 * np.log10(meas[n] + 1e-9), "o", color=COLORS[k], ms=4)
        a.set_xlabel(xl)
        a.set_ylabel("Amplitude, dB rel. $\\Re P_0/4$")
        a.set_ylim(-60, 5)
        a.legend(fontsize=7)
    ax[0].axvline(beta1_opt(), color="k", ls=":")
    ax[0].text(beta1_opt(), -55, " $\\beta_1^{opt}$", fontsize=8)
    ax[0].set_title(f"(a) f1={fi:g} GHz, fc={fc:g} GHz; curves: (13'), markers: model")
    ax[1].set_title("(b)")
    fig.tight_layout()
    return [("fig4_lines_vs_beta", fig)], []


def fig5(cfg, quick):
    sim, imp, comps = cfg.sim, cfg.imp, cfg.comps
    cases = [(cfg.plan, i, f"Down-conversion, fc={cfg.plan.fc_GHz:g} GHz") for i in range(len(cfg.plan.f_lo_GHz))]
    for name, p in scenario_plans(cfg).items():
        cases += [(p, i, f"Scenario '{name}', fc={p.fc_GHz:g} GHz") for i in range(len(p.f_lo_GHz))]
    fig, ax = plt.subplots(len(cases), 1, figsize=(11, 3.3 * len(cases)))
    rows = []
    for a, (plan, i, ttl) in zip(np.atleast_1d(ax), cases):
        res, tx, fs = simulate(plan, sim, imp, comps, i)
        _psd_axes(a, res, fs, plan, i, sim)
        f, sb, m = _rx(res, tx, fs, plan, sim, i)
        inv = " (spectrum inverted, conjugation needed)" if sb == "-" else " (no inversion)"
        a.set_title(f"{ttl}: f{_sub_i(str(i + 1))}={plan.f_lo_GHz[i]:g} -> {f:g} GHz{inv}; EVM={m['evm_pct']:.1f}%",
                    fontsize=9)
        _row(rows, "fig5", f"{ttl} f={plan.f_lo_GHz[i]:g}", f, m)
    fig.tight_layout()
    return [("fig5_multiband", fig)], rows


def fig6(cfg, quick):
    sw = cfg.sweeps
    plan, sim = cfg.plan, cfg.sim
    rimp, rcomps = realistic_parts()
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    nb = sw.get("n_evm_quick", 6) if quick else sw.get("n_evm", 15)
    bms = np.linspace(sw["beta_m_min"], sw["beta_m_max"], nb)
    for lab, (imp, comps) in (("ideal", ideal_parts()), ("realistic", (rimp, rcomps))):
        evm = []
        for bm in bms:
            s = replace(sim, beta_m=bm)
            res, tx, fs = simulate(plan, s, imp, comps, 0)
            f, sb, m = _rx(res, tx, fs, plan, s, 0)
            evm.append(m["evm_pct"])
        ax[0].plot(bms, evm, "o-", ms=4, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("$\\beta_m$")
    ax[0].set_ylabel("EVM, %")
    ax[0].set_title(f"(a) EVM at {f:g} GHz")
    ax[0].legend()
    npw = sw.get("n_rx_quick", 5) if quick else sw.get("n_rx", 11)
    pw = np.linspace(sw["rx_dBm_min"], sw["rx_dBm_max"], npw)
    mc, ev = [], []
    for p in pw:
        imp = replace(rimp, rx_power_dBm=float(p))
        res, tx, fs = simulate(plan, sim, imp, rcomps, 0)
        f, sb, m = _rx(res, tx, fs, plan, sim, 0)
        mc.append((m["n_err"], m["n_bits"]))
        ev.append(m["ber_evm"])
        _row(rows, "fig6b", f"rx={p:.1f} dBm", f, m)
    floor = 1e-7
    for p, (ne, nb_) in zip(pw, mc):
        full = ne >= 100
        ax[1].plot(p, max(ne, 1) / nb_, "o", color="tab:blue", mfc="tab:blue" if full else "none")
    ax[1].plot([], [], "o", color="tab:blue", label="Monte Carlo (hollow: <100 errors, upper bound)")
    ax[1].plot(pw, np.maximum(ev, floor), "s-", color="tab:red", ms=3, label="from EVM via theoryBER")
    ax[1].set_yscale("log")
    ax[1].set_ylim(floor, 1)
    ax[1].set_xlabel("Received optical power, dBm")
    ax[1].set_ylabel("BER")
    ax[1].set_title(f"(b) BER at {f:g} GHz, realistic")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    return [("fig6_evm_ber", fig)], rows


def fig7(cfg, quick):
    plan, sim = cfg.plan, cfg.sim
    sw = cfg.sweeps
    rows = []
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 6)
    axa, axa2, axb = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:])
    xl = plan.f_lo_GHz[0] + 10
    for k, er in enumerate(sw.get("er_dB", [300.0, 40.0, 30.0, 20.0])):
        imp = replace(Impairments(), er_dB=None if er >= 100 else er)
        res, tx, fs = simulate(plan, sim, imp, cfg.comps, 0)
        f, p = optical_spectrum(res.fields["a"], fs)
        _spec(axa, f, p, (-xl, xl), color=COLORS[k], label="ER = inf" if er >= 100 else f"ER = {er:g} dB")
    axa.set_ylim(-90, 5)
    axa.set_xlabel("Offset from optical carrier, GHz")
    axa.set_ylabel("Power, dB rel. max")
    axa.set_title("(a1) Finite ER: optical carrier leak (point a)", fontsize=9)
    axa.legend(fontsize=7)
    fi0, fc0 = plan.f_lo_GHz[0], plan.fc_GHz
    xr = (0, 2 * fi0 + fc0 + 10)
    for k, e in enumerate(sw.get("fig7_bias_err", [0.0, 0.01, 0.03, 0.1])):
        imp = Impairments(bias_err=e)
        res, tx, fs = simulate(plan, sim, imp, cfg.comps, 0)
        f, p = rf_psd(res.I, fs)
        m = measure_lines(0, plan, sim, cfg.comps, ["fi+fc", "2fi+fc"], imp=imp)
        r = 20 * np.log10(max(m["fi+fc"][1], 1e-8) / m["2fi+fc"][1])
        _spec(axa2, f, p, xr, color=COLORS[k], label=f"dVb/Vpi = {e * 100:g}%: f1±fc "
              + ("< -100" if r < -100 else f"{r:.1f}") + " dB rel. 2f1±fc")
    for n in ("fi", "fi+fc", "fi-fc"):
        fx = line_freq_GHz(n, fi0, fc0)
        axa2.axvline(fx, color="gray", ls=":", lw=0.8)
        axa2.text(fx + 0.3, -108, n.replace("fi", "f1"), fontsize=7, rotation=90)
    axa2.set_ylim(-110, 10)
    axa2.set_xlabel("Frequency, GHz")
    axa2.set_ylabel("PSD, dB rel. max")
    axa2.set_title("(a2) MZM1 bias drift (ER = inf)", fontsize=9)
    axa2.legend(fontsize=6)
    fi = plan.f_lo_GHz[0]
    rimp, rcomps = realistic_parts()
    for k, (lab, (imp, comps)) in enumerate((("ideal", ideal_parts()), ("realistic.toml", (rimp, rcomps)))):
        res, tx, fs = simulate(plan, sim, imp, comps, 0)
        f, p = rf_psd(res.I, fs)
        txt = []
        for which in ("lower", "upper"):
            fo, sb, m = _rx(res, tx, fs, plan, sim, 0, which)
            txt.append(f"{fo:g} GHz: EVM {m['evm_pct']:.1f}%")
            _row(rows, "fig7b", f"{lab} {which}", fo, m)
        _spec(axb, f, p, (0, min(fs / 2e9, 1.1 * (2 * fi + 3 * plan.fc_GHz))), color=COLORS[k],
              label=f"{lab}: " + "; ".join(txt))
    axb.set_ylim(-120, 10)
    axb.set_xlabel("Frequency, GHz")
    axb.set_ylabel("PSD, dB rel. max")
    axb.set_title("(b) Ideal vs realistic.toml")
    axb.legend(fontsize=7)
    axs = fig.add_subplot(gs[1, :2])
    cands = sw.get("fig7_flo_compare_GHz", [5.0, 5.5])
    for k, f1 in enumerate(cands):
        pl = _sub(plan, [f1], "lower")
        res, tx, fs = simulate(pl, sim, Impairments(), Components(), 0)
        fo, sb, m = _rx(res, tx, fs, pl, sim, 0)
        f, p = rf_psd(res.I, fs)
        _spec(axs, f - fo, p, (-6, 6), color=COLORS[k], label=f"f1={f1:g} GHz, out {fo:g} GHz")
        warn = pl.check(sim.Rs, sim.alpha_eff)
        axs.text(0.02, 0.92 - 0.1 * k, "; ".join(w["msg"] for w in warn) or f"f1={f1:g}: plan check OK",
                 transform=axs.transAxes, fontsize=6, color="tab:red" if warn else "tab:green")
        axc = fig.add_subplot(gs[1, 2 + 2 * k:4 + 2 * k])
        _const(axc, m, f"f1={f1:g} GHz, out {fo:g} GHz\nEVM={m['evm_pct']:.1f}%")
        _row(rows, "fig7c", f"f1={f1:g}", fo, m)
    _mask(axs, 0, sim)
    axs.set_ylim(-110, 10)
    axs.set_xlabel("Offset from output line, GHz")
    axs.set_ylabel("PSD, dB rel. max")
    axs.set_title("(c) Spurious 4f1 in the receive band", fontsize=9)
    axs.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return [("fig7_impairments", fig)], rows


def fig8(cfg, quick):
    sw = cfg.sweeps
    plan, sim = cfg.plan, cfg.sim
    rimp, rcomps = realistic_parts()
    npt = sw.get("n_fi_quick", 8) if quick else sw.get("n_fi", 21)
    fis = np.round(np.linspace(sw["fi_min_GHz"], sw["fi_max_GHz"], npt), 2)
    ideal = Components()
    variants = {"ideal": ideal, "realistic": rcomps,
                "MZM1 only": replace(ideal, mzm1_rf=rcomps.mzm1_rf),
                "PD only": replace(ideal, pd=rcomps.pd),
                "WDM only": replace(ideal, wdm=rcomps.wdm, wdm_isolation_dB=rcomps.wdm_isolation_dB)}
    data = {k: [] for k in variants}
    xs, fouts, evm_i, evm_r, lev = [], [], [], [], []
    rows = []
    for f1 in fis:
        pl = _sub(plan, [f1])
        fo, sb = pl.line_info(0)
        if fo < 1.0:
            continue
        xs.append(f1)
        fouts.append(fo)
        name = "2fi-fc" if pl.routes[0].sideband == "lower" else "2fi+fc"
        for k, c in variants.items():
            data[k].append(measure_lines(0, pl, sim, c, [name])[name][1])
        lev.append(worst_level(band_report_for_output(pl, rcomps, sim.Rs, sim.alpha_eff, 1)))
        for lst, (imp, c) in ((evm_i, ideal_parts()), (evm_r, (rimp, rcomps))):
            res, tx, fs = simulate(pl, sim, imp, c, 0)
            f, sb, m = _rx(res, tx, fs, pl, sim, 0)
            lst.append(m["evm_pct"])
            if c is rcomps:
                _row(rows, "fig8", f"fi={f1:g}", f, m)
    xs = np.array(xs)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    styles = {"ideal": ("k", "-"), "realistic": ("tab:red", "-"), "MZM1 only": ("tab:blue", "--"),
              "PD only": ("tab:green", "--"), "WDM only": ("tab:orange", "--")}
    for k, v in data.items():
        ax[0].plot(xs, 20 * np.log10(np.array(v) + 1e-9), color=styles[k][0], ls=styles[k][1], label=k)
    ax[1].plot(xs, evm_i, "k-o", ms=3, label="ideal")
    ax[1].plot(xs, evm_r, "r-o", ms=3, label="realistic")
    edges = np.concatenate([[xs[0]], (xs[1:] + xs[:-1]) / 2, [xs[-1]]])
    half = plan.fc_of(0) / 2
    for a in ax:
        for j, l in enumerate(lev):
            if l != "ok":
                a.axvspan(edges[j], edges[j + 1], color="gold" if l == "warn" else "tomato", alpha=0.2, lw=0)
        a.set_xlabel("LO frequency f_i, GHz")
        a.legend(fontsize=7)
        if plan.routes[0].sideband == "lower" and xs.min() < half < xs.max():
            a.axvline(half, color="gray", ls="--", lw=1)
            a.text(half, a.get_ylim()[0], " output → 0 / inversion boundary", fontsize=7, rotation=90,
                   va="bottom", ha="right")
        top = a.twiny()
        top.set_xlim(a.get_xlim())
        step = max(1, len(xs) // 8)
        top.set_xticks(xs[::step])
        top.set_xticklabels([f"{o:g}" for o in np.array(fouts)[::step]], fontsize=7)
        top.set_xlabel("Output frequency, GHz", fontsize=8)
    ax[0].set_ylabel("Output line amplitude, dB rel. $\\Re P_0/4$")
    ax[1].set_ylabel("EVM, %")
    ax[0].set_title(f"(a) Output line vs f_i, fc={plan.fc_GHz:g} GHz (shaded: warn/out)", y=1.12)
    ax[1].set_title("(b) EVM at output", y=1.12)
    fig.text(0.5, 0.005, "'WDM only' can lie slightly above 'ideal': the filter removes higher-order optical "
             "sidebands that also beat into this line.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return [("fig8_bandwidth", fig)], rows


def fig9(cfg, quick):
    plan, sim, imp, comps = cfg.plan, cfg.sim, cfg.imp, cfg.comps
    res, tx, fs = simulate(plan, sim, imp, comps, 0)
    I = res.I
    Ib = apply_bpf(I, fs, plan, sim, comps.bpf, 0)
    f0, bw = bpf_for(plan, sim, comps.bpf, 0)
    fo, sb = plan.line_info(0)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    f, p = rf_psd(I, fs)
    _, pb = rf_psd(Ib, fs)
    pb = pb + (10 * np.log10(np.max(rf_psd_abs(Ib, fs)[1]) / np.max(rf_psd_abs(I, fs)[1])))
    xmax = min(fs / 2e9, 1.1 * (2 * plan.f_lo_GHz[0] + 3 * plan.fc_GHz))
    _spec(ax[0], f, p, (0, xmax), color="gray", label="before BPF")
    _spec(ax[0], f, pb, (0, xmax), color="tab:blue", label="after BPF")
    ax[0].axvspan(f0 - bw / 2, f0 + bw / 2, color="tab:green", alpha=0.2, label=f"BPF {f0:g} GHz / {bw:g} GHz")
    ax[0].set_ylim(-120, 10)
    ax[0].set_xlabel("Frequency, GHz")
    ax[0].set_ylabel("PSD, dB rel. max before BPF")
    ax[0].set_title("(a) Spectrum before / after BPF")
    ax[0].legend(fontsize=7)
    sps = resolve_sps(plan, sim)
    n0, n1 = 40 * sps, 60 * sps
    t = res.t[n0:n1] * 1e9
    ax[1].plot(t, np.abs(hilbert(I[n0:n1])), color="gray", lw=0.8, label="photocurrent before BPF")
    ax[1].plot(t, np.abs(hilbert(Ib[n0:n1])), color="tab:blue", lw=0.8, label="after BPF (useful line)")
    ax[1].set_xlabel("t, ns")
    ax[1].set_ylabel("|analytic signal|, A")
    ax[1].set_title("(b) Magnitude of the analytic photocurrent")
    ax[1].legend(fontsize=7)
    rows = []
    out = {}
    for nm, sig in (("before", I), ("after", Ib)):
        rx, s_ref = receive_line(sig, fs, fo, sb, sim, tx)
        out[nm] = metrics(rx, s_ref, sim.M, sim.rx_block)
        _row(rows, "fig9", f"{nm} BPF", fo, out[nm])
    _const(ax[2], out["after"], f"(c) After BPF {fo:g} GHz: EVM={out['after']['evm_pct']:.1f}% "
           f"(without BPF {out['before']['evm_pct']:.1f}%)")
    fig.tight_layout()
    sup = suppression_table(I, Ib, fs, plan, sim, 0)
    lines = [f"BPF {f0:g} GHz / {bw:g} GHz ({comps.bpf.kind}, order {comps.bpf.order}); spurious line suppression:",
             f"  {'line':10}{'f,GHz':>8}{'before,dBc':>12}{'after,dBc':>11}{'suppr,dB':>10}"]
    lines += [f"  {r['line']:10}{r['f_GHz']:8.2f}{r['before_dBc']:12.1f}{r['after_dBc']:11.1f}"
              f"{r['suppression_dB']:10.1f}" for r in sup]
    return [("fig9_bpf_output", fig)], rows, lines


def fig10(cfg, quick):
    sw = cfg.sweeps
    rimp, rcomps = realistic_parts()
    if quick:
        cns = np.linspace(sw.get("cn_min_dB", 5.0), sw.get("cn_max_dB", 35.0), sw.get("n_cn_quick", 5) - 1)
        cns = np.unique(np.append(cns, 10.0))
    else:
        cns = np.linspace(sw.get("cn_min_dB", 5.0), sw.get("cn_max_dB", 35.0), sw.get("n_cn", 7))
    plan, sim = cfg.plan, cfg.sim
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cases = (("input noise only", Impairments(), Components()),
             ("realistic, with EDFA", rimp, rcomps),
             ("realistic, without EDFA", replace(rimp, edfa=False), rcomps))
    rows = []
    for k, (lab, imp0, comps) in enumerate(cases):
        snr = []
        for cn in cns:
            imp = replace(imp0, input_cn_dB=float(cn))
            res, tx, fs = simulate(plan, sim, imp, comps, 0)
            f, sb, m = _rx(res, tx, fs, plan, sim, 0)
            snr.append(m["snr_evm_dB"])
            _row(rows, "fig10a", f"{lab} CN={cn:.0f}", f, m)
        # пол искажений: тот же набор источников без входного шума
        res, tx, fs = simulate(plan, sim, replace(imp0, input_cn_dB=None), comps, 0)
        floor = _rx(res, tx, fs, plan, sim, 0)[2]["snr_evm_dB"]
        ax[0].plot(cns, snr, "o-", ms=4, color=COLORS[k],
                   label=f"{lab} (C/N degradation at {cns[0]:.0f} dB: {cns[0] - snr[0]:.1f} dB)")
        ax[0].axhline(floor, color=COLORS[k], ls="--", lw=1)
        ax[0].text(cns[-1], floor, f" distortion floor {floor:.0f} dB", color=COLORS[k], fontsize=7,
                   ha="right", va="bottom")
    ax[0].plot(cns, cns, "k:", label="SNR = C/N")
    ax[0].set_xlabel("Input C/N in R_s band, dB")
    ax[0].set_ylabel("Output SNR (from EVM), dB")
    ax[0].set_title("(a) Output SNR vs input C/N")
    ax[0].legend(fontsize=7)
    cn0 = sw.get("fig10_budget_cn_dB", 20.0)
    bcfg = replace(cfg, imp=replace(rimp, input_cn_dB=float(cn0)), comps=rcomps)
    brows = noise_budget(bcfg, 0, cn0)
    ax[1].bar(range(len(brows)), [r["snr_dB"] for r in brows], color=COLORS[:1] * len(brows))
    ax[1].set_xticks(range(len(brows)))
    ax[1].set_xticklabels([r["name"] for r in brows], rotation=30, ha="right", fontsize=8)
    for k, r in enumerate(brows):
        ax[1].text(k, r["snr_dB"] + 0.5, f"{r['snr_dB']:.1f}", ha="center", fontsize=7)
    ax[1].set_ylabel("SNR if the source acts alone, dB")
    ax[1].set_title(f"(b) Noise budget at C/N = {cn0:g} dB (realistic)")
    fig.tight_layout()
    lines = [f"Noise budget, C/N = {cn0:g} dB:", f"  {'source':24}{'EVM,%':>9}{'SNR,dB':>9}"]
    lines += [f"  {r['name']:24}{r['evm_pct']:9.2f}{r['snr_dB']:9.1f}" for r in brows]
    return [("fig10_noise", fig)], rows, lines


def drift_ensemble(cfg, sg, nens):
    """Ансамбль дрейфа смещения при sigma = sg % V_pi: (EVM %, dBc линии |f1-fc|, аналитика по тем же z_e).

    Реализация e имеет b0 = sigma*z_e, z_e — первое нормальное число генератора seed
    (sim.seed + 900 + e). Аналитика: bias_error_ratio(beta1_eff, sigma*|z_e|) с поправкой на |H_pd|
    на линиях |f1-fc| и выходной (beta1_eff учитывает |H_mzm1(f1)|; WDM аналитикой не учитывается).
    """
    from .analytic import bias_error_ratio
    from .spectra import band_power as _band_power
    from .response import effective_betas

    plan, sim, comps = cfg.plan, cfg.sim, cfg.comps
    fi, fc = plan.f_lo_GHz[0], plan.fc_GHz
    bw = (1 + sim.alpha_eff) * sim.Rs
    fo = plan.line_info(0)[0]
    b1 = effective_betas(plan, comps, sim.beta1, sim.beta_m)[0][0]
    hpd = np.abs(comps.pd.H(np.array([abs(fi - fc) * 1e9, fo * 1e9])))
    ev, lv, an = [], [], []
    for e in range(1 if sg == 0 else nens):
        imp = Impairments(bias_drift=True, drift_sigma=sg / 100, drift_ens=e)
        res, tx, fs = simulate(plan, sim, imp, comps, 0)
        f, sb, m = _rx(res, tx, fs, plan, sim, 0)
        ev.append(m["evm_pct"])
        fr, p = rf_psd_abs(res.I, fs, 2 ** 14)
        lv.append(10 * np.log10(_band_power(fr, p, abs(fi - fc), bw) / _band_power(fr, p, fo, bw)))
        z = np.random.default_rng(sim.seed + 900 + e).normal()
        an.append(20 * np.log10(bias_error_ratio(b1, sg / 100 * abs(z)) * hpd[0] / hpd[1] + 1e-12))
    return ev, lv, an


def fig11(cfg, quick):
    from .analytic import bias_error_ratio
    from .response import effective_betas

    sw = cfg.sweeps
    sig = sw.get("fig11_sigma_pct_quick" if quick else "fig11_sigma_pct", [0.0, 2.0, 5.0])
    nens = sw.get("n_ens_quick" if quick else "n_ens", 8 if quick else 30)
    plan, sim = cfg.plan, cfg.sim
    fi, fc = plan.f_lo_GHz[0], plan.fc_GHz
    fo = plan.line_info(0)[0]
    evm, lvl, ana, rows = [], [], [], []
    for sg in sig:
        ev, lv, an = drift_ensemble(cfg, sg, nens)
        evm.append(ev)
        lvl.append(lv)
        ana.append(an)
        rows.append(("fig11", f"sigma={sg:g}% median over {len(ev)}", fo, float(np.median(ev)), 0.0, 0, 0.0))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    sg = np.array(sig, float)
    q = lambda a, p_: np.array([np.percentile(v, p_) for v in a])
    ax[0].plot(sg, q(evm, 50), "o-", label="median")
    ax[0].fill_between(sg, q(evm, 10), q(evm, 90), alpha=0.25, label="10-90 %")
    ax[0].set_ylim(0, 1.5 * max(max(v) for v in evm))
    ax[0].ticklabel_format(useOffset=False)
    ax[0].set_xlabel("Bias drift sigma, % of $V_\\pi$")
    ax[0].set_ylabel(f"EVM at {fo:g} GHz, %")
    ax[0].set_title(f"(a) EVM over {nens} draws: EVM is not affected,\ndrift lines fall outside the receive band",
                    fontsize=9)
    ax[0].legend(fontsize=7)
    nz = sg > 0
    ax[1].plot(sg[nz], q(lvl, 50)[nz], "o-", label="simulation: median")
    ax[1].fill_between(sg[nz], q(lvl, 10)[nz], q(lvl, 90)[nz], alpha=0.25, label="10-90 %")
    ax[1].plot(sg[nz], q(ana, 50)[nz], "s-", color="tab:red", ms=4, label="analytic, same draws (median)")
    b1 = effective_betas(plan, cfg.comps, sim.beta1, sim.beta_m)[0][0]
    dense = np.linspace(max(sg[nz].min(), 1e-3), sg.max(), 100)
    ax[1].plot(dense, 20 * np.log10(bias_error_ratio(b1, 0.6745 * dense / 100)), "k--",
               label="analytic, population median |dVb| = 0.6745 sigma")
    ax[1].set_xlabel("Bias drift sigma, % of $V_\\pi$")
    ax[1].set_ylabel(f"Line |f1 - fc| = {abs(fi - fc):g} GHz rel. useful {fo:g} GHz, dBc")
    ax[1].set_title("(b) Odd line from bias drift", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    return [("fig11_bias_drift", fig)], rows


def fig12(cfg, quick):
    from .interference import ci_runs, npr
    from .spectra import rf_psd_abs

    plan, sim, imp, comps = cfg.plan, cfg.sim, cfg.imp, cfg.comps
    sps = resolve_sps(plan, sim)
    mb = make_multicarrier_baseband(plan, sim, sps)
    res = run_all(plan, sim, imp, comps, baseband=mb, keep_fields=False)
    fs = res[0].fs
    bwsig = (1 + sim.alpha_eff) * sim.Rs
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    # (a) вход: PSD драйва с масками несущих
    ax = fig.add_subplot(gs[0, 0])
    nper = min(16 * sps, len(res[0].meta["u2_in"]))
    f, p = rf_psd_abs(res[0].meta["u2_in"], fs, nper)
    lo, hi = plan.band_GHz(bwsig)
    xr = (lo - 0.5, hi + 0.5)
    _spec(ax, f, 10 * np.log10(p / p.max() + 1e-300), xr, label="input u2")
    for k, c in enumerate(plan.carriers):
        ax.axvspan(c.f_GHz - bwsig / 2, c.f_GHz + bwsig / 2, color=COLORS[k % 10], alpha=0.2,
                   label=f"carrier {k + 1}: {c.f_GHz:g} GHz")
    ax.set_ylim(-100, 5)
    ax.set_xlabel("Frequency, GHz")
    ax.set_ylabel("PSD, dB rel. max")
    ax.set_title(f"(a) Input band: {plan.K} carriers, drive beta_m = {sim.beta_m:g}")
    ax.legend(fontsize=7)
    # (b) оптика ветки 1: боковые f_ck - f_1
    ax = fig.add_subplot(gs[0, 1])
    fo_, pg = res[0].fields["g"]
    ctr = plan.fc_of(0) - plan.f_lo_GHz[0] if plan.K == 1 else plan.f_mid() - plan.f_lo_GHz[0]
    _spec(ax, fo_, pg, (ctr - 1.5, ctr + 1.5), label="after DEMUX")
    for k, c in enumerate(plan.carriers):
        ax.axvline(c.f_GHz - plan.f_lo_GHz[0], color=COLORS[k % 10], ls=":", lw=1)
    ax.set_ylim(-100, 5)
    ax.set_xlabel("Optical offset, GHz")
    ax.set_ylabel("Power, dB rel. max")
    ax.set_title(f"(b) Optics of branch 1 near f_c - f_1 = {ctr:g} GHz (carrier sidebands)")
    # (c) фототок: до и после BPF, зум у выхода
    ax = fig.add_subplot(gs[1, 0])
    fo = plan.line_info(0)[0]
    df = (plan.carriers[1].f_GHz - plan.carriers[0].f_GHz) if plan.K > 1 else bwsig
    xr = (fo - 3 * abs(df), fo + 3 * abs(df))
    ref = None
    Ibs = []
    for i, r in enumerate(res):
        Ib = apply_bpf(r.I, fs, plan, sim, comps.bpf, i)
        Ibs.append(Ib)
        fr, p0 = rf_psd_abs(r.I, fs, nper)
        _, p1 = rf_psd_abs(Ib, fs, nper)
        ref = ref or p0.max()
        _spec(ax, fr, 10 * np.log10(p0 / ref + 1e-300), xr, color="lightgray")
        _spec(ax, fr, 10 * np.log10(p1 / ref + 1e-300), xr, color=COLORS[i % 10], label=f"branch {i + 1} after BPF")
    seen = {}
    for name, fq in plan.spurs(0).items():
        if xr[0] < fq < xr[1] and not name.endswith("@+"):
            seen.setdefault(round(fq, 4), []).append("IM3" if name.startswith("im3(") else name)
    for fq, names in seen.items():
        ax.axvline(fq, color="gray", ls=":", lw=0.7)
        ax.text(fq, 3, ", ".join(sorted(set(names))), fontsize=6, rotation=90, va="bottom")
    ax.set_ylim(-100, 10)
    ax.set_xlabel("Frequency, GHz")
    ax.set_ylabel("PSD, dB rel. max before BPF")
    ax.set_title(f"(c) Photocurrent at {fo:g} GHz: before (gray) / after BPF; marks: images and IM3 of branch 1",
                 fontsize=9)
    ax.legend(fontsize=6)
    # (d) созвездия веток
    sub = gs[1, 1].subgridspec(2, 2)
    rows = []
    lines = [f"{'br':>3}{'c':>3}{'f_i':>7}{'f_out':>7}{'EVM%':>7}{'EVM eq%':>8}{'BER':>9}{'C/I,dB':>8}{'C/ACI,dB':>10}{'C/IMD,dB':>10}{'C/LO,dB':>9}{'NPR,dB':>8}"]
    for i, r in enumerate(res[:4]):
        a = fig.add_subplot(sub[i // 2, i % 2])
        k = plan.routes[i].carrier
        f, sb = plan.line_info(i)
        m = measure(Ibs[i], fs, f, sb, sim, mb, carrier=k)
        if plan.K > 1:
            ci = ci_runs(cfg, i)
            npr_i = npr(cfg, i)
            cis = (ci["ci_dB"], ci["ci_aci_dB"], ci["ci_imd_dB"], ci["ci_lo_dB"], npr_i)
        else:
            cis = (float("nan"),) * 5
        _const(a, m, f"b{i + 1}: c{k + 1} {plan.carriers[k].f_GHz:g}→{f:g}, EVM {m['evm_pct']:.1f}%, C/I {cis[0]:.1f} dB")
        _row(rows, "fig12", f"branch {i + 1}", f, m)
        ee = m["evm_eq_pct"] if m["evm_eq_pct"] is not None else float("nan")
        lines.append(f"{i + 1:3d}{k + 1:3d}{plan.f_lo_GHz[i]:7.2f}{f:7.2f}{m['evm_pct']:7.2f}{ee:8.2f}{m['ber_mc']:9.1e}"
                     f"{cis[0]:8.1f}{cis[1]:10.1f}{cis[2]:10.1f}{cis[3]:9.1f}{cis[4]:8.1f}")
    fig.suptitle(f"Carrier routing: {plan.K} carriers -> {plan.N} branches, Rs = {sim.Rs:g} GBd, fs = {fs / 1e9:g} GHz",
                 fontsize=10)
    return [("fig12_routing", fig)], rows, lines


def _gd_ripple_ns(kind, order, f0, B, half, ripple=0.5, stop=40.0):
    """Неравномерность группового запаздывания (p-p, нс) в полосе f0 ± half по аналоговому прототипу."""
    from scipy import signal

    f = np.linspace(f0 - half, f0 + half, 201)
    om = (f ** 2 - f0 ** 2) / (f * B)
    if kind == "butter":
        b, a = signal.butter(order, 1.0, analog=True)
    elif kind == "cheby1":
        b, a = signal.cheby1(order, ripple, 1.0, analog=True)
    else:
        b, a = signal.ellip(order, ripple, stop, 1.0, analog=True)
    _, h = signal.freqs(b, a, worN=np.abs(om))
    ph = np.unwrap(np.sign(om) * np.angle(h))
    tau = -np.gradient(ph, 2 * np.pi * f * 1e9)
    return float(np.ptp(tau) * 1e9)


def _beta_grid(sw, quick):
    nb = sw.get("fig13_n_beta_quick", 5) if quick else sw.get("fig13_n_beta", 10)
    return np.unique(np.round(np.geomspace(sw.get("fig13_beta_min", 0.1), sw.get("fig13_beta_max", 1.1), nb), 3))


def fig13(cfg, quick):
    from .analytic import ci_multitone
    from .interference import drive_sweep

    sw = cfg.sweeps
    betas = _beta_grid(sw, quick)
    plan, sim = cfg.plan, cfg.sim
    if plan.K < 2:
        return _fig13_single(cfg, betas)
    i = 1 if plan.N > 1 else 0
    real = load_config(CONFIG_DIR / sw.get("fig13_compare_config", "router_realistic.toml"))
    if quick:
        real = quick_cfg(real)
    cmp_name = str(sw.get("fig13_compare_config", "router_realistic.toml")).removesuffix(".toml")
    base_name = f"{cfg.name} (as configured)"
    cases = [(base_name, cfg, True), (cmp_name, replace(real, plan=cfg.plan, sim=cfg.sim), False)]
    res = {name: drive_sweep(c, i, betas, with_npr=with_npr) for name, c, with_npr in cases}
    rows, lines = [], []
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    r0, b0 = res[base_name]
    bb = np.array([r["beta"] for r in r0])
    ax[0].plot(bb, [r["cni_dB"] for r in r0], "o-", label="C/I (model)")
    ax[0].plot(bb, [r["ci_aci_dB"] for r in r0], "s--", label="C/ACI")
    ax[0].plot(bb, [r["ci_imd_dB"] for r in r0], "^--", label="C/IMD")
    dense = np.geomspace(bb.min(), bb.max(), 100)
    fcs = [c.f_GHz for c in plan.carriers]
    a = carrier_weights(plan.carriers)          # уровни несущих (level_dB), как во входном композите
    an = [ci_multitone(sim.beta1, list(b * a), plan.routes[i].carrier, fcs, plan.f_lo_GHz[i],
                       plan.routes[i].sideband)["ci_dB"] for b in dense]
    ax[0].plot(dense, an, "k:", label="tone analytic (ci_multitone)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("$\\beta_m$")
    ax[0].set_ylabel("dB")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title(f"(a) Branch {i + 1}: C/I vs drive; EVM on right axis", fontsize=9)
    a2 = ax[0].twinx()
    a2.plot(bb, [r["evm_pct"] for r in r0], "r-o", ms=3, label="EVM, %")
    if r0[0]["evm_eq_pct"] is not None:
        a2.plot(bb, [r["evm_eq_pct"] for r in r0], "r--s", ms=3, label="EVM with equalizer, %")
    a2.set_yscale("log")
    a2.set_ylabel("EVM, %", color="r")
    a2.legend(fontsize=7, loc="lower left")
    _ibo_axis(ax[0])
    for k, (name, (rr, bs)) in enumerate(res.items()):
        ax[1].plot([r["beta"] for r in rr], [r["cni_dB"] for r in rr], "o-", color=COLORS[k], label=name)
        ax[1].axvline(bs, color=COLORS[k], ls=":", lw=1)
        ax[1].text(bs, ax[1].get_ylim()[0], f" $\\beta^*$={bs:g}", color=COLORS[k], fontsize=8, va="bottom")
        lines.append(f"  {name}: beta* = {bs:g} (max C/(N+I) on grid {', '.join(f'{b:g}' for b in betas)})")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("$\\beta_m$")
    ax[1].set_ylabel("C/(N+I), dB")
    ax[1].legend(fontsize=7)
    ax[1].set_title("(b) C/(N+I) vs drive and optimum $\\beta^*$", fontsize=9)
    ax[2].plot(bb, [r["npr_dB"] for r in r0], "o-")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("$\\beta_m$")
    ax[2].set_ylabel("NPR, dB")
    ax[2].set_title("(c) NPR (noise loading with notch on the carrier)", fontsize=9)
    fig.tight_layout()
    lines = [f"{'beta':>6}{'EVM,%':>8}{'EVM eq,%':>9}{'C/(N+I)':>9}{'C/ACI':>8}{'C/IMD':>8}{'NPR':>7}"] + [
        f"{r['beta']:6.3f}{r['evm_pct']:8.2f}{(r['evm_eq_pct'] if r['evm_eq_pct'] is not None else float('nan')):9.2f}"
        f"{r['cni_dB']:9.1f}{r['ci_aci_dB']:8.1f}{r['ci_imd_dB']:8.1f}{r['npr_dB']:7.1f}" for r in r0] + lines
    return [("fig13_drive_sweep", fig)], rows, lines


def _ibo_axis(ax):
    """Верхняя ось IBO (ориентир 20 lg((pi/2)/beta_m)) для оси beta_m в логарифмическом масштабе."""
    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    ticks = [t for t in (0.1, 0.2, 0.5, 1.0) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    top.set_xticks(ticks)
    top.set_xticklabels([f"{20 * np.log10((np.pi / 2) / t):.0f}" for t in ticks])
    top.minorticks_off()
    top.set_xlabel("IBO, dB (ref.: 20 lg((pi/2)/beta_m))", fontsize=8)


def _fig13_single(cfg, betas):
    """fig13 при K = 1: C/I, ACI/IMD и NPR не определены (нужны >= 2 несущих), показан EVM против drive."""
    sim = cfg.sim
    ev, eq, rows = [], [], []
    for b in betas:
        c = replace(cfg, sim=replace(sim, beta_m=float(b)))
        res, tx, fs = simulate(c.plan, c.sim, c.imp, c.comps, 0)
        f, sb, m = _rx(res, tx, fs, c.plan, c.sim, 0)
        ev.append(m["evm_pct"])
        eq.append(m["evm_eq_pct"])
        _row(rows, "fig13", f"beta_m={b:g}", f, m)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for a_ in (ax[0], ax[2]):
        a_.text(0.5, 0.5, "needs K ≥ 2 carriers\n(C/I, ACI/IMD, NPR are not defined for K = 1)", ha="center",
                va="center", transform=a_.transAxes)
        a_.set_xticks([])
        a_.set_yticks([])
    ax[0].set_title("(a) C/I, C/ACI, C/IMD vs drive: needs K ≥ 2", fontsize=9)
    ax[2].set_title("(c) NPR: needs K ≥ 2", fontsize=9)
    ax[1].plot(betas, ev, "ro-", label="EVM")
    if eq[0] is not None:
        ax[1].plot(betas, eq, "rs--", label="EVM with equalizer")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("$\\beta_m$")
    ax[1].set_ylabel("EVM, %")
    ax[1].legend(fontsize=7)
    ax[1].set_title("(b) K = 1: EVM vs drive", fontsize=9)
    fig.tight_layout()
    lines = ["fig13: K = 1 — C/I, ACI/IMD and NPR skipped (need K >= 2); EVM vs beta_m:"] + [
        f"  beta_m={b:6.3f}  EVM={e:6.2f} %" for b, e in zip(betas, ev)]
    return [("fig13_drive_sweep", fig)], rows, lines


def fig14(cfg, quick):
    from .response import bpf_att_dB, bpf_band

    plan, sim, comps = cfg.plan, cfg.sim, cfg.comps
    sw = cfg.sweeps
    f0, B = bpf_band(plan, sim.Rs, sim.alpha_eff, comps.bpf, 0)
    rs = sim.Rs
    spacing = (abs(plan.carriers[1].f_GHz - plan.carriers[0].f_GHz) if plan.K > 1
               else sw.get("fig14_spacing_Rs", 1.33) * rs)       # шаг несущих; при K = 1 — из [sweeps]
    areq = sw.get("fig14_A_req_dB", 30.0)
    rp0, st0 = comps.bpf.ripple_dB, comps.bpf.stop_dB
    variants = [(v["kind"], int(v["order"]), v.get("ripple_dB", rp0), v.get("stop_dB", st0))
                for v in sw.get("fig14_filters", [dict(kind="butter", order=4), dict(kind="butter", order=8),
                                                  dict(kind="cheby1", order=6), dict(kind="ellip", order=6),
                                                  dict(kind="ellip", order=8)])]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    x = np.linspace(sw.get("fig14_dfrs_min", 0.5), sw.get("fig14_dfrs_max", 3.0), 100)
    for k, (kind, n, rp, st) in enumerate(variants):
        att = [bpf_att_dB(f0 + d * rs, f0, B, n, kind, rp, st) for d in x]
        ax[0].plot(x, att, color=COLORS[k], label=f"{kind} {n}")
    ax[0].axhline(areq, color="k", ls=":", label=f"$A_{{req}}$ = {areq:g} dB")
    ax[0].axvline(spacing / rs, color="gray", ls="--", lw=1)
    ax[0].set_ylim(0, 80)
    ax[0].set_xlabel("$\\Delta f / R_s$")
    ax[0].set_ylabel("Neighbour suppression, dB")
    ax[0].set_title(f"(a) Suppression at f0 ± Δf (B = {B:g} GHz, f0 = {f0:g} GHz; dashed: carrier spacing)", fontsize=9)
    ax[0].legend(fontsize=7)
    fo = np.linspace(sw.get("fig14_fo_min_GHz", 5.0), sw.get("fig14_fo_max_GHz", 40.0), int(sw.get("fig14_n_fo", 36)))
    for k, (kind, n, rp, st) in enumerate(variants):
        q = []
        for f in fo:
            lo, hi = 0.005, 5.0
            for _ in range(40):                     # максимальная B, при которой подавление >= A_req
                mid = (lo + hi) / 2
                ok = bpf_att_dB(f + spacing, f, mid, n, kind, rp, st) >= areq
                lo, hi = (mid, hi) if ok else (lo, mid)
            q.append(f / lo)
        ax[1].plot(fo, q, color=COLORS[k], label=f"{kind} {n}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("$f_{out}$, GHz")
    ax[1].set_ylabel(f"Required Q = $f_{{out}}$/B ({areq:g} dB at the neighbour)")
    ax[1].set_title(f"(b) Required Q vs output frequency (spacing {spacing:g} GHz)", fontsize=9)
    ax[1].legend(fontsize=7)
    half = (1 + sim.alpha_eff) * rs / 2
    orders = sw.get("fig14_gd_orders", [6, 8])
    bws = B * np.array(sw.get("fig14_c_bw_factors", [1.0, 1.6, 2.6]))
    bg = np.linspace(B, 3 * B, 20)
    for k, n in enumerate(orders):
        ax[2].plot(bg, [_gd_ripple_ns("ellip", n, f0, b, half, rp0, st0) for b in bg], color=COLORS[k],
                   label=f"GD ripple, ellip {n}")
    ax[2].set_xlabel("BPF bandwidth B, GHz")
    ax[2].set_ylabel("Group-delay ripple in $(1+\\alpha)R_s$, ns")
    a2 = ax[2].twinx()
    rows, ev, eq = [], [], []
    c_order, c_beta = int(sw.get("fig14_c_order", 8)), sw.get("fig14_c_beta_m", 0.1)
    for b in bws:
        cc = replace(cfg, sim=replace(sim, equalizer="lms", beta_m=c_beta),   # малый beta_m: без нелинейности, чистая ISI
                     comps=replace(comps, bpf=replace(comps.bpf, kind="ellip", order=c_order, bw_GHz=float(b))))
        res, tx, fs = simulate(cc.plan, cc.sim, cc.imp, cc.comps, 0)
        Ib = apply_bpf(res.I, fs, cc.plan, cc.sim, cc.comps.bpf, 0)
        f, sb = cc.plan.line_info(0)
        m = measure(Ib, fs, f, sb, cc.sim, tx, carrier=cc.plan.routes[0].carrier)
        ev.append(m["evm_pct"])
        eq.append(m["evm_eq_pct"])
        _row(rows, "fig14c", f"ellip {c_order}, B={b:.2f} (no eq)", f, m)
        rows.append(("fig14c", f"ellip {c_order}, B={b:.2f} (equalizer)", f, m["evm_eq_pct"], m["ber_eq"], 0, 0.0))
    a2.plot(bws, ev, "ks", label="EVM without equalizer")
    a2.plot(bws, eq, "k^", mfc="none", label="EVM with equalizer")
    a2.set_ylabel("EVM, %")
    h1, l1 = ax[2].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, fontsize=7)
    ax[2].set_title(f"(c) Group-delay ripple (analog prototype) and EVM at beta_m = {c_beta:g}, branch 1", fontsize=9)
    fig.tight_layout()
    return [("fig14_bpf_selectivity", fig)], rows


FIGURES = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8, 9: fig9, 10: fig10, 11: fig11, 12: fig12, 13: fig13, 14: fig14}
