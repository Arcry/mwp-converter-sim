"""CLI: рисунки для публикации. Пример: python make_figures.py --fig all --quick --out figures/"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from mwp_converter import plots
from mwp_converter.params import fs_GHz, load_config, quick as to_quick
from mwp_converter.response import band_report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--fc", type=float)
    ap.add_argument("--flo", type=lambda s: [float(v) for v in s.split(",")])
    ap.add_argument("--direction", choices=["lower", "upper"])
    ap.add_argument("--carriers", type=lambda s: [float(v) for v in s.split(",")],
                    help="несущие входной полосы, ГГц: 29.6,29.8,30.0")
    ap.add_argument("--routes", type=lambda s: [(int(r.split(":")[0]), float(r.split(":")[1]))
                                                + tuple(r.split(":")[2:]) for r in s.split(",")],
                    help="маршруты k:f_out[:sideband[:lo]] (k с 1), например 1:19.8,2:19.8")
    ap.add_argument("--rs", type=float)
    ap.add_argument("--beta1", type=float)
    ap.add_argument("--beta-m", type=float, dest="beta_m")
    ap.add_argument("--fig", default="all", help="all или номер 1..14")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="figures/")
    a = ap.parse_args()
    cfg = load_config(a.config, dict(fc=a.fc, flo=a.flo, direction=a.direction, rs=a.rs,
                                     beta1=a.beta1, beta_m=a.beta_m, carriers=a.carriers, routes=a.routes))
    if a.quick:
        cfg = to_quick(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"config={cfg.name}  quick={a.quick}  fc={', '.join(f'{c.f_GHz:g}' for c in cfg.plan.carriers)} GHz  f_lo={cfg.plan.f_lo_GHz}  "
          f"direction={cfg.plan.direction}  outputs={[round(f, 3) for f in cfg.plan.out_freqs()]}")
    print(f"fs = {fs_GHz(cfg.plan, cfg.sim):g} GHz (SpS auto), n_sym = {cfg.sim.n_sym}")
    rep = band_report(cfg.plan, cfg.comps, cfg.sim.Rs, cfg.sim.alpha_eff, cfg.imp)
    print("band_report (non-ok entries):")
    bad = [r for r in rep if r["level"] != "ok"]
    for r in bad:
        print(f"  [{r['level']:4}] {r['component']:7} ch{r['channel']} {r['line']:>22} "
              f"{r['f_GHz']:8.2f} GHz  {r['att_dB']:6.2f} dB" + (f"  {r['msg']}" if 'msg' in r else ""))
    print(f"  total {len(rep)} entries, non-ok {len(bad)}")
    todo = list(plots.FIGURES) if a.fig == "all" else [int(a.fig)]
    table = []
    extra_lines = []
    for n in todo:
        t0 = time.time()
        out_ = plots.FIGURES[n](cfg, a.quick)
        figs, rows = out_[0], out_[1]
        extra_lines += out_[2] if len(out_) > 2 else []
        for name, fig in figs:
            fig.savefig(out / f"{name}.png", dpi=200)
            fig.savefig(out / f"{name}.pdf")
            plots.plt.close(fig)
        table += rows
        print(f"fig{n}: {', '.join(nm for nm, _ in figs)}  {time.time() - t0:.1f} s")
    for ln in extra_lines:
        print(ln)
    if table:
        print("EVM / BER:")
        print(f"  {'fig':8}{'label':34}{'f,GHz':>8}{'EVM,%':>9}{'BER MC':>11}{'n_err':>7}{'BER(EVM)':>11}")
        for r in table:
            print(f"  {r[0]:8}{r[1]:34}{r[2]:8.2f}{r[3]:9.2f}{r[4]:11.2e}{r[5]:7d}{r[6]:11.2e}")


if __name__ == "__main__":
    main()
