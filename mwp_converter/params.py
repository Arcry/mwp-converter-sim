"""Параметры модели: частотный план, сигнал, неидеальности, компоненты; загрузка конфигов TOML."""
import math
import tomllib
import warnings
from dataclasses import dataclass, field, replace

from .response import Bpf, Response


@dataclass
class Plan:
    """Частотный план. Все частоты в ГГц, задаются пользователем (нигде не зашиты)."""

    fc_GHz: float
    f_lo_GHz: tuple
    direction: str = "lower"
    wdm_spacing_GHz: float = 100.0

    def __post_init__(self):
        self.f_lo_GHz = tuple(float(f) for f in self.f_lo_GHz)
        if self.fc_GHz <= 0:
            raise ValueError("fc должна быть > 0")
        if any(f <= 0 for f in self.f_lo_GHz) or not self.f_lo_GHz:
            raise ValueError("Все f_i должны быть > 0")
        if self.direction not in ("lower", "upper"):
            raise ValueError("direction ∈ {lower, upper}")

    def out_freqs(self):
        """Выходные частоты каналов: |2f_i - fc| (lower) или 2f_i + fc (upper)."""
        if self.direction == "lower":
            return [abs(2 * f - self.fc_GHz) for f in self.f_lo_GHz]
        return [2 * f + self.fc_GHz for f in self.f_lo_GHz]

    def offsets_GHz(self):
        """Оптические отстройки каналов: 0, +D, -D, +2D, ..."""
        d = self.wdm_spacing_GHz
        out = [0.0]
        k = 1
        while len(out) < len(self.f_lo_GHz):
            out.append(k * d)
            if len(out) < len(self.f_lo_GHz):
                out.append(-k * d)
            k += 1
        return out

    def line_info(self, i, which=None):
        """(частота ГГц, sideband '+'|'-') выходной линии канала i.

        which: 'lower' | 'upper' | 'fc' | '3fc' | None (по direction).
        Знак фазы: линия 2f-fc несёт -phi при 2f > fc (инверсия) и +phi при 2f < fc.
        """
        fi, fc = self.f_lo_GHz[i], self.fc_GHz
        which = which or self.direction
        if which == "upper":
            return 2 * fi + fc, "+"
        if which == "lower":
            return abs(2 * fi - fc), ("-" if 2 * fi > fc else "+")
        if which == "fc":
            return fc, "+"
        if which == "3fc":
            return 3 * fc, "x3"
        raise ValueError(which)

    def spurs(self, i):
        """Паразитные линии канала i (кроме выходной): имя -> частота, ГГц.

        2fi, 4fi, fc, 3fc и |2k*f_i ± q*f_c| (k=1,2; q=1,3).
        """
        fi, fc = self.f_lo_GHz[i], self.fc_GHz
        out_name = "2fi-fc" if self.direction == "lower" else "2fi+fc"
        cand = {"2fi": 2 * fi, "4fi": 4 * fi, "fc": fc, "3fc": 3 * fc}
        for k in (1, 2):
            for q in (1, 3):
                for sgn, sg in ((1, "+"), (-1, "-")):
                    name = f"{2 * k}fi{sg}{q if q > 1 else ''}fc"
                    if name != out_name:
                        cand[name] = abs(2 * k * fi + sgn * q * fc)
        return cand

    def check(self, rs_GHz=1.0, alpha=0.35):
        """Проверка частотного плана (шаг 15 вывода).

        Паразитные линии канала (`spurs`): если линия ближе (1+alpha)*Rs к выходу — запись
        уровня out. Запись: component='plan', channel, line, f_GHz, att_dB, level, msg.
        """
        rep = []
        bw = (1 + alpha) * rs_GHz
        for i in range(len(self.f_lo_GHz)):
            fo, _ = self.line_info(i)
            for name, f in self.spurs(i).items():
                if abs(f - fo) < bw:
                    lab = f"{i + 1}".translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
                    msg = (f"channel {i + 1}: {name.replace('fi', 'f' + lab)} = {f:g} GHz "
                           f"coincides with output {fo:g} GHz")
                    rep.append(dict(component="plan", channel=i + 1, line=name, f_GHz=float(f),
                                    att_dB=0.0, level="out", msg=msg))
        return rep


@dataclass
class SimParams:
    """Параметры сигнала, модуляторов, лазера и ФД. Rs — скорость в ГБод."""

    Rs: float = 1.0
    alpha: float = 0.35
    n_sym: int = 2 ** 14
    M: int = 4
    beta1: float = 1.0
    beta_m: float = 0.5
    Vpi: float = 2.0
    P_dBm: float = 10.0
    R: float = 0.7
    RL: float = 50.0
    Tc: float = 25.0
    seed: int = 1
    SpS: object = "auto"
    pulse: str = "rrc"  # rrc | rect
    rx_block: int | None = None  # блок (символов) поблочной подстройки фазы приёмника; None — выключена

    @property
    def alpha_eff(self):
        """Эффективный alpha для ширины полосы: (1+alpha_eff)*Rs; для rect полоса по нулям 2*Rs."""
        return self.alpha if self.pulse == "rrc" else 1.0


@dataclass
class Impairments:
    """Неидеальности (по умолчанию всё выключено)."""

    rin_dBHz: float | None = None
    linewidth_Hz: float = 0.0
    er_dB: float | None = None
    pd_noise: bool = False
    rx_power_dBm: float | None = None
    bias_err: float = 0.0  # уход смещения MZM1, доля Vpi
    # этап 1 (шумы): потери (MZM1, MUX, MZM2, DEMUX), входной шум, усилитель после ФД, EDFA
    loss_dB: tuple = (0.0, 0.0, 0.0, 0.0)
    input_cn_dB: float | None = None   # C/N на входе LNA в полосе Rs; None — без входного шума
    nf_lna_dB: float = 2.0
    g_lna_dB: float = 30.0
    nf_driver_dB: float = 8.0
    input_bw_factor: float = 2.0
    amp_nf_dB: float | None = None     # NF усилителя после ФД; None — выключен
    edfa: bool = False
    edfa_G_dB: float = 16.0
    edfa_NF_dB: float = 5.0
    edfa_lambda_nm: float = 1550.0
    edfa_ase: bool = True
    # этап 2: фазовый шум гетеродина, нелинейность драйверов, дрейф смещения (процесс О-У)
    lo_pn: bool = False
    lo_profile: str = "synth_typical"   # synth_typical | dvbs2_mask | table
    lo_table: tuple = ()
    drivers: bool = False
    oip3_dBm: float = 33.0
    hd2_lo_dBc: float = -30.0
    hd2_sig_dBc: float = -40.0
    bias_drift: bool = False
    drift_sigma: float = 0.02           # доля Vpi
    drift_tau_s: float = 1.0
    drift_ens: int = 0                  # номер реализации ансамбля


@dataclass
class Components:
    """Полосы компонентов."""

    mzm1_rf: Response = field(default_factory=Response)
    mzm2_rf: Response = field(default_factory=Response)
    pd: Response = field(default_factory=Response)
    wdm: Response = field(default_factory=Response)
    wdm_isolation_dB: float | None = None
    bpf: Bpf = field(default_factory=Bpf)


@dataclass
class Config:
    """Всё, что читается из TOML."""

    name: str
    plan: Plan
    sim: SimParams
    imp: Impairments
    comps: Components
    sweeps: dict
    scenarios: dict


def resolve_sps(plan, sim):
    """Наименьшее чётное SpS: SpS*Rs >= 2.5*f_need, не меньше 64.

    f_need = max(4 f_i + 3 fc, 3 f_i + 3 fc + (1+alpha) Rs) по всем каналам.
    """
    if sim.SpS != "auto":
        return int(sim.SpS)
    fneed = max(max(4 * f + 3 * plan.fc_GHz, 3 * f + 3 * plan.fc_GHz + (1 + sim.alpha_eff) * sim.Rs)
                for f in plan.f_lo_GHz)
    sps = max(64, math.ceil(2.5 * fneed / sim.Rs))
    sps += sps % 2
    if sps > 1024:
        warnings.warn(f"SpS={sps} > 1024: большой расход памяти, уменьшите Rs или частоты")
    return sps


def fs_GHz(plan, sim):
    """Частота дискретизации модели, ГГц."""
    return resolve_sps(plan, sim) * sim.Rs


def _resp(d):
    d = dict(d)
    return Response(kind=d.get("kind", "ideal"), f3dB_GHz=d.get("f3dB_GHz"),
                    order=int(d.get("order", 1)), csv_path=d.get("csv_path"),
                    csv_text=d.get("csv_text"))


def _bpf(d):
    c = d.get("center", "auto")
    c = tuple(c) if isinstance(c, (list, tuple)) else (c,)
    return Bpf(kind=d.get("kind", "butter"), order=int(d.get("order", 4)), bw_factor=float(d.get("bw_factor", 2.0)),
               bw_GHz=d.get("bw_GHz"), ripple_dB=float(d.get("ripple_dB", 1.0)), center=c)


def _stage1(raw):
    """Поля Impairments этапа 1 из секций [losses], [input], [edfa], [pd].amp_nf_dB."""
    ls, inp, ed, pd = raw.get("losses", {}), raw.get("input", {}), raw.get("edfa", {}), raw.get("pd", {})
    out = dict(loss_dB=(float(ls.get("mzm1_dB", 0)), float(ls.get("mux_dB", 0)), float(ls.get("mzm2_dB", 0)),
                        float(ls.get("demux_dB", 0))),
               input_cn_dB=float(inp.get("cn_dB", 20.0)) if inp.get("noise", False) else None,
               nf_lna_dB=float(inp.get("nf_lna_dB", 2.0)), g_lna_dB=float(inp.get("g_lna_dB", 30.0)),
               nf_driver_dB=float(inp.get("nf_driver_dB", 8.0)), input_bw_factor=float(inp.get("bw_factor", 2.0)),
               amp_nf_dB=float(pd["amp_nf_dB"]) if pd.get("amp_nf_dB", 0) > 0 else None,
               edfa=bool(ed.get("enabled", False)), edfa_G_dB=float(ed.get("G_dB", 16.0)),
               edfa_NF_dB=float(ed.get("NF_dB", 5.0)), edfa_lambda_nm=float(ed.get("lambda_nm", 1550.0)))
    lo, dr, bd = raw.get("lo", {}), raw.get("drivers", {}), raw.get("bias_drift", {})
    out.update(lo_pn=bool(lo.get("phase_noise", False)), lo_profile=lo.get("profile", "synth_typical"),
               lo_table=tuple(tuple(r) for r in lo.get("table", [])),
               drivers=bool(dr.get("nonlinear", False)), oip3_dBm=float(dr.get("oip3_dBm", 33.0)),
               hd2_lo_dBc=float(dr.get("hd2_lo_dBc", -30.0)), hd2_sig_dBc=float(dr.get("hd2_sig_dBc", -40.0)),
               bias_drift=bool(bd.get("enabled", False)), drift_sigma=float(bd.get("sigma_pct", 2.0)) / 100,
               drift_tau_s=float(bd.get("tau_s", 1.0)))
    return out


def load_config(path=None, overrides=None, text=None):
    """Читает TOML (путь или текст). overrides: fc, flo, direction, rs, beta1, beta_m, n_sym, M, wdm_spacing."""
    if text is not None:
        raw = tomllib.loads(text)
        name = "uploaded"
    else:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        name = str(path).rsplit("/", 1)[-1].removesuffix(".toml")
    ov = {k: v for k, v in (overrides or {}).items() if v is not None}
    p, s = raw["plan"], raw.get("signal", {})
    m, la, pd = raw.get("modulators", {}), raw.get("laser", {}), raw.get("pd", {})
    plan = Plan(fc_GHz=ov.get("fc", p["fc_GHz"]), f_lo_GHz=tuple(ov.get("flo", p["f_lo_GHz"])),
                direction=ov.get("direction", p.get("direction", "lower")),
                wdm_spacing_GHz=ov.get("wdm_spacing", p.get("wdm_spacing_GHz", 100.0)))
    sim = SimParams(Rs=ov.get("rs", s.get("Rs_GBd", 1.0)), alpha=s.get("alpha", 0.35),
                    n_sym=int(ov.get("n_sym", s.get("n_sym", 2 ** 14))), M=int(ov.get("M", s.get("M", 4))),
                    beta1=ov.get("beta1", m.get("beta1", 1.0)), beta_m=ov.get("beta_m", m.get("beta_m", 0.5)),
                    Vpi=m.get("Vpi", 2.0), P_dBm=la.get("P_dBm", 10.0), R=pd.get("R", 0.7),
                    RL=pd.get("RL", 50.0), Tc=pd.get("T_C", 25.0), seed=int(s.get("seed", 1)),
                    SpS=s.get("SpS", "auto"), pulse=s.get("pulse", "rrc"),
                    rx_block=int(raw.get("receiver", {}).get("block_symbols", 0)) or None)
    noise = la.get("noise", False)
    imp = Impairments(rin_dBHz=la.get("rin_dBHz", -150.0) if noise else None,
                      linewidth_Hz=la.get("linewidth_Hz", 0.0) if noise else 0.0,
                      er_dB=m.get("er_dB", 30.0) if m.get("er_enabled", False) else None,
                      pd_noise=bool(pd.get("noise", False)), rx_power_dBm=pd.get("rx_power_dBm"),
                      bias_err=float(m.get("bias_err", 0.0)), **_stage1(raw))
    c = raw.get("components", {})
    comps = Components(mzm1_rf=_resp(c.get("mzm1_rf", {})), mzm2_rf=_resp(c.get("mzm2_rf", {})),
                       pd=_resp(c.get("pd", {})), wdm=_resp(c.get("wdm", {})),
                       wdm_isolation_dB=c.get("wdm", {}).get("isolation_dB"), bpf=_bpf(c.get("bpf", {})))
    return Config(name=name, plan=plan, sim=sim, imp=imp, comps=comps,
                  sweeps=raw.get("sweeps", {}), scenarios=raw.get("scenarios", {}))


def quick(cfg):
    """Копия конфига в режиме QUICK: n_sym = 2**11."""
    return replace(cfg, sim=replace(cfg.sim, n_sym=2 ** 11))


def scenario_plans(cfg):
    """Планы из секции [scenarios.*] конфига: dict имя -> Plan."""
    return {k: Plan(fc_GHz=v["fc_GHz"], f_lo_GHz=tuple(v["f_lo_GHz"]),
                    direction=v.get("direction", "lower"),
                    wdm_spacing_GHz=v.get("wdm_spacing_GHz", cfg.plan.wdm_spacing_GHz))
            for k, v in cfg.scenarios.items()}
