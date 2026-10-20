"""Параметры модели: частотный план, сигнал, неидеальности, компоненты; загрузка конфигов TOML."""
import math
import tomllib
import warnings
from dataclasses import dataclass, field, replace

from .response import Bpf, OpticalFilter, Response, bpf_att_dB, bpf_band


@dataclass
class Carrier:
    """Несущая входной полосы: частота (ГГц), уровень (дБ), порядок PSK (None — sim.M), смещение seed."""

    f_GHz: float
    level_dB: float = 0.0
    M: int | None = None
    seed_offset: int = 0


@dataclass
class Route:
    """Маршрут (ветка): гетеродин f_i, номер несущей (с 0), боковая выходной линии lower|upper."""

    f_lo_GHz: float
    carrier: int = 0
    sideband: str = "lower"

    @staticmethod
    def from_output(carriers, k, f_out, sideband="lower", lo="low"):
        """Маршрут по желаемой выходной частоте f_out для несущей k.

        lower/low:  f_i = (f_ck - f_out)/2 (без инверсии, 2f_i < f_ck);
        lower/high: f_i = (f_ck + f_out)/2 (с инверсией);
        upper:      f_i = (f_out - f_ck)/2.
        Невозможная комбинация (f_i <= 0) — ValueError.
        """
        fck = carriers[k].f_GHz if hasattr(carriers[k], "f_GHz") else float(carriers[k])
        if sideband == "upper":
            fi = (f_out - fck) / 2
        elif sideband == "lower":
            if lo not in ("low", "high"):
                raise ValueError("lo ∈ {low, high}")
            fi = (fck - f_out) / 2 if lo == "low" else (fck + f_out) / 2
        else:
            raise ValueError("sideband ∈ {lower, upper}")
        if fi <= 0 or f_out <= 0:
            raise ValueError(f"невозможный маршрут: несущая {fck} ГГц -> {f_out} ГГц ({sideband}/{lo})")
        return Route(fi, k, sideband)


_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@dataclass(init=False)
class Plan:
    """Частотный план: несущие входной полосы и маршруты (ветки). Частоты в ГГц.

    Старый конструктор Plan(fc, f_lo, direction, wdm) сохраняется (K = 1, N маршрутов от одной
    несущей). Новый: Plan(carriers=(Carrier, ...), routes=(Route, ...), wdm_spacing_GHz=...).
    Поля dataclass — f_lo_GHz, direction, wdm_spacing_GHz, carriers, routes (replace/eq/repr их и
    используют); fc_GHz — свойство, при K > 1 даёт предупреждение: используйте fc_of(i).
    """

    f_lo_GHz: tuple
    direction: str
    wdm_spacing_GHz: float
    carriers: tuple
    routes: tuple

    def __init__(self, fc_GHz=None, f_lo_GHz=(), direction="lower", wdm_spacing_GHz=100.0, carriers=(), routes=()):
        f_lo = tuple(float(f) for f in f_lo_GHz)
        carriers, routes = tuple(carriers), tuple(routes)
        if direction not in ("lower", "upper"):
            raise ValueError("direction ∈ {lower, upper}")
        if carriers and routes:
            derived = tuple(r.f_lo_GHz for r in routes)
            sides = {r.sideband for r in routes}
            derived_dir = sides.pop() if len(sides) == 1 else routes[0].sideband
            legacy_edit = len(carriers) == 1   # dataclasses.replace на K = 1 перекрывает производные поля
            if legacy_edit and f_lo and f_lo != derived:
                routes = tuple(Route(f, 0, direction) for f in f_lo)
            elif legacy_edit and direction != derived_dir:
                routes = tuple(Route(r.f_lo_GHz, 0, direction) for r in routes)
            if legacy_edit and fc_GHz is not None and abs(fc_GHz - carriers[0].f_GHz) > 1e-12:
                carriers = (replace(carriers[0], f_GHz=float(fc_GHz)),)
        else:
            if fc_GHz is None or fc_GHz <= 0:
                raise ValueError("fc должна быть > 0")
            carriers = (Carrier(float(fc_GHz)),)
            routes = tuple(Route(f, 0, direction) for f in f_lo)
        if not routes:
            raise ValueError("Нужен хотя бы один маршрут")
        if any(c.f_GHz <= 0 for c in carriers):
            raise ValueError("fc должна быть > 0")
        if any(r.f_lo_GHz <= 0 for r in routes):
            raise ValueError("Все f_i должны быть > 0")
        if any(r.sideband not in ("lower", "upper") or not 0 <= r.carrier < len(carriers) for r in routes):
            raise ValueError("Некорректный маршрут")
        self.carriers, self.routes = carriers, routes
        self.f_lo_GHz = tuple(r.f_lo_GHz for r in routes)
        sides = {r.sideband for r in routes}
        self.direction = sides.pop() if len(sides) == 1 else routes[0].sideband
        self.wdm_spacing_GHz = wdm_spacing_GHz

    @property
    def fc_GHz(self):
        """Несущая плана (K = 1). При K > 1 неоднозначна — используйте fc_of(i)."""
        if len(self.carriers) > 1:
            warnings.warn("Plan.fc_GHz при K > 1 неоднозначна: используйте fc_of(i)", stacklevel=2)
        return self.carriers[0].f_GHz

    @property
    def K(self):
        """Число несущих."""
        return len(self.carriers)

    @property
    def N(self):
        """Число веток (маршрутов)."""
        return len(self.routes)

    def fc_of(self, i):
        """Несущая ветки i, ГГц."""
        return self.carriers[self.routes[i].carrier].f_GHz

    def f_mid(self):
        """Середина входной полосы (между крайними несущими), ГГц."""
        fs = [c.f_GHz for c in self.carriers]
        return (min(fs) + max(fs)) / 2

    def band_GHz(self, bw=0.0):
        """Входная полоса (от самой нижней до самой верхней несущей), расширенная на bw/2 с краёв."""
        fs = [c.f_GHz for c in self.carriers]
        return min(fs) - bw / 2, max(fs) + bw / 2

    def out_freqs(self):
        """Выходные частоты веток: |2f_i - fc| (lower) или 2f_i + fc (upper)."""
        return [self.line_info(i)[0] for i in range(self.N)]

    def offsets_GHz(self):
        """Оптические отстройки каналов: 0, +D, -D, +2D, ..."""
        d = self.wdm_spacing_GHz
        out = [0.0]
        k = 1
        while len(out) < self.N:
            out.append(k * d)
            if len(out) < self.N:
                out.append(-k * d)
            k += 1
        return out

    def line_info(self, i, which=None):
        """(частота ГГц, sideband '+'|'-') выходной линии ветки i.

        which: 'lower' | 'upper' | 'fc' | '3fc' | None (по боковой маршрута).
        Знак фазы: линия 2f-fc несёт -phi при 2f > fc (инверсия) и +phi при 2f < fc.
        """
        fi, fc = self.f_lo_GHz[i], self.fc_of(i)
        which = which or self.routes[i].sideband
        if which == "upper":
            return 2 * fi + fc, "+"
        if which == "lower":
            return abs(2 * fi - fc), ("-" if 2 * fi > fc else "+")
        if which == "fc":
            return fc, "+"
        if which == "3fc":
            return 3 * fc, "x3"
        raise ValueError(which)

    def _transfer(self, i, p):
        """Частота продукта p (входная полоса) на выходе ветки i: боковая маршрута."""
        fi = self.f_lo_GHz[i]
        return abs(2 * fi - p) if self.routes[i].sideband == "lower" else 2 * fi + p

    def spurs(self, i):
        """Паразитные линии ветки i (кроме выходной): имя -> частота, ГГц.

        2fi, 4fi, fc, 3fc и |2k*f_i ± q*f_c| (k=1,2; q=1,3). При K > 1 добавляются образы соседних
        несущих img(cj) = выход несущей j через ветку i, продукты IM3 2f_cj - f_cl и тройные
        f_cj + f_cl - f_cm: как есть (im3(...)) и после переноса (im3(...)@- и @+ = |2f_i ∓ p|).
        """
        fi, fc = self.f_lo_GHz[i], self.fc_of(i)
        out_name = "2fi-fc" if self.routes[i].sideband == "lower" else "2fi+fc"
        cand = {"2fi": 2 * fi, "4fi": 4 * fi, "fc": fc, "3fc": 3 * fc}
        for k in (1, 2):
            for q in (1, 3):
                for sgn, sg in ((1, "+"), (-1, "-")):
                    name = f"{2 * k}fi{sg}{q if q > 1 else ''}fc"
                    if name != out_name:
                        cand[name] = abs(2 * k * fi + sgn * q * fc)
        if self.K > 1:
            k0 = self.routes[i].carrier
            fcs = [c.f_GHz for c in self.carriers]
            for j in range(self.K):
                if j != k0:
                    cand[f"img(c{j + 1})"] = self._transfer(i, fcs[j])
            prods = {}
            for a in range(self.K):
                for b in range(self.K):
                    if a != b:
                        prods[f"im3(2c{a + 1}-c{b + 1})"] = 2 * fcs[a] - fcs[b]
            for a in range(self.K):
                for b in range(a + 1, self.K):
                    for c in range(self.K):
                        if c not in (a, b):
                            prods[f"im3(c{a + 1}+c{b + 1}-c{c + 1})"] = fcs[a] + fcs[b] - fcs[c]
            for name, p in prods.items():
                cand[name] = p
                cand[name + "@-"] = abs(2 * fi - p)
                cand[name + "@+"] = 2 * fi + p
        return cand

    def check(self, rs_GHz=1.0, alpha=0.35, bpf=None):
        """Проверка частотного плана (шаг 15 вывода).

        Паразитные линии ветки (`spurs`): если линия ближе (1+alpha)*Rs к выходу — запись
        уровня out. Образы соседних несущих и IM3 дают по одной сводной записи warn на ветку
        (line='img'/'im3', поле lines — имена). При N > floor(4400/D_WDM) — warn о сетке WDM.
        Запись: component='plan', channel, line, f_GHz, att_dB, level, msg.
        Если задан bpf (Bpf): немодулированные линии гетеродина (2fi, 4fi) — out, только если
        попадают в полосу пропускания BPF ветки, иначе warn с подавлением BPF на этой частоте
        (bpf_att_dB); окно отчёта для тона — (1+alpha)Rs + половина полосы BPF; модулированные —
        прежний критерий (1+alpha)Rs. Без bpf — прежнее поведение.
        """
        rep = []
        bw = (1 + alpha) * rs_GHz
        for i in range(self.N):
            fo, _ = self.line_info(i)
            lab = f"{i + 1}".translate(_SUB)
            grp = {"img": [], "im3": []}
            for name, f in self.spurs(i).items():
                tone = bpf is not None and name in ("2fi", "4fi")
                thr = bw + (bpf_band(self, rs_GHz, alpha, bpf, i)[1] / 2 if tone else 0.0)
                if abs(f - fo) >= thr:
                    continue
                if name.startswith("img("):
                    grp["img"].append((name, f))
                elif name.startswith("im3("):
                    grp["im3"].append((name, f))
                else:
                    nm = name.replace("fi", "f" + lab)
                    if tone:
                        f0, bwf = bpf_band(self, rs_GHz, alpha, bpf, i)
                        if abs(f - f0) > bwf / 2:
                            att = bpf_att_dB(f, f0, bwf, bpf.order, bpf.kind, bpf.ripple_dB, bpf.stop_dB)
                            rep.append(dict(component="plan", channel=i + 1, line=name, f_GHz=float(f),
                                            att_dB=att, level="warn",
                                            msg=f"channel {i + 1}: tone {nm} = {f:g} GHz is {abs(f - fo):g} GHz from "
                                                f"output {fo:g} GHz outside BPF {f0:g}±{bwf / 2:g} GHz; "
                                                f"BPF suppression {att:.0f} dB"))
                            continue
                    msg = (f"channel {i + 1}: {nm} = {f:g} GHz "
                           f"coincides with output {fo:g} GHz")
                    rep.append(dict(component="plan", channel=i + 1, line=name, f_GHz=float(f),
                                    att_dB=0.0, level="out", msg=msg))
            for key, title in (("img", "images of neighbour carriers"), ("im3", "IM3 products")):
                if grp[key]:
                    names = [n for n, _ in grp[key]]
                    msg = (f"channel {i + 1}: {title} within {bw:g} GHz of output {fo:g} GHz: "
                           + ", ".join(f"{n} = {f:g}" for n, f in grp[key]))
                    rep.append(dict(component="plan", channel=i + 1, line=key, f_GHz=float(grp[key][0][1]),
                                    att_dB=0.0, level="warn", msg=msg, lines=names))
        nmax = int(4400 // self.wdm_spacing_GHz)
        if self.N > nmax:
            rep.append(dict(component="plan", channel=0, line="wdm grid", f_GHz=float(self.wdm_spacing_GHz),
                            att_dB=0.0, level="warn",
                            msg=f"{self.N} branches exceed the WDM grid: at most {nmax} with {self.wdm_spacing_GHz:g} GHz spacing"))
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
    equalizer: str = "none"   # none | lms: эквалайзер приёмника (data-aided, T/2, МНК по известным символам)
    eq_taps: int = 11
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
    ofilt: OpticalFilter = field(default_factory=OpticalFilter)


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

    f_need = max(4 f_i + 3 fc, 3 f_i + 3 fc + (1+alpha_eff) Rs) по веткам; fc — наибольшая
    несущая ветки (при K > 1 — наибольшая из всех несущих). Если оценка памяти больше 4 ГБ —
    предупреждение.
    """
    if sim.SpS != "auto":
        return int(sim.SpS)
    fcmax = max(c.f_GHz for c in plan.carriers)
    fneed = max(max(4 * f + 3 * fcmax, 3 * f + 3 * fcmax + (1 + sim.alpha_eff) * sim.Rs)
                for f in plan.f_lo_GHz)
    sps = max(64, math.ceil(2.5 * fneed / sim.Rs))
    sps += sps % 2
    gb = plan.N * sim.n_sym * sps * 16 * 6 / 1e9
    if gb > 4:
        warnings.warn(f"Оценка памяти {gb:.1f} ГБ > 4 ГБ (SpS={sps}): уменьшите Rs, частоты или число символов")
    return sps


def mem_estimate_GB(plan, sim):
    """Оценка памяти, ГБ: N веток * n отсчётов * 16 байт * 6 массивов."""
    n = sim.n_sym * resolve_sps(plan, sim)
    return plan.N * n * 16 * 6 / 1e9


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
    bw = d.get("bw_GHz")
    bw = tuple(float(v) for v in bw) if isinstance(bw, (list, tuple)) else bw
    return Bpf(kind=d.get("kind", "butter"), order=int(d.get("order", 4)), bw_factor=float(d.get("bw_factor", 2.0)),
               bw_GHz=bw, ripple_dB=float(d.get("ripple_dB", 1.0)), stop_dB=float(d.get("stop_dB", 40.0)), center=c)


def _ofilt(d):
    pb = d.get("passbands", "auto")
    pb = tuple(float(v) for v in pb) if isinstance(pb, (list, tuple)) else pb
    return OpticalFilter(kind=d.get("kind", "none"), fwhm_GHz=float(d.get("fwhm_GHz", 1.0)),
                         order=int(d.get("order", 2)), passbands=pb)


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


def _routes_from(carriers, rows):
    """Маршруты из таблиц TOML/CLI: carrier (с 1), f_out_GHz (+ sideband, lo) либо f_lo_GHz."""
    routes = []
    for r in rows:
        k = int(r["carrier"]) - 1
        if "f_lo_GHz" in r:
            routes.append(Route(float(r["f_lo_GHz"]), k, r.get("sideband", "lower")))
        else:
            routes.append(Route.from_output(carriers, k, float(r["f_out_GHz"]), r.get("sideband", "lower"),
                                            r.get("lo", "low")))
    return tuple(routes)


def _carriers_from(rows):
    return tuple(Carrier(float(r["f_GHz"]) if isinstance(r, dict) else float(r),
                         float(r.get("level_dB", 0.0)) if isinstance(r, dict) else 0.0,
                         r.get("M") if isinstance(r, dict) else None,
                         int(r.get("seed_offset", k)) if isinstance(r, dict) else k)
                 for k, r in enumerate(rows))


def plan_from_dict(p, ov=None, wdm_default=100.0):
    """Plan из секции [plan] (или сценария): старый вид fc_GHz/f_lo_GHz либо carriers/routes.

    ov — overrides: fc, flo, direction, wdm_spacing (старый вид), carriers (частоты или таблицы),
    routes (список таблиц/кортежей (k, f_out[, sideband[, lo]])).
    """
    ov = ov or {}
    wdm = ov.get("wdm_spacing", p.get("wdm_spacing_GHz", wdm_default))
    legacy_ov = any(k in ov for k in ("fc", "flo", "direction"))
    if "carriers" in p and not legacy_ov or "carriers" in ov:
        rows = ov.get("carriers", p.get("carriers"))
        carriers = _carriers_from(rows)
        if "routes" in ov:
            rr = [dict(zip(("carrier", "f_out_GHz", "sideband", "lo"), t)) if not isinstance(t, dict) else t
                  for t in ov["routes"]]
        elif "carriers" in ov and "carriers" in p:
            base = plan_from_dict(p, None, wdm_default)
            rr = [dict(carrier=r.carrier + 1, f_out_GHz=base.line_info(j)[0], sideband=r.sideband,
                       lo="high" if (r.sideband == "lower" and 2 * r.f_lo_GHz > base.fc_of(j)) else "low")
                  for j, r in enumerate(base.routes)]
        else:
            rr = p["routes"]
        return Plan(carriers=carriers, routes=_routes_from(carriers, rr), wdm_spacing_GHz=wdm)
    if "routes" in ov:
        base = plan_from_dict(p, None, wdm_default)
        carriers = base.carriers
        rr = [dict(zip(("carrier", "f_out_GHz", "sideband", "lo"), t)) if not isinstance(t, dict) else t
              for t in ov["routes"]]
        return Plan(carriers=carriers, routes=_routes_from(carriers, rr), wdm_spacing_GHz=wdm)
    if "fc_GHz" not in p and "carriers" in p:   # старый override поверх router-конфига
        base = plan_from_dict(p, None, wdm_default)
        p = dict(p, fc_GHz=base.fc_of(0), f_lo_GHz=list(base.f_lo_GHz), direction=base.direction)
    return Plan(fc_GHz=ov.get("fc", p["fc_GHz"]), f_lo_GHz=tuple(ov.get("flo", p["f_lo_GHz"])),
                direction=ov.get("direction", p.get("direction", "lower")), wdm_spacing_GHz=wdm)


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
    plan = plan_from_dict(p, ov)
    sim = SimParams(Rs=ov.get("rs", s.get("Rs_GBd", 1.0)), alpha=s.get("alpha", 0.35),
                    n_sym=int(ov.get("n_sym", s.get("n_sym", 2 ** 14))), M=int(ov.get("M", s.get("M", 4))),
                    beta1=ov.get("beta1", m.get("beta1", 1.0)), beta_m=ov.get("beta_m", m.get("beta_m", 0.5)),
                    Vpi=m.get("Vpi", 2.0), P_dBm=la.get("P_dBm", 10.0), R=pd.get("R", 0.7),
                    RL=pd.get("RL", 50.0), Tc=pd.get("T_C", 25.0), seed=int(s.get("seed", 1)),
                    SpS=s.get("SpS", "auto"), pulse=s.get("pulse", "rrc"),
                    rx_block=int(raw.get("receiver", {}).get("block_symbols", 0)) or None,
                    equalizer=raw.get("receiver", {}).get("equalizer", "none"),
                    eq_taps=int(raw.get("receiver", {}).get("eq_taps", 11)))
    noise = la.get("noise", False)
    imp = Impairments(rin_dBHz=la.get("rin_dBHz", -150.0) if noise else None,
                      linewidth_Hz=la.get("linewidth_Hz", 0.0) if noise else 0.0,
                      er_dB=m.get("er_dB", 30.0) if m.get("er_enabled", False) else None,
                      pd_noise=bool(pd.get("noise", False)), rx_power_dBm=pd.get("rx_power_dBm"),
                      bias_err=float(m.get("bias_err", 0.0)), **_stage1(raw))
    c = raw.get("components", {})
    comps = Components(mzm1_rf=_resp(c.get("mzm1_rf", {})), mzm2_rf=_resp(c.get("mzm2_rf", {})),
                       pd=_resp(c.get("pd", {})), wdm=_resp(c.get("wdm", {})),
                       wdm_isolation_dB=c.get("wdm", {}).get("isolation_dB"), bpf=_bpf(c.get("bpf", {})), ofilt=_ofilt(c.get("ofilt", {})))
    return Config(name=name, plan=plan, sim=sim, imp=imp, comps=comps,
                  sweeps=raw.get("sweeps", {}), scenarios=raw.get("scenarios", {}))


def quick(cfg):
    """Копия конфига в режиме QUICK: n_sym = 2**11 (при K > 1 несущих — 2**10)."""
    return replace(cfg, sim=replace(cfg.sim, n_sym=2 ** 10 if cfg.plan.K > 1 else 2 ** 11))


def scenario_plans(cfg):
    """Планы из секции [scenarios.*] конфига: dict имя -> Plan (старый вид или carriers/routes)."""
    return {k: plan_from_dict(v, wdm_default=cfg.plan.wdm_spacing_GHz) for k, v in cfg.scenarios.items()}
