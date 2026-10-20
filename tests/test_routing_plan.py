"""Plan с несущими и маршрутами: from_output, совместимость со старым Plan, проверка плана."""
import json
import warnings

import pytest

from mwp_converter.params import Carrier, Plan, Route, SimParams, load_config, resolve_sps
import conftest

REF = json.load(open(conftest.ROOT / "tests" / "data" / "legacy_plan_reference.json"))
CS = [Carrier(f) for f in (29.6, 29.8, 30.0, 30.2)]


def example():
    return Plan(carriers=tuple(CS), routes=tuple(Route.from_output(CS, k, 19.8) for k in range(4)))


def test_from_output():
    lo = Route.from_output(CS, 0, 19.8, "lower", "low")
    hi = Route.from_output(CS, 0, 19.8, "lower", "high")
    up = Route.from_output([Carrier(14.0)], 0, 30.0, "upper")
    assert lo.f_lo_GHz == pytest.approx(4.9) and hi.f_lo_GHz == pytest.approx(24.7) and up.f_lo_GHz == 8.0
    assert Plan(carriers=tuple(CS), routes=(hi,)).line_info(0)[1] == "-"     # инверсия
    assert Plan(carriers=tuple(CS), routes=(lo,)).line_info(0)[1] == "+"
    for bad in (dict(f_out=35.0, sideband="lower", lo="low"), dict(f_out=-1.0, sideband="upper", lo="low"),
                dict(f_out=10.0, sideband="upper", lo="low")):
        with pytest.raises(ValueError):
            Route.from_output(CS, 0, **bad)


def test_example_plan():
    p = example()
    assert p.K == 4 and p.N == 4
    assert p.f_lo_GHz == pytest.approx((4.9, 5.0, 5.1, 5.2))
    assert p.out_freqs() == pytest.approx([19.8] * 4)
    assert p.f_mid() == pytest.approx(29.9) and p.band_GHz(0.2) == pytest.approx((29.5, 30.3))


@pytest.mark.parametrize("name", list(REF))
def test_legacy_equivalence(name):
    d = REF[name]
    p = Plan(*d["plan"])
    assert p.out_freqs() == d["out_freqs"] and p.offsets_GHz() == d["offsets"]
    for i in range(len(p.f_lo_GHz)):
        for w in (None, "lower", "upper", "fc", "3fc"):
            assert list(p.line_info(i, w)) == d["line_info"][str(i)][str(w)]
        assert p.spurs(i) == d["spurs"][str(i)]
    assert p.check(1.0, 0.35) == d["check"] and p.check(0.15, 0.35) == d["check_rect"]
    assert resolve_sps(p, SimParams()) == d["sps"]


def test_fc_warns_only_for_multicarrier():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Plan(30.0, (5.0,)).fc_GHz == 30.0
    with pytest.warns(UserWarning):
        example().fc_GHz
    assert example().fc_of(2) == 30.0


def test_spurs_and_check_example():
    p = example()
    names = p.spurs(0)
    assert names["img(c2)"] == pytest.approx(20.0) and "img(c1)" not in names
    assert names["im3(2c1-c2)"] == pytest.approx(29.4)            # как есть
    assert names["im3(2c1-c2)@-"] == pytest.approx(19.6)          # после переноса |2f_i - p|
    # пример router (alpha = 0.2, BPF ellip 8, B = 0.19): спектры соседей не перекрываются
    cfg = load_config(conftest.ROOT / "configs" / "router.toml")
    rep = cfg.plan.check(cfg.sim.Rs, cfg.sim.alpha_eff, cfg.comps.bpf)
    assert not any(r["level"] == "out" for r in rep)
    assert not any(r["line"] == "img" for r in rep)               # образы соседей на 0,2 ГГц вне занимаемой 0,18 ГГц
    for ch in (1, 2, 3, 4):
        assert sum(r["line"] == "im3" and r["level"] == "warn" and r["channel"] == ch for r in rep) == 1
    tones = {r["channel"]: r for r in rep if r["line"] == "4fi"}
    assert set(tones) == {1, 2}
    for r in tones.values():
        assert r["level"] == "warn" and r["att_dB"] >= 30 and "BPF suppression" in r["msg"]
    # прежний критерий без BPF (alpha = 0.35): тон 4f_i ближе (1+alpha)Rs — out у веток 1 и 2
    old = p.check(0.15, 0.35)
    assert {(r["channel"], r["line"]) for r in old if r["level"] == "out"} == {(1, "4fi"), (2, "4fi")}
    assert sum(r["line"] == "img" and r["level"] == "warn" for r in old) == 4   # alpha = 0.35: образы перекрываются


def test_wdm_grid_limit():
    many = Plan(30.0, tuple(5.0 + 0.1 * i for i in range(45)), "lower", 100.0)
    assert any(r["line"] == "wdm grid" and r["level"] == "warn" for r in many.check())
    assert not any(r["line"] == "wdm grid" for r in Plan(30.0, (5.0, 9.0), "lower", 100.0).check())


def test_replace_keeps_working_on_k1():
    from dataclasses import replace

    p = Plan(30.0, (5.5, 9.0))
    q = replace(p, f_lo_GHz=(6.0,))
    assert q.f_lo_GHz == (6.0,) and q.routes[0].f_lo_GHz == 6.0
    assert replace(p, direction="upper").out_freqs() == [2 * 5.5 + 30.0, 2 * 9.0 + 30.0]
    assert Plan(14.0, p.f_lo_GHz, "upper").out_freqs() == [25.0, 32.0]


def test_router_config():
    c = load_config(conftest.ROOT / "configs" / "router.toml")
    assert c.plan.K == 4 and c.plan.f_lo_GHz == pytest.approx((4.9, 5.0, 5.1, 5.2))


def test_no_fc_warning_in_replace_eq_repr():
    from dataclasses import replace

    p = example()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q = replace(p, wdm_spacing_GHz=50.0)
        assert q.wdm_spacing_GHz == 50.0 and q.routes == p.routes and q.carriers == p.carriers
        assert p == p and p == replace(p) and repr(p)
    mixed = Plan(carriers=(Carrier(30.0),), routes=(Route(5.0, 0, "lower"), Route(9.0, 0, "upper")))
    assert replace(mixed, wdm_spacing_GHz=60.0).routes == mixed.routes      # смешанные боковые не сплющиваются


def test_mixed_sidebands_useful_chain():
    from mwp_converter.chain import useful_sidebands
    from mwp_converter.params import Components
    from mwp_converter.response import band_report_for_output

    c = (Carrier(29.6), Carrier(30.0))
    p = Plan(carriers=c, routes=(Route.from_output(c, 0, 19.8), Route.from_output(c, 1, 48.0, "upper")))
    assert p.line_info(0)[0] == pytest.approx(19.8) and p.line_info(1)[0] == pytest.approx(48.0)
    assert useful_sidebands(p, 1) == pytest.approx([9.0, 39.0])
    assert useful_sidebands(p, 0) == pytest.approx([4.9, 24.7])
    rep = band_report_for_output(p, Components(), 0.15, 0.2, 2)
    pd = [r for r in rep if r["component"] == "pd"]
    assert len(pd) == 1 and pd[0]["f_GHz"] == pytest.approx(48.0)
    rep1 = band_report_for_output(p, Components(), 0.15, 0.2, 1)
    assert [r["f_GHz"] for r in rep1 if r["component"] == "pd"] == pytest.approx([19.8])
