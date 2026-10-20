"""Конфиги, overrides, проверка частотного плана, CLI."""
import subprocess
import sys

import pytest

from mwp_converter.params import Plan, load_config, scenario_plans
import conftest

CFG = conftest.ROOT / "configs"


def test_load_both():
    for n in ("default", "realistic"):
        c = load_config(CFG / f"{n}.toml")
        assert c.plan.f_lo_GHz == (5.5, 9.0, 17.0)
    assert load_config(CFG / "default.toml").comps.mzm1_rf.kind == "ideal"
    r = load_config(CFG / "realistic.toml")
    assert r.comps.wdm.kind == "supergauss" and r.imp.pd_noise and r.imp.er_dB == 30.0
    assert "up" in scenario_plans(r)


def test_overrides():
    c = load_config(CFG / "default.toml", dict(fc=14.0, flo=[8.0], direction="upper", rs=2.0,
                                               beta1=1.2, beta_m=0.3))
    assert (c.plan.fc_GHz, c.plan.f_lo_GHz, c.plan.direction) == (14.0, (8.0,), "upper")
    assert (c.sim.Rs, c.sim.beta1, c.sim.beta_m) == (2.0, 1.2, 0.3)
    assert c.plan.out_freqs() == [30.0]


def test_plan_check():
    bad = Plan(30.0, (5.0,), "lower").check()
    assert any(r["level"] == "out" and r["line"] == "4fi" for r in bad)
    assert Plan(30.0, (5.5,), "lower").check() == []


def test_cli_fig5(tmp_path):
    r = subprocess.run([sys.executable, str(conftest.ROOT / "make_figures.py"), "--fc", "14", "--flo", "8",
                        "--direction", "upper", "--fig", "5", "--quick", "--out", str(tmp_path)],
                       capture_output=True, text=True, cwd=conftest.ROOT)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "fig5_multiband.png").exists()


def test_stage1_fields():
    d, r = load_config(CFG / "default.toml"), load_config(CFG / "realistic.toml")
    assert d.imp.loss_dB == (0.0, 0.0, 0.0, 0.0) and d.imp.input_cn_dB is None
    assert not d.imp.edfa and d.imp.amp_nf_dB is None and not d.imp.pd_noise
    assert r.imp.loss_dB == (5.0, 3.0, 5.0, 3.0) and r.imp.input_cn_dB == 20.0
    assert r.imp.edfa and r.imp.edfa_G_dB == 16.0 and r.imp.amp_nf_dB == 3.0
    assert "cn_min_dB" in d.sweeps and "fig10_budget_cn_dB" in r.sweeps


@pytest.mark.slow
def test_cli_fig10(tmp_path):
    r = subprocess.run([sys.executable, str(conftest.ROOT / "make_figures.py"), "--fig", "10", "--quick",
                        "--out", str(tmp_path)], capture_output=True, text=True, cwd=conftest.ROOT)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "fig10_noise.png").exists()


def test_stage2_fields():
    d, r = load_config(CFG / "default.toml"), load_config(CFG / "realistic.toml")
    assert not d.imp.lo_pn and not d.imp.drivers and not d.imp.bias_drift
    assert r.imp.lo_pn and r.imp.lo_profile == "synth_typical" and r.imp.drivers and r.imp.bias_drift
    assert r.imp.oip3_dBm == 33.0 and r.imp.hd2_lo_dBc == -30.0 and r.imp.hd2_sig_dBc == -40.0
    assert r.imp.drift_sigma == pytest.approx(0.02) and r.imp.drift_tau_s == 1.0
    assert d.sim.rx_block == 256 and r.sim.rx_block == 256
    assert "fig11_sigma_pct" in d.sweeps and d.sweeps["n_cn_quick"] == 5


@pytest.mark.slow
def test_cli_fig11(tmp_path):
    r = subprocess.run([sys.executable, str(conftest.ROOT / "make_figures.py"), "--fig", "11", "--quick",
                        "--out", str(tmp_path)], capture_output=True, text=True, cwd=conftest.ROOT)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "fig11_bias_drift.png").exists()


def test_router_and_scenarios():
    from mwp_converter.params import quick, scenario_plans

    c = load_config(CFG / "router.toml")
    assert c.plan.K == 4 and c.plan.N == 4 and c.sim.Rs == 0.15
    assert c.comps.bpf.kind == "ellip" and c.comps.bpf.order == 8 and c.comps.bpf.bw_GHz == 0.19 and c.sim.alpha == 0.2
    assert set(scenario_plans(c)) == {"multicast", "subband", "inverted"}
    assert quick(c).sim.n_sym == 2 ** 10 and quick(load_config(CFG / "default.toml")).sim.n_sym == 2 ** 11
    assert load_config(CFG / "router_realistic.toml").imp.edfa
    assert load_config(CFG / "default.toml").comps.bpf.stop_dB == 40.0


def test_carrier_route_overrides():
    c = load_config(CFG / "router.toml", dict(carriers=[29.8, 30.0], routes=[(1, 19.8), (2, 19.8)]))
    assert c.plan.K == 2 and c.plan.f_lo_GHz == pytest.approx((5.0, 5.1))
    c = load_config(CFG / "router.toml", dict(routes=[(1, 12.0), (4, 12.0, "lower", "high")]))
    assert c.plan.out_freqs() == pytest.approx([12.0, 12.0]) and c.plan.line_info(1)[1] == "-"
    c = load_config(CFG / "router.toml", dict(fc=14.0, flo=[8.0], direction="upper"))
    assert c.plan.K == 1 and c.plan.out_freqs() == [30.0]


@pytest.mark.slow
def test_cli_fig12_two_carriers(tmp_path):
    r = subprocess.run([sys.executable, str(conftest.ROOT / "make_figures.py"), "--config", "configs/router.toml",
                        "--carriers", "29.8,30.0", "--routes", "1:19.8,2:19.8", "--fig", "12", "--quick",
                        "--out", str(tmp_path)], capture_output=True, text=True, cwd=conftest.ROOT)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "fig12_routing.png").exists() and all(h in r.stdout for h in ("C/I,dB", "C/ACI,dB", "C/IMD,dB", "C/LO,dB", "NPR,dB"))


def test_receiver_equalizer_fields():
    assert load_config(CFG / "router.toml").sim.equalizer == "lms" and load_config(CFG / "router.toml").sim.eq_taps == 11
    assert load_config(CFG / "default.toml").sim.equalizer == "none"


@pytest.mark.slow
def test_cli_fig13_and_all_single_carrier(tmp_path):
    """K = 1 (default.toml): fig13 и --fig all --quick завершаются с кодом 0 (C/I и NPR пропускаются)."""
    run = lambda *a: subprocess.run([sys.executable, str(conftest.ROOT / "make_figures.py"), "--config", "configs/default.toml",
                                     *a, "--quick", "--out", str(tmp_path)], capture_output=True, text=True, cwd=conftest.ROOT)
    r = run("--fig", "13")
    assert r.returncode == 0, r.stderr
    assert "need K >= 2" in r.stdout and (tmp_path / "fig13_drive_sweep.png").exists()
    r = run("--fig", "all")
    assert r.returncode == 0, r.stderr
    for name in ("fig12_routing", "fig13_drive_sweep", "fig14_bpf_selectivity"):
        assert (tmp_path / f"{name}.png").exists()
