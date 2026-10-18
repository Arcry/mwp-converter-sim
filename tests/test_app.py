"""Страница Streamlit через AppTest."""
import pytest
from streamlit.testing.v1 import AppTest

import conftest

APP = str(conftest.ROOT / "app.py")
POINTS = ["a", "b", "c", "d", "m(t)", "e", "g", "h", "i", "k", "l", "m", "BPF₁", "BPF₂", "BPF₃"]


def fresh(cfg="default"):
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert not at.exception
    if cfg != "realistic":
        at.selectbox(key="cfg_choice").select(cfg).run()
    return at


def test_runs_realistic_default():
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception


@pytest.mark.parametrize("point", POINTS)
def test_every_point(point):
    at = fresh("default")
    at.radio(key="point").set_value(point).run()
    assert not at.exception, point


def test_bpf1_ber_zero():
    at = fresh("default")
    at.radio(key="point").set_value("BPF₁").run()
    assert any("BER = 0" in m.value for m in at.markdown), [m.value for m in at.markdown]


def test_error_for_mzm1_at_40ghz():
    at = fresh("realistic")
    at.number_input(key="f1").set_value(40.0).run()
    assert not at.exception
    assert any("MZM1" in e.value for e in at.error), [e.value for e in at.error]


def test_bias_error_slider():
    at = fresh("default")
    at.slider(key="bias_err").set_value(3.0).run()
    assert not at.exception


def test_chain_toggle_every_point():
    at = fresh("default")
    at.checkbox(key="chain_resp").check().run()
    assert not at.exception
    for p in ("a", "d", "m(t)", "g", "k", "BPF₂"):
        at.radio(key="point").set_value(p).run()
        assert not at.exception, p


def test_bpf_fixed_center_error_and_pulse():
    at = fresh("default")
    at.number_input(key="f1").set_value(6.5).run()
    at.text_input(key="bpf_c1").set_value("19").run()
    assert not at.exception
    assert any("outside BPF" in e.value for e in at.error), [e.value for e in at.error]
    at.selectbox(key="pulse").select("rect").run()
    at.radio(key="point").set_value("BPF₁").run()
    assert not at.exception


@pytest.mark.parametrize("key", ["n_loss", "n_input", "n_laser", "n_er", "n_pd", "n_amp", "n_edfa", "fix_rx"])
def test_noise_checkboxes(key):
    at = fresh("default")
    at.checkbox(key=key).check().run()
    assert not at.exception, key
    at.radio(key="point").set_value("k").run()
    assert not at.exception, key


def test_m_t_shows_noisy_input_and_edfa_node():
    at = fresh("default")
    at.checkbox(key="n_input").check().run()
    at.checkbox(key="n_edfa").check().run()
    at.radio(key="point").set_value("m(t)").run()
    assert not at.exception
    assert any("EDFA ASE is not filtered" in m.value for m in at.markdown)  # warn при ideal WDM


def test_budget_button():
    at = fresh("realistic")
    at.button(key="budget_btn").click().run()
    assert not at.exception


@pytest.mark.parametrize("key", ["n_lo", "n_drv", "n_drift"])
def test_stage2_checkboxes(key):
    at = fresh("default")
    at.checkbox(key=key).check().run()
    assert not at.exception, key
    at.radio(key="point").set_value("BPF₁").run()
    assert not at.exception, key


def test_rx_block_and_preset():
    at = fresh("default")
    at.checkbox(key="n_lo").check().run()
    at.selectbox(key="lo_prof").select("dvbs2_mask").run()
    at.number_input(key="rx_block").set_value(0).run()
    assert not at.exception
