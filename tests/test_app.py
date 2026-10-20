"""Страница Streamlit через AppTest: N веток, K несущих, точки «тип + ветка», кнопки."""
import pytest
from streamlit.testing.v1 import AppTest

import conftest

APP = str(conftest.ROOT / "app.py")
PTYPES = ["a", "d", "m(t)", "e", "g", "k", "BPF"]


def fresh(cfg="default"):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    if cfg != "realistic":
        at.selectbox(key="cfg_choice").select(cfg).run()
    assert not at.exception
    return at


@pytest.fixture(scope="module")
def default_app():
    return fresh("default")


@pytest.fixture(scope="module")
def router_app():
    return fresh("router")


def go(at, ptype, branch):
    at.radio(key="ptype").set_value(ptype).run()
    at.selectbox(key="branch").select(branch).run()
    assert not at.exception, (ptype, branch)


def test_realistic_renders():
    assert not AppTest.from_file(APP, default_timeout=300).run().exception


def test_default_all_points(default_app):          # K = 1, N = 3: как раньше по смыслу
    for p in PTYPES:
        for b in (1, 3):
            go(default_app, p, b)
    sbar = default_app.sidebar
    assert any("fs =" in c.value and "PAPR" in c.value for c in sbar.caption)         # сводка в боковой панели
    assert any(list(getattr(d.value, "columns", [])) == ["branch", "carrier", "f_i_GHz", "f_out_GHz", "inverted"]
               for d in sbar.dataframe)                                                  # вычисляемые f_i и инверсия
    tbl = default_app.dataframe[0].value
    assert list(tbl["f_i_GHz"]) == pytest.approx([5.5, 9.0, 17.0]) and list(tbl["inverted"]) == ["no", "no", "yes"]


@pytest.mark.slow
def test_router_all_points(router_app):            # K = 4, N = 4
    for p in PTYPES:
        for b in (1, 4):
            go(router_app, p, b)
    tbl = router_app.dataframe[0].value
    assert list(tbl["f_i_GHz"]) == pytest.approx([4.9, 5.0, 5.1, 5.2])
    assert any("★" in str(x) for x in router_app.dataframe[1].value.to_numpy().ravel())   # матрица маршрутизации


def test_edit_f_out_changes_f_i():
    at = fresh("default")
    at.session_state["routes_ed"] = {"edited_rows": {0: {"f_out_GHz": 17.0}}, "added_rows": [], "deleted_rows": []}
    at.run()
    assert not at.exception
    assert at.dataframe[0].value["f_i_GHz"][0] == pytest.approx(6.5)


def test_invalid_route_reports_error():
    at = fresh("default")
    at.session_state["routes_ed"] = {"edited_rows": {0: {"f_out_GHz": 99.0}}, "added_rows": [], "deleted_rows": []}
    at.run()
    assert at.error and "Invalid" in at.error[0].value


def test_f1_40ghz_error_realistic():
    at = fresh("realistic")
    at.session_state["routes_ed"] = {"edited_rows": {0: {"f_out_GHz": 10.0, "lo": "low"}}, "added_rows": [], "deleted_rows": []}
    at.session_state["carriers_ed"] = {"edited_rows": {0: {"f_GHz": 90.0}}, "added_rows": [], "deleted_rows": []}
    at.run()          # f_i = (90 - 10)/2 = 40 ГГц: МЦМ1 skin 40 ГГц -> out
    assert not at.exception
    assert any("MZM1" in e.value for e in at.error), [e.value for e in at.error]


def test_bpf_center_fixed_error(default_app):
    at = fresh("default")
    at.session_state["routes_ed"] = {"edited_rows": {0: {"bpf_center": "25"}}, "added_rows": [], "deleted_rows": []}
    at.run()
    assert not at.exception
    assert any("outside BPF" in e.value for e in at.error), [e.value for e in at.error]


def test_bpf_equalizer_pulse_and_optical_filter():
    at = fresh("default")
    at.selectbox(key="pulse").select("rect").run()
    at.selectbox(key="eq_mode").select("lms").run()
    at.selectbox(key="of_kind").select("lorentz").run()
    at.radio(key="ptype").set_value("BPF").run()
    assert not at.exception


def test_chain_toggle(default_app):
    default_app.checkbox(key="chain_resp").check().run()
    for p in ("a", "d", "m(t)", "g", "k", "BPF"):
        default_app.radio(key="ptype").set_value(p).run()
        assert not default_app.exception, p
    default_app.checkbox(key="chain_resp").uncheck().run()


@pytest.mark.parametrize("key", ["n_loss", "n_input", "n_laser", "n_er", "n_pd", "n_amp", "n_edfa", "fix_rx", "n_lo", "n_drv", "n_drift"])
def test_noise_checkboxes(key):
    at = fresh("default")
    at.checkbox(key=key).check().run()
    assert not at.exception, key
    at.radio(key="ptype").set_value("k").run()
    assert not at.exception, key


def test_bias_error_slider_and_rx_block():
    at = fresh("default")
    at.slider(key="bias_err").set_value(3.0).run()
    at.number_input(key="rx_block").set_value(0).run()
    assert not at.exception


def test_budget_button_default():
    at = fresh("default")
    at.button(key="budget_btn").click().run()
    assert not at.exception


@pytest.mark.slow
def test_router_buttons(router_app):
    for k in ("ci_btn", "npr_btn"):
        router_app.button(key=k).click().run()
        assert not router_app.exception, k
    tbl = router_app.dataframe
    assert any("C/I, dB" in getattr(d.value, "columns", []) for d in tbl)
    router_app.text_input(key="sweep_betas").set_value("0.2").run()
    router_app.button(key="sweep_btn").click().run()
    assert not router_app.exception
    router_app.button(key="budget_btn").click().run()
    assert not router_app.exception
