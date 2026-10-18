"""АЧХ по точкам схемы и сквозное ослабление."""
import pytest

from mwp_converter.chain import UPTO, chain_attenuation, response_view
from mwp_converter.params import load_config
import conftest

CFG = conftest.ROOT / "configs"


@pytest.mark.parametrize("point", list(UPTO))
def test_views(point):
    c = load_config(CFG / "realistic.toml")
    i = {"a": 0, "b": 1, "c": 2, "g": 0, "h": 1, "i": 2, "k": 0, "l": 1, "m": 2}.get(point, 0)
    if point.startswith("BPF"):
        i = "₁₂₃".index(point[-1])
    v = response_view(point, c.plan, c.sim, c.comps, 396e9, i)
    assert v["curves"] and v["table"]
    ch = chain_attenuation(point, c.plan, c.sim, c.comps, 396e9, i)
    assert ch[-1]["cum_dB"] == pytest.approx(sum(r["att_dB"] for r in ch))


def test_chain_ideal_is_zero():
    c = load_config(CFG / "default.toml")
    ch = chain_attenuation("BPF₁", c.plan, c.sim, c.comps, 396e9, 0)
    assert ch[-1]["cum_dB"] < 0.1
