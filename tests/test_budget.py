"""Бюджет шумов: сумма 1/SNR против всех источников сразу."""


from mwp_converter.budget import noise_budget
from mwp_converter.params import load_config, quick
import conftest


def test_sum_vs_all():
    cfg = quick(load_config(conftest.ROOT / "configs" / "realistic.toml"))
    rows = {r["name"]: r for r in noise_budget(cfg, 0, 20.0)}
    assert set(rows) >= {"distortion floor", "input C/N + NF", "EDFA ASE", "all sources", "sum of 1/SNR",
                         "LO phase noise (synth)", "LO phase noise (DVB-S2, ref.)", "drivers (OIP3, HD2)",
                         "bias drift (1 draw)"}
    assert rows["LO phase noise (DVB-S2, ref.)"]["snr_dB"] < rows["LO phase noise (synth)"]["snr_dB"]
    assert abs(rows["sum of 1/SNR"]["snr_dB"] - rows["all sources"]["snr_dB"]) < 1.5
    assert rows["input C/N + NF"]["snr_dB"] < rows["distortion floor"]["snr_dB"]
    assert all(r["snr_dB"] <= rows["distortion floor"]["snr_dB"] + 0.5 for r in rows.values()
               if r["name"] != "distortion floor")
