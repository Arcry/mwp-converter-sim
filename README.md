# mwp-converter-sim

Time-domain simulator of a multi-band microwave-photonic (MWP) frequency converter for satellite
transponders. It models the scheme from Scherbinin and Belov, "Multi-band Microwave Optics
Frequency Converter for Satellite Transmitter", SYNCHROINFO 2020
([doi:10.1109/SYNCHROINFO49631.2020.9166130](https://doi.org/10.1109/SYNCHROINFO49631.2020.9166130)),
using a corrected derivation of the paper's formulas. It extends the scheme to multi-carrier input,
carrier routing and realistic component impairments.

## Signal chain

```
LD_i → MZM1_i (LO f_i, null bias) → WDM MUX → shared MZM2 (input band, quadrature bias)
     → [EDFA] → DEMUX → [optical filter] → PD_i → BPF_i
```

Branch i converts the input band around f_c to the line |2f_i ± f_c|, and BPF_i selects the output.
With K input carriers, the shared MZM2 is driven by the whole band, so each of the N branches can
route any carrier to any output frequency (K → N routing).

## What is modelled

- **Signals:** PSK with RRC or rectangular pulses; one carrier or K carriers with their own levels
  and modulation orders. Frequencies come from TOML configs and are never hard-coded.
- **Components:** Mach-Zehnder modulators (via OptiCommPy) with finite extinction ratio and bias
  error; frequency responses of the MZM RF inputs, photodiodes and WDM channels (RC, skin effect,
  brick-wall, super-Gaussian or a measured S21 from CSV); output BPF (Butterworth, Chebyshev I or
  elliptic); an optional optical filter after the DEMUX.
- **Noise and distortion:** laser RIN and linewidth, insertion losses, input noise with a cascaded
  noise figure, EDFA ASE, photodiode shot and thermal noise, post-photodiode amplifier noise, LO
  phase noise, driver nonlinearity (OIP3, HD2), MZM bias drift (Ornstein-Uhlenbeck process),
  adjacent WDM-channel crosstalk.
- **Receiver and metrics:** block phase tracking and an optional fractionally spaced (T/2)
  equaliser trained on known symbols; EVM, SNR and BER; C/I split into adjacent-channel
  interference, intermodulation and LO spurs; noise-power ratio (NPR); drive sweeps with the optimum
  modulation index; a per-source noise budget; a spur check that flags parasitic lines near each
  output.

## Validation

- Simulated line amplitudes agree with the closed-form expressions (13′) and (16) of the corrected
  derivation (`mwp_converter/analytic.py`; `tests/test_lines_13prime.py`,
  `tests/test_quadrature_16.py`).
- Multi-carrier lines and IM3 products agree with a Jacobi-Anger multitone expansion
  (`analytic.lines_multitone`, `tests/test_lines_multitone.py`).
- 182 tests in total: 175 fast and 7 marked `slow`.

## Installation

Developed and tested with Python 3.11.

```bash
git clone https://github.com/Arcry/mwp-converter-sim.git
cd mwp-converter-sim
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Quick start

Command line (run from the repository root):

```bash
.venv/bin/python make_figures.py --fig all --quick --out figures/
.venv/bin/python make_figures.py --config configs/realistic.toml --fig 8 --out figures/
.venv/bin/python make_figures.py --config configs/router.toml --fig 12 --quick
```

Flags override the config: `--fc`, `--flo 5.5,9,17`, `--direction lower|upper`, `--rs`,
`--beta1`, `--beta-m`, `--carriers`, `--routes`. `--fig` takes `all` or a figure number from 1 to 14.

Python (from the repository root):

```python
from mwp_converter.params import load_config, quick
from mwp_converter.model import simulate
from mwp_converter.signals import measure

cfg = quick(load_config("configs/realistic.toml"))
res, tx, fs = simulate(cfg.plan, cfg.sim, cfg.imp, cfg.comps, i=0)  # branch 1
f_out, sideband = cfg.plan.line_info(0)
m = measure(res.I, fs, f_out, sideband, cfg.sim, tx, carrier=cfg.plan.routes[0].carrier)
print(f"output {f_out:.1f} GHz: EVM {m['evm_pct']:.1f} %, SNR {m['snr_evm_dB']:.1f} dB")
```

Interactive page: `.venv/bin/streamlit run app.py`. Pick a config, edit carriers and routes, and
inspect spectra, waveforms, constellations, component responses, interference and the noise budget
at any point of the chain.

Notebook: `notebooks/demo.ipynb`.

## Configurations

| File | Contents |
|---|---|
| `configs/default.toml` | Ideal model: one carrier at 30 GHz, three branches (LO 5.5 / 9 / 17 GHz → outputs 19 / 12 / 4 GHz), unlimited bandwidths, no noise |
| `configs/realistic.toml` | The same plan with component bandwidths and noise sources on: input C/N 20 dB, EDFA, bias drift, finite ER 30 dB |
| `configs/router.toml` | Carrier router, ideal: 4 carriers at 29.6–30.2 GHz, 0.15 GBd, each routed to 19.8 GHz |
| `configs/router_realistic.toml` | Carrier router with losses, noise, EDFA and driver nonlinearity |
| `configs/example_s21.csv` | Example measured S21 for a `csv` frequency response |

## Testing

```bash
.venv/bin/python -m pytest tests -q -m "not slow"   # 175 fast tests, about 2 minutes
.venv/bin/python -m pytest tests -q                 # all 182 tests
```

## Documentation

- [`docs/model_ru.md`](docs/model_ru.md): detailed model description in Russian: conventions (β,
  MZM bias, extinction ratio, RIN, spectrum inversion, sampling rate), component responses, each
  noise source, multi-carrier routing, interference metrics and the interactive page.
- [`docs/ml_roadmap.md`](docs/ml_roadmap.md): planned ML work: neural digital predistortion, a
  learned blind receiver, a differentiable twin, surrogate models and fault detection.

## Status

Research code by Andrei Shcherbinin. A packaged v1.0.0 release with a licence and citation
metadata is in preparation.
