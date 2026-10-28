# ML roadmap

The simulator in this repository is a time-domain digital twin of a microwave-photonic frequency
converter for satellite transponders. It produces labelled waveforms, spectra and constellations at
every point of the block diagram, under configurable component bandwidths, noise and
nonlinearities. This makes it a natural data source and test bench for machine-learning methods.
This document lists the directions we consider and the order in which we plan to pursue them.

The multi-carrier routing model referred to below (K input carriers, N branches, C/I, ACI/IMD, NPR,
drive sweep) is introduced in PR #1.

## Directions

### 1. Neural digital predistortion (DPD) for multi-carrier operation

**Problem.** With several input carriers, the common input modulator (MZM2) generates IM3 products
that land exactly in the carrier slots. Filters cannot remove them, so C/(N+I) is limited by the
drive level. The simulator finds an optimum drive near β* ≈ 0.33 for the `router_realistic`
configuration.

**Idea.** Learn a predistorter for the composite input so that IM3 is suppressed at the output of
every branch. Candidate models:
- a small 1D-CNN or GRU;
- a polynomial-neural hybrid.

Baseline: memory-polynomial DPD.

**Metrics** (all already computed by the simulator): NPR, C/IMD, EVM, the shift of β* and the
resulting capacity gain.

### 2. Learned blind receiver

**Problem.** The current receiver is data-aided: timing, gain, phase and equalizer taps are
estimated from known symbols.

**Idea.** Train a neural receiver that equalizes and recovers carrier and timing blindly. Candidate
architectures: a CNN or a transformer equalizer with recovery heads.

Baselines:
- Costas carrier recovery;
- Gardner timing recovery;
- a decision-directed LMS equalizer.

Impairments to cover:
- LO phase noise;
- BPF group-delay ISI;
- bias drift;
- frequency offset.

**Key question.** Generalization to configurations that were not seen in training: other frequency
plans, filters and drive levels.

### 3. Differentiable twin and inverse design

**Idea.** Port the signal chain to JAX or PyTorch so that it becomes differentiable end to end.
Then optimize the design by gradient descent for a given routing table. Design variables:
- LO frequencies;
- BPF widths and orders;
- the drive level.

Constraints:
- no spurious coincidences;
- C/I above a target;
- a size, weight and power (SWaP) proxy as a cost term.

### 4. Surrogate model and Bayesian optimization of the frequency plan

**Idea.**
- Train a fast neural surrogate of the simulator, 10³–10⁴× faster.
- Use it for multi-objective Bayesian optimization of frequency plans.
- Use reinforcement learning for dynamic carrier routing under changing beam traffic.

### 5. Health monitoring and fault detection

**Idea.** Detect and localize degradations from photocurrent spectra:
- MZM bias drift;
- laser RIN growth;
- EDFA gain drop;
- filter detuning.

This is anomaly detection with labels that come for free from the simulator.

### 6. Open dataset and benchmark

**Idea.** Publish simulated spectra, constellations and point-of-diagram signals with impairment
labels on Zenodo or HuggingFace, together with baseline models for directions 1, 2 and 5.

## Plan

1. **Datasets and baselines.**
   - Generate datasets from the existing configurations (`default`, `realistic`, `router`,
     `router_realistic`) with randomized frequency plans, drive levels and impairments.
   - Implement the classical baselines (memory-polynomial DPD; Costas, Gardner and LMS receiver).
2. **Directions 1 and 2.** Neural DPD and learned receiver, evaluated with the simulator metrics
   (NPR, C/IMD, EVM, BER) against the baselines.
3. **Direction 6.** Release the dataset and benchmark with the code of steps 1–2.
4. **Directions 3 and 4.** Differentiable twin or surrogate-based design optimization.

Direction 5 can be added at any point, since it reuses the same data.

## Out of scope for now

- Hardware measurements; the simulator stays the only data source until a laboratory set-up is
  available.
- Downlink and ground-terminal modelling beyond an additive C/N term.
