# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Identify a reliable and correct strategy for **diffusion posterior sampling in latent space** that produces **calibrated posteriors** suitable for scientific applications (cosmology, medical imaging, geophysics). We benchmark methods on **MNISTVAE** -- a pretrained VAE decoder on MNIST where the exact posterior is available via grid evaluation.

**Central open problem:** No existing diffusion-based method provides calibrated posteriors with latent diffusion models. The decoder Jacobian distortion, representation error, and decoder nonlinearity remain unsolved.

**For research loop instructions, see `program.md`.** Current best results are in `results/scoreboard.md`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .                   # installs lip package (jax)
pip install -r requirements.txt    # additional deps: diffrax, numpyro, optax
```

Python 3.14 venv is already present at `.venv/`.

## Running Benchmarks

```bash
python scripts/run_mnist_vae.py    # MNISTVAE benchmark (all solvers)
python experiment.py               # Quick prototype / scratch pad
```

## Repository Structure

- `lip/` -- Minimal JAX library for posterior sampling benchmarks (installable via `pip install -e .`)
  - `problems.py` -- `MNISTVAE` dataclass: pretrained MLP VAE decoder on MNIST. `D: R^2 -> [0,1]^784`. Grid-based exact calibration.
  - `vae.py` -- Pure-JAX VAE forward pass (encoder/decoder) with weight loading from `.npz`.
  - `data/` -- Pretrained VAE weights (`vae_mnist_d2.npz`).
  - `solvers/` -- One file per solver, signature `(problem, y, key, **kwargs) -> z`.
  - `metrics.py` -- `latent_calibration_test` (HPD test), `latent_posterior_test`, `latent_benchmark`.
  - `__init__.py` -- Re-exports `MNISTVAE` and metrics.
- `scripts/` -- `run_mnist_vae.py` (benchmark), `train_vae.py` (VAE training)
- `experiment.py` -- Scratch pad for prototyping new solvers
- `program.md` -- Autonomous research loop instructions
- `report.md` -- Literature survey (31 papers)
- `results.tsv` -- Experiment log (append-only)
- `results/` -- Benchmark outputs, `scoreboard.md`, `insights.md`
- `papers/` -- Paper summaries (knowledge base for research loop)
- `archive/` -- Previous experiments (1D Gaussian, 2D toy problems, old solvers)

## Key Dependencies and Patterns

- **JAX** for autodiff and JIT; MNISTVAE uses `jax.jacfwd` for decoder Jacobians
- **optax** for VAE training
- **diffrax** for ODE/SDE integration (diffusion solvers)
- **diffrax 0.7+**: diffusion coefficients must return `lineax.DiagonalLinearOperator`, not raw arrays
