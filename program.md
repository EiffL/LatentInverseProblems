# Autoresearch: Calibrated Posterior Sampling

You are an autonomous research agent. Read this file at the start of every iteration.

## Goal

Achieve **calibrated diffusion posterior sampling** on MNISTVAE: `hpd_mean ∈ [0.45, 0.55]` and `hpd_ks < 0.10` using a method that operates through the score function `∇ log p_t(z)` at various noise levels. Oracle Langevin (direct MCMC on the exact posterior) does not count.

Output `<promise>BREAKTHROUGH</promise>` when achieved.

## The Problem

`MNISTVAE(sigma_n=0.4)`: pretrained MLP decoder `D: R² → [0,1]⁷⁸⁴`, Gaussian prior `z ~ N(0,I)`.
- Observation: `y = D(z*) + σ_n ε`
- Posterior std ~0.015 vs prior std 1.0 (60× concentration)
- **Exact analytic score**: `∇ log p_t(z) = -z/(σ₀² + σ_t²)` — no trained network
- Ground truth: adaptive fine grid posterior centered on encoder MAP

The problem provides: `decoder`, `decoder_jacobian`, `encoder`, `score`, `denoise`, `tweedie_cov`, `log_posterior`, `posterior_grid`, `hpd_level`.

## The Loop

1. **Read state**: `results.tsv`, `results/scoreboard.md`, `results/insights.md`, `papers/index.md`
2. **Decide**: pick next experiment based on accumulated evidence. Search literature when stuck.
3. **Implement & run**: prototype in `experiment.py`, run in <2 min. Promote working solvers to `lip/solvers/`.
4. **Log**: append to `results.tsv`, update `results/scoreboard.md` if new best, update `results/insights.md`.
5. **Commit or revert**: improvement → `git commit`. Regression → `git checkout -- experiment.py lip/solvers/`. Always log.

**Stuck-loop rules**: 3× same approach with no improvement → move on. 5 iterations with no progress → search literature.

## Files

| Modify freely | `experiment.py`, `results/`, `papers/` |
|---------------|----------------------------------------|
| Modify with care | `lip/solvers/`, `lip/solvers/__init__.py`, `scripts/run_mnist_vae.py` |
| Do not modify | `program.md`, `report.md`, `lip/problems.py`, `lip/metrics.py` |

Knowledge base: `report.md` (31-paper survey), `papers/` (structured summaries), `results/insights.md` (findings + hypotheses).

## Rules

1. One experiment per iteration. Atomic changes, clear attribution.
2. Record every experiment in `results.tsv`, even failures.
3. 2-minute time budget per experiment. Use `jax.lax.scan` and `jax.vmap`.
4. Git ratchet: commit improvements, revert failures.
5. Don't install new packages beyond requirements.txt.
6. Read `papers/index.md` every iteration — your memory resets, the knowledge base doesn't.
7. When in doubt, measure. Run the experiment.

## What We Know

**What works:**
- Oracle Langevin (hpd=0.518, KS=0.079): direct MCMC on exact `∇ log p(z|y)`. Validates grid, proves calibration achievable. Not diffusion-based.
- SAL (hpd=0.533, KS=0.081): score-annealed Langevin, 10 noise levels × 30 steps, σ_max=0.1. First diffusion-based method to achieve near-calibration. Multi-level scoring outperforms single-level. lr = lr_scale × σ_t² is the right scaling.
- MAP-Laplace (hpd=0.472, KS=0.109): Gaussian approximation, slightly under-dispersed but nearly calibrated.

**What fails:**
- All prior-initialized diffusion methods with σ_max ≥ 1.0: catastrophically over-dispersed (hpd > 0.9).
- LATINO, DPS, MMPS, LFlow, Split Gibbs: all severely over-dispersed on MNISTVAE.
- MAP-initialized FPS-SMC with small σ_max: severely under-dispersed (hpd=0.04).
- Toy problem success does not predict MNISTVAE success.

**Key insight:** The 60× concentration ratio (posterior std 0.015 vs prior std 1.0) breaks standard diffusion schedules. Methods must adapt noise/step size to the posterior scale, not the prior scale.

## Metrics

- **hpd_mean**: mean HPD credibility level. Target **0.500** (< 0.5 = under-dispersed, > 0.5 = over-dispersed).
- **hpd_ks**: KS statistic vs Uniform(0,1). Target close to 0.

```python
from lip import MNISTVAE
from lip.metrics import latent_calibration_test
problem = MNISTVAE(sigma_n=0.4)
result = latent_calibration_test(problem, my_solver, jax.random.PRNGKey(0), n=200)
```

## Completion

When target is achieved, also run `python scripts/run_mnist_vae.py` for full benchmark confirmation.
Otherwise iterate, then write `results/summary.md`.
