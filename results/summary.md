# Results Summary: Calibrated Diffusion Posterior Sampling on MNISTVAE

## Achievement

**MALA-SAL** (Metropolis-Adjusted Score-Annealed Langevin) achieves calibrated diffusion posterior sampling on MNISTVAE(σ_n=0.4):

| Metric | Target | MALA-SAL | SAL (ULA) | Oracle Langevin |
|--------|--------|----------|-----------|-----------------|
| HPD mean | [0.45, 0.55] | **0.508** | 0.534 | 0.507 |
| KS stat | < 0.10 | **0.036** | 0.066 | 0.036 |

Results validated with n=500 calibration samples across 3 random seeds, and confirmed by the full benchmark (`scripts/run_mnist_vae.py` with n_cal=500).

## Method

MALA-SAL is a diffusion-based posterior sampler that uses:

1. **Multi-level noise annealing**: 10 noise levels σ_t from 0.1 to 0.01 (geometric spacing)
2. **Prior score at each level**: ∇ log p_t(z) = -z/(σ₀² + σ_t²)
3. **Tweedie-based likelihood**: D(α_t · z) where α_t = σ₀²/(σ₀² + σ_t²)
4. **MALA kernel**: Metropolis-Hastings correction at each Langevin step

The key innovation over SAL is replacing the ULA (Unadjusted Langevin Algorithm) kernel with MALA. This eliminates the step-size-dependent bias that inflated the stationary distribution by ~11% per dimension.

### Algorithm

```
Input: observation y, noise schedule [σ₁, ..., σ_K], lr_scale
z ← encoder(y)                          # MAP initialization
for k = 1 to K:                          # Anneal through noise levels
  σ_t ← σ_k,  lr ← lr_scale · σ_t²
  for j = 1 to N_steps:                  # MALA at this level
    g ← ∇_z log p_t(z|y)                # Prior score + Tweedie likelihood
    z' ← z + lr·g + √(2·lr)·ε           # Langevin proposal
    α ← min(1, p_t(z'|y)·q(z|z') / (p_t(z|y)·q(z'|z)))  # MH ratio
    z ← z' with probability α            # Accept/reject
return z
```

### Hyperparameters
- N_levels = 10, N_langevin = 30, σ_max = 0.1, σ_min = 0.01, lr_scale = 0.5
- Total: 300 MALA steps, ~600 gradient evaluations

## Key Finding

**The dominant source of miscalibration in score-annealed Langevin was ULA bias, not the annealing schedule or Tweedie approximation.** The annealing framework (noised prior + Tweedie likelihood at decreasing noise levels) is already well-designed. It just needed a bias-free MCMC kernel.

This is important for the field because:
- Most diffusion posterior sampling methods use ULA implicitly (via SDE discretization)
- ULA bias grows with step size relative to posterior variance
- For concentrated posteriors (like MNISTVAE with 60× prior/posterior ratio), this bias dominates

## Approaches That Failed

| Approach | Issue | Insight |
|----------|-------|---------|
| Extra ULA steps ("polishing") | More over-dispersion | ULA bias accumulates with more steps |
| Explicit likelihood annealing | No improvement | Tweedie already provides implicit annealing |
| Geometric β-tempering | Severely over-dispersed | Gradient too weak at low β |
| Prior-initialized σ_max ≥ 1.0 | Catastrophic over-dispersion | Can't transport from prior to 60× concentrated posterior |
| FPS-SMC (MAP init, small σ) | Severe under-dispersion | Tailored proposal over-concentrates |

## Reproducibility

```bash
# Quick validation (n=200, ~12s)
python experiment.py

# Full benchmark (n=500, ~60s)
python scripts/run_mnist_vae.py --solvers "MALA-SAL" "SAL" "Oracle Langevin" --n-cal 500
```

## Files
- Solver: `lip/solvers/mala_sal.py`
- Benchmark: `scripts/run_mnist_vae.py`
- Results: `results/9766178/`
- Full log: `results/log.md`
