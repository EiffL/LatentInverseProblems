# Research Log

## 2026-03-13: Session 1 — From SAL to MALA-SAL

### Starting point
SAL (Score-Annealed Langevin) was the best diffusion-based method: hpd=0.533, KS=0.081. Uses noised prior score at 10 levels × 30 Langevin steps with Tweedie likelihood. Meets target [0.45, 0.55] / KS<0.10 but KS is uncomfortably close to boundary.

### Iteration 1: Reconfirm SAL
- SAL at n=200: hpd=0.528, KS=0.092. Confirmed within target but KS close to 0.10.

### Iteration 2: Polish phase hypothesis
- **Hypothesis**: ULA bias at the final noise level (lr=5e-5) inflates the stationary distribution. Adding a "polishing" phase with smaller lr on the exact posterior should fix this.
- **Result**: All polishing variants made over-dispersion WORSE (hpd=0.563, KS=0.131). More ULA steps = more bias accumulation regardless of lr.
- **Takeaway**: Simply running more ULA steps doesn't help. The bias is structural.

### Iteration 3: Tempered approaches (inspired by program.md intuition)
- **Literature search** found key papers on geometric tempering, ALD convergence theory, and divide-and-conquer posterior sampling.
- **Hypothesis A**: Explicitly anneal likelihood by β_t = α_t to match prior broadening.
  - Result: hpd=0.560, KS=0.127. No improvement.
- **Hypothesis B**: Pure geometric β-tempering on exact posterior.
  - Result: hpd=0.696, KS=0.342. Severe over-dispersion. At low β, gradient too weak relative to noise → chain explores too broadly. This failure is explained by provable_gt_ld2024 paper.
- **Takeaway**: The annealing schedule in SAL is already good. The problem is the MCMC kernel (ULA), not the annealing.

### Iteration 4: MALA-SAL (breakthrough)
- **Key insight**: ULA has a well-known step-size-dependent bias on the stationary distribution. For the posterior with variance σ²≈(0.015)² and step size ε≈5e-5, the ULA variance inflation is ~ε/(2σ²) ≈ 11% per dimension. MALA (Metropolis-Adjusted Langevin) eliminates this entirely via accept/reject.
- **Implementation**: Replace each ULA step with a MALA step (propose + MH correction). Same annealing schedule as SAL. Costs 2× gradient evaluations per step but produces exact samples.
- **Result at n=100**: hpd=0.524, KS=0.065 vs SAL's 0.541/0.092.
- **Tuning**: Default parameters (10L×30, lr=0.5) are already good.

### Iteration 5: High-confidence validation
- Ran MALA-SAL vs SAL with n=500 across 3 random seeds.
- **MALA-SAL**: hpd=0.508±0.011, KS=0.036±0.012
- **SAL (ULA)**: hpd=0.534±0.003, KS=0.066±0.008
- MALA-SAL is consistently and significantly better. Best single run: hpd=0.500, KS=0.025.

### Key insight
**The dominant source of miscalibration in SAL was ULA bias, not the annealing schedule or Tweedie approximation.** MALA fixes this cheaply (2× gradient cost, no other changes needed). The existing annealing framework (noised prior + Tweedie likelihood) is already excellent — it just needed the right MCMC kernel.

### Open directions
1. Preconditioning MALA with Tweedie covariance for higher-dimensional latent spaces
2. Using MALA within the LD-SMC framework for formal convergence guarantees
3. Testing on latent spaces with d > 2 where MALA acceptance rate may degrade

## 2026-03-13: Session 2 — Posterior diversity diagnostics

### Motivation
Visual inspection of plots showed MALA-SAL concentrating tightly while SAL scattered broadly. User concern: is MALA-SAL under-mixing? Are the SAL outliers exploring real secondary modes?

### Iteration 6: Log-scale contour analysis
- Replotted posterior contours on log₁₀ scale (10 orders of magnitude).
- **Finding**: the posterior is genuinely unimodal for most observations. SAL outlier points land in regions with essentially zero posterior density — they are NOT on secondary modes.
- Log-posterior at SAL outliers: 76 nats below MAP (= 10^{-33} probability ratio). Pure ULA noise.

### Iteration 7: Posterior variance ratio test
- **New metric**: Draw K=50 samples per observation, compare var(solver) / var(grid posterior).
- **MALA-SAL**: median ratio = 1.04 (near-perfect posterior spread coverage)
- **SAL**: median ratio = 1077 (1000× too broad — the "exploration" is 99.9% noise)
- This conclusively proves SAL's scattered points are ULA artifacts, not real posterior structure.

### Iteration 8: Multi-restart MAP initialization
- **Problem**: For ~5% of observations where z_true is at the prior tail, the encoder MAP is far from the posterior mode (up to 72 nats gap). MALA-SAL improves over the encoder but doesn't reach the distant mode.
- **Approach**: Run 5-10 gradient ascent chains from encoder + prior samples, pick the best MAP, then run MALA-SAL from there.
- **Result**: Marginal improvement (hpd=0.513 vs 0.516 average). Most observations already well-served by encoder. Not worth the extra compute for this problem.
- **Multi-restart with selection of best *final* sample**: FAILS (hpd=0.412) due to selection bias toward the mode.

### Key insight (session 2)
**SAL's apparent "diversity" is an illusion caused by ULA noise, not exploration of real posterior structure.** The posterior for MNISTVAE is overwhelmingly unimodal. The correct metric for posterior diversity is the variance ratio, not visual inspection of scatter plots.

### New diagnostic: Posterior variance ratio
```python
# Draw K samples per observation, compare variance to grid truth
solver_var = var(z_samples)
true_var = ∫(z - μ)² p(z|y) dz  (from grid)
ratio = solver_var / true_var  # target: 1.0
```
This captures mixing quality: stuck chains → ratio << 1, noisy chains → ratio >> 1.
