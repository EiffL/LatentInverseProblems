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
