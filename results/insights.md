# Research Insights

## Key Findings
1. Oracle Langevin (ULA, N=3000, lr=5e-7) achieves calibrated posteriors on MNISTVAE: hpd=0.518, KS=0.079
2. **Score-Annealed Langevin (SAL) achieves calibration with a diffusion-based approach**: hpd=0.534, KS=0.066. Uses the prior score at 10 noise levels (sigma from 0.1 to 0.01) combined with Tweedie-based likelihood guidance.
3. **MALA-SAL achieves near-perfect calibration**: hpd=0.508, KS=0.036. Adding Metropolis-Hastings correction to SAL eliminates ULA bias, yielding the best diffusion-based result. Consistently better than SAL across all random seeds.
4. MAP-initialized FPS-SMC with small sigma_max is severely under-dispersed (hpd=0.04). The tailored proposal concentrates too aggressively around the MAP.
5. Multi-level scoring genuinely helps vs single-level: ablation shows 10-level SAL (KS=0.062) outperforms 1-level (KS=0.109) at same total steps.
6. lr = lr_scale * sigma_t^2 is the right scaling for Langevin steps across noise levels — larger steps at higher noise, smaller at lower noise.
7. JIT + lax.scan gives 25x speedup (230s → 9s for n=100).
8. **ULA bias is the dominant source of over-dispersion in SAL**: at sigma_min=0.01, lr=5e-5 inflates the stationary distribution by ~11% per dimension. MALA correction eliminates this entirely.
9. Polishing phases (extra ULA steps at final noise level) make over-dispersion WORSE, not better — confirms ULA bias hypothesis.
10. Explicit likelihood annealing (tempering p(y|z) by α_t) and geometric β-tempering both fail — they create weaker gradients that cause more over-dispersion.

## Open Questions
- Can MALA-SAL be further improved? Current hpd=0.508 is very close to ideal 0.500
- Does the approach scale to higher-dimensional latent spaces? (MALA acceptance rate may degrade)
- Can we use the Tweedie covariance (MMPS-style) to further improve the likelihood approximation?
- Would preconditioning by the decoder Jacobian (Rao et al.) help in higher dimensions?

## Confirmed Hypotheses
- H_MALA: Metropolis correction eliminates ULA bias → near-perfect calibration (iter 5)
- H1: Multi-level annealing helps mixing (SAL vs single-level)

## Dead Ends
- Polishing phases (extra ULA steps at sigma_min): makes over-dispersion worse (iter 2)
- Explicit likelihood annealing (α_t tempering): no improvement over Tweedie SAL (iter 3)
- Geometric β-tempering (p(z|y)^β): severely over-dispersed, gradient too weak at low β (iter 4)
- MAP-initialized FPS-SMC with small sigma_max: tailored proposal over-concentrates (iter 1)
- All prior-initialized diffusion methods with sigma_max ≥ 1.0: catastrophically over-dispersed (archive)
