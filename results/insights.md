# Research Insights

## Key Findings
1. Oracle Langevin (ULA, N=3000, lr=5e-7) achieves calibrated posteriors on MNISTVAE: hpd=0.518, KS=0.079
2. **Score-Annealed Langevin (SAL) achieves calibration with a diffusion-based approach**: hpd=0.533, KS=0.081. Uses the prior score at 10 noise levels (sigma from 0.1 to 0.01) combined with Tweedie-based likelihood guidance.
3. MAP-initialized FPS-SMC with small sigma_max is severely under-dispersed (hpd=0.04). The tailored proposal concentrates too aggressively around the MAP.
4. Multi-level scoring genuinely helps vs single-level: ablation shows 10-level SAL (KS=0.062) outperforms 1-level (KS=0.109) at same total steps.
5. lr = lr_scale * sigma_t^2 is the right scaling for Langevin steps across noise levels — larger steps at higher noise, smaller at lower noise.
6. JIT + lax.scan gives 25x speedup (230s → 9s for n=100).

## Open Questions
- Can SAL be improved further? Current hpd=0.533 is slightly over-dispersed vs target 0.500
- Would more noise levels help? Or more steps at the final (lowest) level?
- How sensitive is SAL to lr_scale? Current 0.5 was chosen without tuning
- Does this approach scale to higher-dimensional latent spaces?
- Is the slight over-dispersion due to the Tweedie approximation (alpha_t * z ≠ exact z_0)?

## New Hypotheses
- H11: SAL with learned/adaptive lr_scale at each noise level
- H12: SAL with non-uniform step allocation (more steps at lower sigma)
- H13: SAL with exact likelihood gradient at final level (no Tweedie) to reduce bias

## Dead Ends (from previous experiments)
- MAP-initialized FPS-SMC with small sigma_max: tailored proposal over-concentrates (iter 1)
- All prior-initialized diffusion methods with sigma_max ≥ 1.0: catastrophically over-dispersed (archive)
