# Scoreboard -- Best Results per Method on MNISTVAE

## MNISTVAE (sigma_n=0.4)

| Method | HPD mean | KS stat | Iter | Notes |
|--------|----------|---------|------|-------|
| Grid Sampler | 0.498 | 0.015 | - | Gold standard |
| **MALA-SAL** | **0.508** | **0.036** | 5 | **Best diffusion-based. MALA + SAL annealing, 10L×30, lr=0.5** |
| **Oracle Langevin** | **0.518** | **0.079** | 0 | ULA N=3000, lr=5e-7 |
| **SAL** | **0.534** | **0.066** | 1 | 10 levels×30 steps, smax=0.1, ULA (n=500 avg) |
| MAP-Laplace | 0.472 | 0.109 | - | Gaussian approximation, slightly tight |
| FPS-SMC-MAP | 0.040 | 0.848 | 1 | MAP-init, smax=0.3, severely under-dispersed |
| Latent LATINO | 0.998 | 0.982 | 0 | Severely over-dispersed |

## Target: HPD mean in [0.45, 0.55], KS stat < 0.10
