# Scoreboard -- Best Results per Method on MNISTVAE

## MNISTVAE (sigma_n=0.4)

| Method | HPD mean | KS stat | Iter | Notes |
|--------|----------|---------|------|-------|
| Grid Sampler | 0.498 | 0.015 | - | Gold standard |
| **Oracle Langevin** | **0.518** | **0.079** | 0 | ULA N=3000, lr=5e-7 |
| **SAL** | **0.533** | **0.081** | 1 | 10 levels×30 steps, smax=0.1, **diffusion-based** |
| MAP-Laplace | 0.472 | 0.109 | - | Gaussian approximation, slightly tight |
| FPS-SMC-MAP | 0.040 | 0.848 | 1 | MAP-init, smax=0.3, severely under-dispersed |
| Latent LATINO | 0.998 | 0.982 | 0 | Severely over-dispersed |

## Target: HPD mean in [0.45, 0.55], KS stat < 0.10
