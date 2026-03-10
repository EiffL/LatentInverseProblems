# Posterior Sampling for Inverse Problems with Diffusion/Flow Priors

This report surveys methods for solving inverse problems using diffusion and flow matching priors, with emphasis on posterior calibration and latent-space models.

**Test problem (this repo):** Prior $x \sim \mathcal{N}(\mu_0, \sigma_0^2)$, forward model $y = x + n$, $n \sim \mathcal{N}(0, \sigma_n^2)$.

---

## Part I: Methods Implemented in This Repository

### 1. LATINO — LAtent consisTency INverse sOlver

**Paper:** Spagnoletti, Prost, Almansa, Papadakis, Pereyra. *"LATINO-PRO"* ([arXiv:2503.12615](https://arxiv.org/abs/2503.12615), ICCV 2025).

Iterative noise–denoise–proximal loop. At each step $k$:

1. **Noise:** $x_{\text{noisy}} = x + \sigma_k \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$
2. **Denoise:** $u = \text{PF-ODE}(x_{\text{noisy}}, \sigma_k \to 0)$ using the prior score
3. **Proximal step:** $x = \frac{\delta_k y + \sigma_n^2 u}{\delta_k + \sigma_n^2}$ for $A = I$

**Gaussian calibration:** Under-dispersed. The proximal step is a contraction; deterministic PF-ODE provides no mechanism to restore lost variance.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

### 2. DPS — Diffusion Posterior Sampling

**Paper:** Chung, Kim, McCann, Klasky, Ye. *"Diffusion Posterior Sampling for General Noisy Inverse Problems"* ([arXiv:2209.14687](https://arxiv.org/abs/2209.14687), ICLR 2023).

Reverse-time SDE with likelihood gradient guidance:

$$\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y | x_t)$$

Likelihood approximated using only the Tweedie posterior **mean**:

$$p(y | x_t) \approx \mathcal{N}(y \mid \hat{x}_0(x_t),\; \sigma_n^2)$$

**Gaussian calibration:** Over-confident. Ignoring $V[x|x_t]$ makes guidance too strong at large $\sigma_t$, producing biased and under-dispersed posteriors.

**Critical limitation:** [Xu et al. (ICLR 2025)](https://arxiv.org/abs/2501.18913) showed that DPS actually behaves as **implicit MAP estimation**, not posterior sampling — it produces high-quality but low-diversity outputs.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

### 3. MMPS — Moment-Matching Posterior Sampling

**Paper:** Rozet, Andry, Lanusse, Louppe. *"Learning Diffusion Priors from Observations by Expectation Maximization"* ([arXiv:2405.13712](https://arxiv.org/abs/2405.13712), 2024).

Improves DPS by incorporating the Tweedie posterior covariance:

$$p(y | x_t) \approx \mathcal{N}\big(y \mid \hat{x}_0,\; \sigma_n^2 + V[x | x_t]\big)$$

**Gaussian calibration:** **Exact.** For Gaussian priors, the moment-matched likelihood is the true marginal likelihood, so MMPS with $\zeta=1$ recovers the exact posterior.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

### 4. LATINO + SDE

Variant replacing the deterministic PF-ODE denoiser with the stochastic reverse SDE, restoring variance lost by the proximal step.

**Gaussian calibration:** Nearly calibrated (z-std ≈ 0.98).

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

### 5. LFlow — Latent Refinement via Flow Matching

**Paper:** Askari, Luo, Sun, Roosta. *"Latent Refinement via Flow Matching for Training-free Linear Inverse Problem Solving"* ([arXiv:2511.06138](https://arxiv.org/abs/2511.06138), NeurIPS 2025).

Uses flow matching with OT interpolant $x_t = (1-t)x_0 + tz_1$ instead of diffusion SDE. Posterior velocity derived via the continuity equation:

$$v_t^y(x) = v_t(x) - \frac{t}{1-t}\,\nabla_{x_t} \log p(y|x_t)$$

with MMPS-style covariance in the likelihood. Pure ODE (deterministic given initial noise).

**Gaussian calibration:** Theoretically exact — the guided ODE recovers the exact posterior mean and variance (verified analytically with a high-precision ODE solver). Practically, Euler discretization introduces slow convergence (z-std=0.986 at N=200 vs MMPS's 1.002) because the ODE has stiff-like behavior from the $t/(1-t)$ factor and lacks the SDE's self-correcting noise.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

### Gaussian Calibration Summary

| Method | μ (target: 1.200) | σ (target: 0.447) | z-std (target: 1.000) |
|--------|-------------------|--------------------|-----------------------|
| Vanilla LATINO | 1.337 | 0.327 | 0.765 |
| DPS | 1.452 | 0.356 | 0.916 |
| **MMPS** | **1.179** | **0.446** | **1.002** |
| LATINO + SDE | 1.209 | 0.436 | 0.979 |
| **LFlow** | **1.199** | **0.442** | **0.986** |

---

## Part II: State of the Art for Calibrated Posteriors

### The fundamental hardness result

[Gupta et al. (ICML 2024)](https://arxiv.org/abs/2402.12727) proved that **the worst-case complexity of diffusion posterior sampling is super-polynomial**, even when unconditional sampling is fast. This means no algorithm can be simultaneously general, fast, and exact. All practical methods must trade off between these.

### Best current approaches for calibrated posteriors

#### Tier 1 — Asymptotically exact (SMC-based)

- **Twisted Diffusion Sampler (TDS)** [Wu et al., NeurIPS 2023]: Uses SMC with "twisting" — incorporates heuristic likelihood approximations as proposals while correcting via importance weights. Asymptotically exact as #particles → ∞, improvements seen with as few as 2 particles. Pixel-space only. [arXiv:2306.17775](https://arxiv.org/abs/2306.17775)
- **Filtering Posterior Sampling (FPS)** [Dou & Song, ICLR 2024]: Establishes equivalence between diffusion posterior sampling and Bayesian filtering, then applies SMC. Global convergence guarantees. Pixel-space, linear problems. [OpenReview](https://openreview.net/forum?id=tplXNcHZs1)
- **MCGdiff** [Cardoso et al., 2023]: Constructs Feynman-Kac model from the SGM prior, solved by SMC. Consistent under mild assumptions. Pixel-space, linear problems. [arXiv:2308.07983](https://arxiv.org/abs/2308.07983)
- **LD-SMC** [Achituve et al., ICML 2025]: The only **latent-space** method with formal convergence guarantees. Converges to the true posterior as #particles → ∞. Decoder appears only in likelihood evaluations, not gradient flows. In practice, tested with only N ∈ {1,5,10} particles. [arXiv:2502.05908](https://arxiv.org/abs/2502.05908)

#### Tier 1b — Split Gibbs / MCMC (non-asymptotic guarantees)

- **PnP-DM** [Wu et al., NeurIPS 2024]: Split Gibbs sampler alternating between exact likelihood conditional and denoising diffusion prior. Non-asymptotic convergence in Fisher information divergence at rate $O(1/K)$. **Best empirical calibration: 97.46% coverage within 3σ credible intervals** (vs 88.77% for DPS). Pixel-space only. [arXiv:2405.18782](https://arxiv.org/abs/2405.18782)
- **DPnP** [Xu & Chi, NeurIPS 2024]: First provably-robust posterior sampling for nonlinear inverse problems using unconditional diffusion priors. Alternates proximal consistency sampler and denoising diffusion sampler. Pixel-space. [arXiv:2403.17042](https://arxiv.org/abs/2403.17042)
- **G-DPS** [Giovannelli, 2026]: Augments the problem with the full diffusion chain as auxiliary variables. All conditionals are Gaussian. Convergence guaranteed, linear forward models only. [arXiv:2602.11059](https://arxiv.org/abs/2602.11059)

#### Tier 2 — Theoretically motivated, exact for Gaussians

- **MMPS** [Rozet et al., 2024]: Best calibration among single-trajectory methods. Exact for Gaussian, well-motivated for non-Gaussian.
- **LFlow** [Askari et al., 2025]: Same Tweedie covariance idea in the flow matching framework. Theoretically exact for Gaussian. ODE-based, more sensitive to discretization.
- **D-Flow SGLD** [2026]: Performs posterior inference in the source/noise space of flow matching models via preconditioned SGLD, then pushes forward through the learned flow. [arXiv:2602.21469](https://arxiv.org/abs/2602.21469)

#### Tier 3 — Practically robust, no formal guarantees

- **DAPS** [Zhang et al., CVPR 2025 Oral]: Best for hard nonlinear problems (phase retrieval). Decoupled annealing enables better mode exploration. Works in both pixel and **latent space**. [arXiv:2407.01521](https://arxiv.org/abs/2407.01521)
- **LATINO-PRO** [Spagnoletti et al., 2025]: Fastest (8 NFEs). Best for latent consistency models. Provably under-dispersed.

### Other notable methods (not yet implemented)

- **PSLD** [Rout et al., NeurIPS 2023]: First framework extending DPS to latent diffusion. Adds a "gluing objective." [arXiv:2307.00619](https://arxiv.org/abs/2307.00619)
- **STSL** [Rout et al., CVPR 2024]: Second-order Tweedie via surrogate loss. 4–8× fewer NFEs than PSLD. [arXiv:2312.00852](https://arxiv.org/abs/2312.00852)
- **ReSample** [Song et al., ICLR 2024]: Hard data consistency via optimization. [arXiv:2307.08123](https://arxiv.org/abs/2307.08123)
- **SILO** [Raphaeli et al., ICCV 2025]: Learned latent degradation operators. 2.6–10× faster. [arXiv:2501.11746](https://arxiv.org/abs/2501.11746)
- **MGPS** [Moufad et al., ICLR 2025 Oral]: Midpoint guidance with transition decomposition. [arXiv:2410.09945](https://arxiv.org/abs/2410.09945)
- **P2L** [Chung et al., ICML 2024]: Prompt-tuning for latent diffusion. [arXiv:2310.01110](https://arxiv.org/abs/2310.01110)
- **C-DPS** [Mohajer Hamidi & Yang, NeurIPS 2025]: Coupled measurement-space dynamics. [arXiv:2510.09676](https://arxiv.org/abs/2510.09676)
- **FlowDPS** [Kim et al., ICCV 2025]: DPS-style guidance for flow matching. [arXiv:2503.08136](https://arxiv.org/abs/2503.08136)

---

## Part III: Failure Modes None of the Methods Fully Address

### 1. The Tweedie approximation is fundamentally unimodal

DPS, MMPS, LFlow all approximate $p(x_0|x_t)$ as Gaussian. This is exact for Gaussians but breaks for multimodal posteriors. At large noise, $\mathbb{E}[x_0|x_t]$ is a blurry average over modes — the Gaussian approximation is wrong. MMPS adds the covariance but remains unimodal.

**Affected:** DPS, MMPS, TMPD, LFlow, and all first/second-order Tweedie methods.
**Mitigation:** MCMC correction (PnP-DM), multi-particle methods with repulsion.

### 2. Latent space introduces three compounding errors

This is the **key unsolved problem** for latent models:

- **Decoder Jacobian distortion:** The Jacobian $J_D(z)$ has decaying singular values, creating anisotropic latent dimensions where some directions matter far more than others for data-space fidelity. ([arXiv:2511.20592](https://arxiv.org/abs/2511.20592))
- **Representation error:** The encoder is many-to-one. Many latents decode to images consistent with measurements. [PSLD](https://arxiv.org/abs/2307.00619) showed vanilla DPS extensions to latent space simply don't work without a "gluing" penalty.
- **Nonlinearity of decode(encode(·)):** Even linear forward models $y = Ax + n$ become nonlinear in latent space: $y = A \cdot D(z) + n$, destroying closed-form proximal steps.

**Proposed solutions:** [ReSample](https://arxiv.org/abs/2307.08123) (hard data consistency via optimization), [SILO](https://arxiv.org/abs/2501.11746) (learned latent operators), Jacobian-aware weighting. None fully resolve the issue.

### 3. ODE methods systematically under-disperse

Deterministic ODE samplers concentrate samples due to score errors propagating coherently (no stochastic correction). SDE methods self-correct via noise injection.

**Affected:** LFlow, LATINO (PF-ODE), consistency models.
**Mitigation:** Use SDE samplers or hybrid approaches.

### 4. Calibration ≠ reconstruction quality

A [comprehensive UQ benchmark (Feb 2026)](https://arxiv.org/abs/2602.04189) found dramatic differences:

| Method | Reconstruction quality | Calibration |
|--------|----------------------|-------------|
| DPS, DiffPIR, DDNM | Good PSNR/SSIM | Substantially overconfident |
| REDDiff | Good PSNR/SSIM | Near-zero variance (point estimate) |
| PnP-DM, MCG-Diff | Good PSNR/SSIM | Reasonably calibrated |

Most papers report PSNR/SSIM/LPIPS but never validate calibration. The BIPSDA benchmark ([arXiv:2503.03007](https://arxiv.org/abs/2503.03007)) is beginning to address this.

### 5. No posterior guarantees with learned scores

- Unconditional diffusion sampling requires only L2 score accuracy.
- Posterior sampling requires much stronger conditions (MGF bounds, log-concavity).
- [Annealed Langevin (Wu et al., 2025)](https://arxiv.org/abs/2510.26324) achieves polynomial convergence with L4 score error bounds under local log-concavity — the best known result, but still restrictive.
- Performance guarantees can **diverge with increasing dimension** ([arXiv:2505.18276](https://arxiv.org/abs/2505.18276)).

### 6. Nonlinear forward models break guidance

All theory assumes linear $A$. Nonlinear models introduce expensive Jacobians, severe nonconvexity, and local minima. All gradient-guidance methods degrade. Proximal methods lose closed forms.

### 7. Manifold departure

Score functions are trained only on the noisy data manifold. Measurement-consistency projections can throw samples off-manifold where scores are unreliable. [MCG (Chung et al., NeurIPS 2022)](https://arxiv.org/abs/2206.00941) mitigates this with tangent-plane corrections.

---

## Part IV: Error Decomposition

Total error in diffusion posterior sampling decomposes into:

1. **Initialization/truncation error** — starting from finite rather than infinite noise
2. **Score approximation error** — learned score ≠ true score (**often dominant in practice**)
3. **Discretization error** — finite steps in ODE/SDE solver (mitigable with more steps)
4. **Likelihood approximation error** — Tweedie, guidance heuristics (**structural to the method class**)
5. **Latent-space error** — decoder nonlinearity, Jacobian distortion, representation gap (**fundamental to non-invertible architectures**)

For inverse problems, items 4 and 5 are the additional error sources that don't exist in unconditional sampling. This is the core reason why unconditional diffusion models work well but posterior sampling remains challenging.

---

## Part V: Summary — What to Use When

| Goal | Best approach | Trade-off |
|------|--------------|-----------|
| Fast reconstructions | LFlow / LATINO | Not calibrated; ~8 NFEs |
| Calibrated posteriors (pixel space) | **PnP-DM** (Split Gibbs) | ~100-1000× slower |
| Calibrated posteriors (latent space) | **Open problem** | Decoder Jacobian unsolved |
| Nonlinear forward models | DPnP / PnP-DM | Even more expensive |
| Multimodal posteriors | PnP-DM / DAPS | Must avoid Tweedie-based guidance |

**The uncomfortable truth:** No existing method provides calibrated posteriors with latent models efficiently. PnP-DM works but only in pixel space. Latent methods (LFlow, LATINO, PSLD) trade calibration for speed. The decoder Jacobian problem — bridging pixel-space measurements to latent-space priors without expensive or approximate Jacobian computation — remains the key open challenge.

---

## References

### Implemented in this repository

1. Chung, Kim, McCann, Klasky, Ye. "Diffusion Posterior Sampling for General Noisy Inverse Problems." ICLR 2023. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

2. Rozet, Andry, Lanusse, Louppe. "Learning Diffusion Priors from Observations by Expectation Maximization." 2024. [arXiv:2405.13712](https://arxiv.org/abs/2405.13712)

3. Spagnoletti, Prost, Almansa, Papadakis, Pereyra. "LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization." ICCV 2025. [arXiv:2503.12615](https://arxiv.org/abs/2503.12615)

4. Askari, Luo, Sun, Roosta. "Latent Refinement via Flow Matching for Training-free Linear Inverse Problem Solving." NeurIPS 2025. [arXiv:2511.06138](https://arxiv.org/abs/2511.06138)

### Asymptotically exact / SMC-based methods

5. Wu, Trippe, Naesseth, Blei, Cunningham. "Practical and Asymptotically Exact Conditional Sampling in Diffusion Models." NeurIPS 2023. [arXiv:2306.17775](https://arxiv.org/abs/2306.17775)

6. Cardoso, Janati El Idrissi, Le Corff, Moulines. "Monte Carlo Guided Diffusion for Bayesian Linear Inverse Problems." 2023. [arXiv:2308.07983](https://arxiv.org/abs/2308.07983)

7. Dou, Song. "Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective." ICLR 2024. [OpenReview](https://openreview.net/forum?id=tplXNcHZs1)

8. Achituve, Habi, Rosenfeld, Netzer, Diamant, Fetaya. "Inverse Problem Sampling in Latent Space Using Sequential Monte Carlo." ICML 2025. [arXiv:2502.05908](https://arxiv.org/abs/2502.05908)

9. Wu, Han, Naesseth, Cunningham. "Reverse Diffusion Sequential Monte Carlo Samplers." 2025. [arXiv:2508.05926](https://arxiv.org/abs/2508.05926)

### Split Gibbs / MCMC methods

10. Wu, Sun, Chen, Zhang, Yue, Bouman. "Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors." NeurIPS 2024. [arXiv:2405.18782](https://arxiv.org/abs/2405.18782)

11. Xu, Chi. "Provably Robust Score-Based Diffusion Posterior Sampling for Plug-and-Play Image Reconstruction." NeurIPS 2024. [arXiv:2403.17042](https://arxiv.org/abs/2403.17042)

12. Giovannelli. "A Gibbs Posterior Sampler for Inverse Problem Based on Prior Diffusion Model." 2026. [arXiv:2602.11059](https://arxiv.org/abs/2602.11059)

### Latent-space methods

13. Rout, Raoof, Daras, Caramanis, Dimakis, Shakkottai. "Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models." NeurIPS 2023. [arXiv:2307.00619](https://arxiv.org/abs/2307.00619)

14. Rout, Chen, Kumar, Caramanis, Shakkottai, Chu. "Beyond First-Order Tweedie: Solving Inverse Problems using Latent Diffusion." CVPR 2024. [arXiv:2312.00852](https://arxiv.org/abs/2312.00852)

15. Song, Kwon, Zhang, Hu, Qu, Shen. "Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency." ICLR 2024. [arXiv:2307.08123](https://arxiv.org/abs/2307.08123)

16. Chung, Ye, Milanfar, Delbracio. "Prompt-tuning Latent Diffusion Models for Inverse Problems." ICML 2024. [arXiv:2310.01110](https://arxiv.org/abs/2310.01110)

17. Raphaeli, Man, Elad. "SILO: Solving Inverse Problems with Latent Operators." ICCV 2025. [arXiv:2501.11746](https://arxiv.org/abs/2501.11746)

18. Rao, Qu, Moyer. "Latent Diffusion Inversion Requires Understanding the Latent Space." 2025. [arXiv:2511.20592](https://arxiv.org/abs/2511.20592)

### Guidance and decoupling methods

19. Zhang, Chu, Berner, Meng, Anandkumar, Song. "Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing." CVPR 2025 (Oral). [arXiv:2407.01521](https://arxiv.org/abs/2407.01521)

20. Moufad et al. "Variational Diffusion Posterior Sampling with Midpoint Guidance." ICLR 2025 (Oral). [arXiv:2410.09945](https://arxiv.org/abs/2410.09945)

21. Mohajer Hamidi, Yang. "C-DPS: Coupled Data and Measurement Space Dynamics for Enhanced Diffusion Posterior Sampling." NeurIPS 2025. [arXiv:2510.09676](https://arxiv.org/abs/2510.09676)

22. Li, Pereira. "Solving Inverse Problems via Diffusion Optimal Control." NeurIPS 2024. [arXiv:2412.16748](https://arxiv.org/abs/2412.16748)

### Flow matching methods

23. Kim, Kim, Ye. "FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems." ICCV 2025. [arXiv:2503.08136](https://arxiv.org/abs/2503.08136)

24. "D-Flow SGLD: Source-Space Posterior Sampling for Scientific Inverse Problems with Flow Matching." 2026. [arXiv:2602.21469](https://arxiv.org/abs/2602.21469)

### Analysis and benchmarks

25. Xu, Cai, Zhang, Ge, He, Sun, Liu, Zhang, Li, Wang. "Rethinking Diffusion Posterior Sampling: From Conditional Score Estimator to Maximizing a Posterior." ICLR 2025. [arXiv:2501.18913](https://arxiv.org/abs/2501.18913)

26. Gupta, Chen, Chen. "Diffusion Posterior Sampling is Computationally Intractable." ICML 2024. [arXiv:2402.12727](https://arxiv.org/abs/2402.12727)

27. Scope Crafts, Villa. "Benchmarking Diffusion Annealing-Based Bayesian Inverse Problem Solvers." 2025. [arXiv:2503.03007](https://arxiv.org/abs/2503.03007)

28. Qiu, Yang, Liu, Wang, Shen. "Benchmarking Uncertainty Quantification of Plug-and-Play Diffusion Priors for Inverse Problems Solving." 2026. [arXiv:2602.04189](https://arxiv.org/abs/2602.04189)

29. Chung, Kim, Ye. "Diffusion Models for Inverse Problems." Survey, 2025. [arXiv:2508.01975](https://arxiv.org/abs/2508.01975)

30. Chung, Sim, Ye. "Improving Diffusion Models for Inverse Problems using Manifold Constraints." NeurIPS 2022. [arXiv:2206.00941](https://arxiv.org/abs/2206.00941)

31. Wu et al. "Posterior Sampling by Combining Diffusion Models with Annealed Langevin." 2025. [arXiv:2510.26324](https://arxiv.org/abs/2510.26324)
