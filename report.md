# Methods for Solving Inverse Problems with Generative Priors

This report summarizes the diffusion/flow-based inverse problem solvers studied in this repository, tested on a 1D Gaussian calibration problem: prior $x \sim \mathcal{N}(\mu_0, \sigma_0^2)$, forward model $y = x + n$ with $n \sim \mathcal{N}(0, \sigma_n^2)$.

---

## Part I: Methods Implemented in This Repository

### 1. LATINO — LAtent consisTency INverse sOlver

**Paper:** Spagnoletti, Prost, Almansa, Papadakis, Pereyra. *"LATINO-PRO"* ([arXiv:2503.12615](https://arxiv.org/abs/2503.12615), ICCV 2025).

**Core idea:** Iterative noise-denoise-proximal loop. At each step $k$:

1. **Noise:** $x_{\text{noisy}} = x + \sigma_k \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$
2. **Denoise:** $u = \text{PF-ODE}(x_{\text{noisy}}, \sigma_k \to 0)$ using the prior score
3. **Proximal step:** $x = \text{prox}_{\delta_k \cdot g}(u) = \frac{\delta_k y + \sigma_n^2 u}{\delta_k + \sigma_n^2}$ for $A = I$

The sigma schedule is geometric: $\sigma_k \in [\sigma_{\max}, \sigma_{\min}]$. Delta schedule choices: vanishing ($\delta = \sigma_k^2$), constant ($\delta = \sigma_n^2$), or adaptive.

**Gaussian calibration result:** The PF-ODE denoiser is deterministic, so the only stochasticity comes from injected noise. The proximal step contracts the distribution, producing **under-dispersed** posteriors (variance too small). The analytic stationary distribution at fixed $\sigma$ does not match the true posterior for any delta schedule choice.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

### 2. DPS — Diffusion Posterior Sampling

**Paper:** Chung, Kim, Mccann, Klasky, Ye. *"Diffusion Posterior Sampling for General Noisy Inverse Problems"* ([arXiv:2209.14687](https://arxiv.org/abs/2209.14687), ICLR 2023).

**Core idea:** Run the reverse-time SDE from noise to data, adding a likelihood gradient at each step. Decomposes the posterior score via Bayes' rule:

$$\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y | x_t)$$

The likelihood is approximated using only the Tweedie posterior **mean**:

$$p(y | x_t) \approx \mathcal{N}(y \mid \hat{x}_0(x_t), \sigma_n^2)$$

where $\hat{x}_0 = x_t + \sigma_t^2 \nabla \log p_t(x_t)$ is the Tweedie denoised estimate.

**Gaussian calibration result:** This approximation ignores the posterior covariance $V[x|x_t]$, which causes the guidance to be **too strong** at large noise levels. The resulting posteriors have correct mean but slightly **under-dispersed** variance and shifted mean (biased toward the observation).

**Critical limitation:** [Kwon et al. (2025)](https://arxiv.org/abs/2501.18913) showed that DPS actually behaves as **implicit MAP estimation**, not posterior sampling — it produces high-quality but low-diversity outputs.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

### 3. MMPS — Moment-Matching Posterior Sampling

**Paper:** Rozet, Andry, Lanusse, Louppe. *"Learning Diffusion Priors from Observations by Expectation Maximization"* ([arXiv:2405.13712](https://arxiv.org/abs/2405.13712), 2024).

**Core idea:** Improves DPS by incorporating both Tweedie posterior moments (mean **and** covariance):

$$p(y | x_t) \approx \mathcal{N}\big(y \mid \hat{x}_0, \sigma_n^2 + V[x | x_t]\big)$$

where $V[x|x_t] = \sigma_t^2 \sigma_0^2 / (\sigma_0^2 + \sigma_t^2)$ is computed from Tweedie's second identity. The added covariance in the denominator tempers the guidance at large noise levels.

**Gaussian calibration result:** For the Gaussian case, this likelihood approximation is **exact** (not an approximation). With $\zeta = 1$, MMPS produces **perfectly calibrated** posteriors matching the analytic posterior in both mean and variance.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

### 4. LATINO + SDE

**Variant:** Replace the deterministic PF-ODE denoiser in LATINO with the stochastic reverse SDE. The SDE injects additional noise during denoising, which can restore variance lost in the proximal step.

**Gaussian calibration result:** Significantly improves over vanilla LATINO, producing **nearly calibrated** posteriors. The stochastic denoiser compensates for the variance contraction of the proximal step.

**Status:** Implemented in `GaussianLATINO.ipynb`.

---

### 5. LFlow — Latent Refinement via Flow Matching

**Paper:** Askari, Luo, Sun, Roosta. *"Latent Refinement via Flow Matching for Training-free Linear Inverse Problem Solving"* ([arXiv:2511.06138](https://arxiv.org/abs/2511.06138), NeurIPS 2025).

**Core idea:** Uses flow matching (optimal transport interpolant) instead of diffusion for the generative prior. The OT path is $x_t = (1-t) x_0 + t z_1$ with $z_1 \sim \mathcal{N}(0, I)$, evolving from data ($t=0$) to noise ($t=1$).

**Velocity field:**

$$v_t(x) = \mathbb{E}[z_1 - x_0 \mid x_t = x] = \frac{x}{t} - \frac{\hat{x}_0(x_t)}{t}$$

**Posterior guidance** (from continuity equation):

$$v_t^y(x) = v_t(x) - \frac{t}{1-t} \nabla_{x_t} \log p(y | x_t)$$

The likelihood uses the full Tweedie moments (like MMPS):

$$p(y | x_t) \approx \mathcal{N}\big(y \mid \hat{x}_0, \sigma_n^2 + V[x_0 | x_t]\big)$$

**Posterior covariance** for a Gaussian prior with OT interpolant:

$$V[x_0 | x_t] = \frac{\sigma_0^2 t^2}{(1-t)^2 \sigma_0^2 + t^2}$$

**Gaussian calibration result:** Theoretically exact — the guided ODE recovers the exact posterior mean and variance (verified analytically with a high-precision ODE solver). Practically, Euler discretization introduces slow convergence (z-std=0.986 at N=200 vs MMPS's 1.002) because the ODE has stiff-like behavior from the $t/(1-t)$ factor and lacks the SDE's self-correcting noise.

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

#### PnP-DM — Plug-and-Play Diffusion Models (NeurIPS 2024)

**Paper:** Wu, Sun, Chen, Zhang, Yue, Bouman. ([arXiv:2405.18782](https://arxiv.org/abs/2405.18782)) | [Project](https://imaging.cms.caltech.edu/pnpdm/)

The strongest current method for calibrated posteriors. Uses a **split Gibbs sampler** (MCMC) that alternates:
1. **Likelihood step:** Sample $z \sim p(z | x, y)$ — a Gaussian proximal step
2. **Prior step:** Sample $x \sim p(x | z)$ — a Bayesian denoising problem (exactly what diffusion models do)

**Key advantage:** No Tweedie approximation. No guidance heuristics. The diffusion model is used as a denoising oracle, not for likelihood estimation. Non-asymptotic stationarity guarantees. Captures 97.5% of ground truth pixels in 3-sigma credible intervals. Demonstrated on black hole imaging with multimodal posteriors.

**Limitation:** MCMC — requires many iterations. Operates in **pixel space only**.

#### DPnP — Diffusion Plug-and-Play (NeurIPS 2024)

**Paper:** Xu & Chi. ([arXiv:2403.17042](https://arxiv.org/abs/2403.17042))

First provably robust posterior sampling method for **nonlinear** inverse problems. Both asymptotic and non-asymptotic guarantees, with graceful degradation under score estimation error.

#### G-DPS — Gibbs Posterior Sampler (Feb 2025)

**Paper:** Giovannelli. ([arXiv:2602.11059](https://arxiv.org/abs/2602.11059))

Augments the problem with the full diffusion chain as auxiliary variables. All conditionals are Gaussian — "remarkably simple." Convergence guaranteed, but linear forward models only.

#### DAPS — Decoupled Annealing Posterior Sampling (CVPR 2025 Oral)

**Paper:** Zhang, Chu, Berner, Meng, Anandkumar, Song. ([arXiv:2407.01521](https://arxiv.org/abs/2407.01521))

Decouples consecutive diffusion steps, allowing large jumps in sample space. Time-marginals provably anneal to the true posterior. Works in both pixel and **latent space** (demonstrated with Stable Diffusion). Very expensive.

---

## Part III: Failure Modes None of the Methods Fully Address

### 1. The Tweedie approximation is fundamentally unimodal

DPS, MMPS, LFlow all approximate $p(x_0|x_t)$ as Gaussian. This is exact for Gaussians but breaks for multimodal posteriors. At large noise, $\mathbb{E}[x_0|x_t]$ is a blurry average over modes — the Gaussian approximation is wrong. MMPS adds the covariance but remains unimodal.

**Affected:** DPS, MMPS, TMPD, LFlow, and all first/second-order Tweedie methods.
**Mitigation:** MCMC correction (PnP-DM), multi-particle methods with repulsion.

### 2. Latent space introduces three compounding errors

This is the **key unsolved problem** for latent models:

- **Decoder Jacobian distortion:** The Jacobian $J_D(z)$ has decaying singular values, creating anisotropic latent dimensions where some directions matter far more than others for data-space fidelity. ([arXiv:2511.20592](https://arxiv.org/pdf/2511.20592))
- **Representation error:** The encoder is many-to-one. Many latents decode to images consistent with measurements. [PSLD](https://arxiv.org/abs/2307.00619) showed vanilla DPS extensions to latent space simply don't work without a "gluing" penalty.
- **Nonlinearity of decode(encode(·)):** Even linear forward models $y = Ax + n$ become nonlinear in latent space: $y = A \cdot D(z) + n$, destroying closed-form proximal steps.

**Proposed solutions:** [ReSample](https://arxiv.org/abs/2307.08123) (hard data consistency via optimization), [SILO](https://openaccess.thecvf.com/content/ICCV2025/papers/Raphaeli_SILO_Solving_Inverse_Problems_with_Latent_Operators_ICCV_2025_paper.pdf) (learned latent operators), Jacobian-aware weighting. None fully resolve the issue.

### 3. ODE methods systematically under-disperse

[Analysis of deterministic ODE samplers](https://arxiv.org/abs/2508.16154) shows they concentrate samples due to score errors propagating coherently (no stochastic correction). SDE methods self-correct via noise injection.

**Affected:** LFlow, LATINO (PF-ODE), consistency models.
**Mitigation:** Use SDE samplers or hybrid approaches.

### 4. Calibration ≠ reconstruction quality

A [comprehensive UQ benchmark (Feb 2026)](https://arxiv.org/abs/2602.04189) found dramatic differences:

| Method | Reconstruction quality | Calibration |
|--------|----------------------|-------------|
| DPS, DiffPIR, DDNM | Good PSNR/SSIM | Substantially overconfident |
| REDDiff | Good PSNR/SSIM | Near-zero variance (point estimate) |
| PnP-DM, MCG-Diff | Good PSNR/SSIM | Reasonably calibrated |

Most papers report PSNR/SSIM/LPIPS but never validate calibration.

### 5. No posterior guarantees with learned scores

- Unconditional diffusion sampling requires only L2 score accuracy.
- Posterior sampling requires much stronger conditions (MGF bounds, log-concavity).
- [Annealed Langevin (Wu et al., 2025)](https://arxiv.org/abs/2510.26324) achieves polynomial convergence with L4 score error bounds under local log-concavity — the best known result, but still restrictive.
- Performance guarantees can **diverge with increasing dimension** ([arXiv:2505.18276](https://arxiv.org/pdf/2505.18276)).

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
