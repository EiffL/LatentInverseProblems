# Novel Ideas for Calibrated Latent Posterior Sampling

## Design Constraints

Any viable method must:
1. Work for **arbitrary neural network decoders** (black-box, not architecture-specific)
2. Scale to **high-dimensional latent spaces** (e.g., Flux/SD: d_latent ~ 4×64×64 = 16384)
3. Use only **forward passes + vector-Jacobian products** (VJPs) through the decoder — never the full Jacobian J(z) or J^T J
4. Produce **calibrated posteriors**, not just point estimates

### Available cheap operations (per decoder call)
- Forward: D(z) — one eval
- VJP: J(z)^T v — one backward pass
- Score: ∇_z log p_σ(z) — one score network eval
- Posterior gradient: ∇_z log p(z|y) = −z + J(z)^T(y−D(z))/σ_n² — one forward + one backward

### Expensive / unavailable operations
- Full Jacobian J(z) ∈ R^{d_pixel × d_latent} — d_latent backward passes
- Gram matrix J^T J ∈ R^{d_latent × d_latent} — O(d²) cost
- Tweedie covariance in latent space — O(d²) storage

---

## Idea 1: Noise-Space Posterior Sampling (NSPS)

### Core Insight

Replace the diffusion model prior with a **normalizing flow** F: ε → z that maps isotropic Gaussian noise to the latent data distribution. Then **sample the posterior in ε-space** where the prior is trivially N(0,I) and the posterior is provably smoother (Denker et al., ICML 2025). The gradient is computed via standard backprop through the flow + decoder — no ODE solver, no adjoint method, no numerical instability.

This is made practical by the **2024-2025 normalizing flow renaissance**: TarFlow, STARFlow, and SimFlow now achieve image quality competitive with diffusion models (FID 1.91–2.66 on ImageNet) while maintaining exact invertibility and exact log-likelihood. STARFlow and SimFlow operate in VAE latent spaces — exactly the same architecture as Stable Diffusion / Flux.

### The NF Landscape (2024–2026)

| Method | FID (IN-256) | Exact density | Exact inverse | Latent space | Venue |
|--------|-------------|---------------|---------------|-------------|-------|
| **SimFlow+REPA-E** | **1.91** | Yes | Yes | Yes (VAE) | ByteDance 2025 |
| **STARFlow** | 2.40 | Yes | Yes | Yes (VAE) | NeurIPS 2025 Spotlight |
| **TarFlow** | 2.66 (64×64) | Yes | Yes | Pixel | ICML 2025 Oral |
| Flowing Backwards | SOTA (64,256) | Yes | Yes | Pixel | 2025 |
| BiFlow | 2.39 | No | No | Pixel | 2025 |
| DiT (diffusion) | 2.27 | No | Approx | Yes | Reference |

**Key point:** SimFlow at FID 1.91 **beats most diffusion models** while providing exact density and exact invertibility — properties diffusion models fundamentally lack.

**STARFlow** and **SimFlow** are the most relevant to our setting: they are autoregressive Transformer flows trained in the latent space of a pretrained VAE (the same VAE used by Stable Diffusion). They provide:
- Forward mapping ε → z (generation): autoregressive, sequential across patches
- Inverse mapping z → ε (encoding): fully parallel, fast
- Exact log p(z) via change-of-variables formula
- Standard backprop through the entire flow

**Autoregressive bottleneck:** The forward pass (ε→z) is sequential. GS-Jacobi acceleration ([arXiv:2505.12849](https://arxiv.org/abs/2505.12849)) gives 2.5–5.3× speedups. This is the main practical limitation, but it's an active area of improvement.

### Formulation

Given a normalizing flow F: ε → z with exact inverse F⁻¹ and exact log-density:

```
log p_F(z) = log N(F⁻¹(z)|0,I) − log|det ∂F/∂ε|
```

The posterior in noise space:

```
p(ε|y) ∝ N(ε|0,I) · p(y | D(F(ε)))
```

The gradient:

```
∇_ε log p(ε|y) = −ε + [∂(D∘F)/∂ε]ᵀ · (y − D(F(ε))) / σ_n²
                  └─────────── standard backprop ───────────┘
```

The composition D∘F is just a neural network (flow + decoder). The VJP is standard backprop — **no ODE solver, no adjoint method, no checkpointing**.

### Algorithm

1. Initialize ε ~ N(0, I)
2. Run HMC targeting p(ε|y):
   - Leapfrog integrator with L steps
   - Each step: one forward pass through F then D, one backward pass (standard backprop)
   - Metropolis accept/reject
3. Map accepted ε to z₀ = F(ε), then x = D(z₀)

No annealing schedule. No Tweedie approximation. No score-based guidance. No temperature tuning.

### Why This Works

- **Posteriors are provably smoother in noise space.** The flow F acts as a learned preconditioner that "straightens" the posterior geometry. Proved by Denker et al. (ICML 2025) for flow matching models; extends naturally to normalizing flows.
- **The prior in ε-space is N(0,I)** — isotropic Gaussian. No Jacobian distortion, no decoder nonlinearity in the prior term. All complexity is pushed into the likelihood.
- **No Tweedie approximation.** Direct MCMC on the exact (unnormalized) posterior density.
- **Standard backprop.** The flow F is a finite-depth neural network. Gradients via autodiff are exact and numerically stable. No adjoint method needed.
- **Exact density available.** Unlike diffusion models, normalizing flows provide log p_F(z) exactly. This enables Metropolis-Hastings corrections and importance weighting.

### Scalability

- Each HMC leapfrog step: 1 forward pass (F + D) + 1 backward pass ≈ 2× generation cost
- HMC needs O(d^{1/4}) leapfrog steps. For d=16384: ~11 steps
- Total per independent sample: ~22 forward+backward passes through the flow+decoder
- Compare: DPS/MMPS use ~50–200 NFEs; SAL uses ~300

### Novelty

D-Flow SGLD ([arXiv:2602.21469](https://arxiv.org/abs/2602.21469)) does noise-space Langevin for **pixel-space** flow matching models. Nobody has applied this to **latent normalizing flows + decoder composition**. The key differences:

1. **NF instead of ODE:** eliminates the adjoint method bottleneck entirely. Gradients are exact and cheap.
2. **Latent + decoder composition:** the composition D∘F maps noise → latent → pixel, and the noise-space posterior absorbs both the flow prior geometry AND the decoder nonlinearity into a single likelihood term, while keeping the prior trivially simple.
3. **Exact density:** NFs provide log p(z), enabling exact Metropolis corrections. ODE-based flows require expensive Hutchinson trace estimates.

### Risks

- **Autoregressive bottleneck:** The forward pass F(ε) is sequential for TarFlow/STARFlow/SimFlow. Each HMC leapfrog step requires one forward pass. Mitigation: GS-Jacobi acceleration, or future non-autoregressive NF architectures.
- **NF model quality:** If the NF prior p_F(z) ≠ p_true(z), the posterior will be biased. Mitigation: (a) use the best available NF (SimFlow at FID 1.91 is very close to ground truth), (b) importance weight correction using the exact NF density.
- **HMC mixing in noise space:** Although the prior geometry is isotropic, the likelihood landscape in ε-space may have complex structure. Mitigation: NUTS for adaptive trajectory length.

### Fallback: ODE-based variant

If no NF prior is available, the same algorithm works with a probability flow ODE F, using the adjoint method for gradients. This is more expensive and less numerically stable, but does not require training a separate NF model. Rectified flows (as used by Flux) partially mitigate the cost by straightening trajectories, but the adjoint backprop remains the bottleneck.

### Key References

#### Normalizing flows (the prior model)
- Zhai, Zhai, McGill, Littwin, Susskind, Brafman. "TarFlow: Normalizing Flows are Capable Generative Models." ICML 2025 (Oral). [arXiv:2412.06329](https://arxiv.org/abs/2412.06329) — Transformer autoregressive flow, first NF competitive with diffusion. Code: [github.com/apple/ml-tarflow](https://github.com/apple/ml-tarflow)
- Zhai, McGill, Zhai, Susskind, Brafman. "STARFlow: Scaling TARFlow for High-resolution Image Synthesis." NeurIPS 2025 (Spotlight). [arXiv:2506.06276](https://arxiv.org/abs/2506.06276) — TarFlow in VAE latent space, FID 2.40 on IN-256. Code: [github.com/apple/ml-starflow](https://github.com/apple/ml-starflow)
- "SimFlow: Simplified and End-to-End Training of Latent Normalizing Flows." 2025. [arXiv:2512.04084](https://arxiv.org/abs/2512.04084) — Joint VAE+NF training, FID 1.91 with REPA-E. Code: [github.com/ByteDance-Seed/SimFlow](https://github.com/ByteDance-Seed/SimFlow)
- "Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment." 2025. [arXiv:2511.22345](https://arxiv.org/abs/2511.22345) — REPA for NFs, accelerates training 3.3×.
- "GS-Jacobi Iteration for TarFlow Acceleration." 2025. [arXiv:2505.12849](https://arxiv.org/abs/2505.12849) — 2.5–5.3× speedup for autoregressive NF sampling.

#### Noise-space posterior sampling (theoretical foundation)
- Denker, Bhatt, Goan, Drovandi. "Outsourced Diffusion Sampling." ICML 2025. [arXiv:2502.06999](https://arxiv.org/abs/2502.06999) — Proves posteriors are smoother in noise space.
- "D-Flow SGLD: Source-Space Posterior Sampling for Scientific Inverse Problems with Flow Matching." 2026. [arXiv:2602.21469](https://arxiv.org/abs/2602.21469) — SGLD in source space of flow matching (pixel space only).
- Park. "Posterior Sampling via Langevin Dynamics on Generative Priors." 2024. [arXiv:2410.02078](https://arxiv.org/abs/2410.02078) — Langevin in noise space, theoretical bounds.

#### HMC
- Hoffman, Gelman. "The No-U-Turn Sampler." JMLR 2014. — NUTS for automatic HMC tuning.

---

## Idea 2: Latent Split Gibbs with Pixel Augmentation (LSG)

### Core Insight

Extend PnP-DM's Split Gibbs framework to latent space by introducing a **pixel-space auxiliary variable** x. This separates "matching the observation" (exact Gaussian step in pixel space) from "finding the latent code" (decoder inversion via HMC, using only VJPs).

### Formulation

Augmented target:

```
p(z, x | y) ∝ N(z|0,I) · N(x | D(z), γ²I) · p(y | x)
```

where x ∈ R^{d_pixel} is an auxiliary variable and γ controls the coupling strength.

As γ → 0: x → D(z) and the marginal over z converges to p(z|y).

### Algorithm

**Gibbs alternation:**

**Step 1 — x | z, y (exact Gaussian, closed-form):**

For the denoising case (A = I):
```
x ~ N( (D(z)/γ² + y/σ_n²) / (1/γ² + 1/σ_n²),  1/(1/γ² + 1/σ_n²) · I )
```
Just a weighted average of D(z) and y. Exact. Zero approximation error.

For general linear A:
```
x ~ N( (D(z)/γ² + Aᵀy/σ_n²) / (1/γ² + AᵀA/σ_n²),  (1/γ² + AᵀA/σ_n²)^{-1} )
```

**Step 2 — z | x (decoder inversion via HMC):**
```
p(z|x) ∝ N(z|0,I) · N(x | D(z), γ²I)
∇_z log p(z|x) = −z + J(z)ᵀ(x − D(z)) / γ²
```

This is decoder inversion: find z such that D(z) ≈ x, regularized by the prior. Gradient requires only **one VJP**. Run a few HMC steps.

**Annealing:** Start γ large (loose coupling, easy mixing, prior-dominated) → decrease toward zero (tight coupling, likelihood-dominated, approaches true posterior).

### Why This Works

- **Step 1 is exact** — no score approximation, no Tweedie. The pixel-space likelihood is handled analytically in closed form.
- **Step 2 is decoder inversion** — a massively studied problem (GAN inversion, encoder optimization). Well-conditioned for good decoders. Requires only VJPs.
- **The decoder Jacobian distortion is automatically handled.** The Gibbs split separates "matching observations" (Step 1, pixel space, easy) from "finding latent codes" (Step 2, latent space, where decoder geometry lives).
- **Convergence guarantees** from PnP-DM theory: non-asymptotic O(1/K) convergence in Fisher divergence.

### Scalability

- Step 1: O(d_pixel) — trivially parallel, no neural network evaluation
- Step 2: O(L × backward_pass_cost) per Gibbs sweep, where L = HMC trajectory length
- With L=10 HMC steps, K=100 Gibbs sweeps: ~1000 decoder evaluations total
- Embarrassingly parallelizable across observations

### Novelty

PnP-DM ([arXiv:2405.18782](https://arxiv.org/abs/2405.18782)) and the PnP Split Gibbs Sampler ([arXiv:2304.11134](https://arxiv.org/abs/2304.11134)) operate in **pixel space only**. The diffusion model IS the pixel-space prior, so the prior step is just "run denoiser." PSI3D ([arXiv:2512.18367](https://arxiv.org/abs/2512.18367)) uses latent diffusion for 3D volumes but doesn't address the fundamental question of calibrated posterior sampling through a nonlinear decoder.

The extension to latent space via pixel augmentation is non-trivial:
- The auxiliary x bridges latent ↔ pixel, absorbing the decoder nonlinearity
- The γ-annealing provides a smooth path from easy to hard that doesn't exist in standard PnP-DM
- Step 2 (decoder inversion) replaces the "run denoiser" step with a geometrically meaningful MCMC sub-problem

### Risks

- γ schedule sensitivity: too fast → poor mixing; too slow → wasted computation. Mitigate with adaptive γ based on acceptance rate.
- Step 2 mixing in high-d: HMC must adequately explore p(z|x) at each sweep. For tight γ, this is a narrow distribution that's easy to sample; for loose γ, it's broad but smooth.
- Representation error: if D is not surjective, some x values may not have good latent pre-images. Mitigate by keeping γ > 0 (soft coupling).

### Key References

- Wu, Sun, Chen, Zhang, Yue, Bouman. "Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors." NeurIPS 2024. [arXiv:2405.18782](https://arxiv.org/abs/2405.18782) — PnP-DM: Split Gibbs in pixel space, 97.46% coverage, O(1/K) convergence.
- Xu, Chi. "Provably Robust Score-Based Diffusion Posterior Sampling for Plug-and-Play Image Reconstruction." NeurIPS 2024. [arXiv:2403.17042](https://arxiv.org/abs/2403.17042) — DPnP: proximal consistency + denoising diffusion, pixel space.
- Laumont, De Bortoli, Almansa, Delon, Durmus, Pereyra. "On Maximum a Posteriori Estimation with Plug & Play Priors and Stochastic Gradient Descent." JMLR 2022. [arXiv:2201.06133](https://arxiv.org/abs/2201.06133) — PnP-SGD convergence theory.
- "PSI3D: Posterior Sampling-Based Inference from a Single 3D Image." Dec 2025. [arXiv:2512.18367](https://arxiv.org/abs/2512.18367) — Latent diffusion + split Gibbs for 3D (but no calibration analysis).
- Giovannelli. "A Gibbs Posterior Sampler for Inverse Problem Based on Prior Diffusion Model." 2026. [arXiv:2602.11059](https://arxiv.org/abs/2602.11059) — Full Gibbs with diffusion chain as auxiliary, linear forward models.

---

## Idea 3: Simulated Tempering with Score-Derived Free Energies (ST-SFE)

### Core Insight

Diffusion noise levels σ are temperatures. The score function at level σ gives the tempered prior p_σ(z). **Treat σ as a dynamic variable** sampled jointly with z via simulated tempering. The free energies needed for tempering can be computed analytically for the prior component and estimated cheaply for the likelihood component.

### Formulation

Joint target:

```
p(z, σ | y) ∝ p_σ(z|y) · w(σ)
```

where:
- p_σ(z|y) ∝ p_σ(z) · p(y|z)^{α(σ)} is the tempered posterior at noise level σ
- p_σ(z) = N(0, (σ₀²+σ²)I) is the noised prior (analytic for Gaussian prior)
- α(σ) controls likelihood tempering (e.g., α(σ) = σ₀²/(σ₀²+σ²) to match prior tempering)
- w(σ) = 1/Z(σ) are free energy weights

### Algorithm

The Markov chain makes two types of moves:

**z-move (fixed σ):** HMC step targeting p_σ(z|y)
```
∇_z log p_σ(z|y) = −z/(σ₀²+σ²) + α(σ) · J(z)ᵀ(y−D(z))/σ_n²
```
Requires one VJP per leapfrog step. Standard HMC.

**σ-move (fixed z):** Propose σ' ~ q(σ'|σ), accept with Metropolis ratio:
```
log α = log p_{σ'}(z|y) + log w(σ') − log p_σ(z|y) − log w(σ)
```

The chain **automatically discovers the right annealing schedule** by exploring (z, σ) jointly.

### Computing the Free Energies

- **Prior contribution:** log Z_prior(σ) = (d/2) log(2π(σ₀²+σ²)) — **analytic** for Gaussian prior
- **Likelihood contribution:** estimated via thermodynamic integration along the σ path:
  ```
  log Z(σ) = log Z(σ_max) + ∫_{σ_max}^{σ} E_{p_{σ'}(z|y)}[∂ log p_{σ'}(z|y)/∂σ'] dσ'
  ```
  The integrand can be estimated from short MCMC runs at a few σ values, then interpolated.
- **Alternatively:** Wang-Landau or expanded ensemble methods to estimate w(σ) adaptively during the run.

### Why This Works

- **Self-tuning.** No fixed schedule of noise levels. The chain visits whichever σ values are needed to bridge the gap between prior and posterior. Narrow posteriors → chain spends more time at small σ. Broad posteriors → larger σ suffices.
- **Eliminates the most sensitive hyperparameters** in SAL/MALA-SAL: N_levels, σ_max, σ_min, steps_per_level. All replaced by a single adaptive Markov chain.
- **Correct by construction.** Simulated tempering with exact free energies produces samples from the target p(z|y) = p_{σ=0}(z|y). Even with approximate free energies, the algorithm remains valid (just less efficient mixing).
- **Theoretically novel connection.** This is the first formulation linking diffusion noise levels to simulated tempering free energies. The score function provides both the tempered prior AND the machinery to estimate free energies.

### Scalability

- z-moves: same cost as SA-HMC — O(d^{1/4}) gradient evaluations per move, each requiring one VJP
- σ-moves: essentially free (evaluate log-density ratio at two σ values)
- No per-level equilibration: the chain flows continuously in (z, σ) space
- Parallelizable: multiple independent chains with different initializations

### Novelty

Simulated tempering is classical (Marinari & Parisi 1992, Geyer & Thompson 1995). Score-annealed Langevin (SAL) uses a fixed temperature schedule. **Nobody has connected these**: treating the diffusion noise level as a tempering variable with score-derived free energies. The analytic prior free energy (a unique property of diffusion models with Gaussian priors) makes this practical — in classical simulated tempering, estimating free energies is the main bottleneck.

### Risks

- Free energy estimation for the likelihood component may require significant upfront computation. Mitigate with Wang-Landau adaptive estimation.
- σ-move acceptance rate may be low if the free energies are inaccurate. Mitigate with small σ proposals and iterative refinement of w(σ).
- The coupling between z and σ may slow mixing if the posterior shape changes dramatically across σ values.

### Key References

- Marinari, Parisi. "Simulated Tempering: A New Monte Carlo Scheme." Europhysics Letters 1992. — Original simulated tempering.
- Wang, Landau. "Efficient, Multiple-Range Random Walk Algorithm to Calculate the Density of States." PRL 2001. — Adaptive free energy estimation.
- Wu et al. "Posterior Sampling by Combining Diffusion Models with Annealed Langevin." 2025. [arXiv:2510.26324](https://arxiv.org/abs/2510.26324) — Annealed Langevin with diffusion scores (fixed schedule).
- Geyer, Thompson. "Annealing Markov Chain Monte Carlo with Applications to Ancestral Inference." JASA 1995. — Simulated tempering theory.

---

## Idea 4: Stochastic Localization for Inverse Problems (SLIPS-Posterior)

### Core Insight

Stochastic localization (El Alaoui, Montanari, Sellke 2023) provides a fundamentally different paradigm from diffusion: instead of adding noise and reversing, it **progressively reveals information** about the target through a virtual observation process. The denoiser at each step is estimated via MCMC — no neural network approximation needed.

### Formulation

The stochastic localization process evolves a virtual observation Y_t:
```
dY_t = m_t dt + dW_t,    Y_0 = 0
```
where m_t = E_{π}[z | Y_t] is the conditional mean under the target π(z) = p(z|y).

At each time t, the tilted target is:
```
q_t(z) ∝ π(z) · N(Y_t | tz, tI) = p(z|y) · exp(−‖z − Y_t/t‖² · t/2)
```

The tilting adds a quadratic term that makes q_t **increasingly concentrated** as t grows → MCMC at later steps is trivially easy (fast mixing).

### Algorithm

1. Y_0 = 0, initialize z from prior
2. For t = 0, δt, 2δt, ..., T:
   - Run K HMC steps on q_t(z):
     ```
     ∇_z log q_t = −z + J(z)ᵀ(y−D(z))/σ_n² + t(Y_t/t − z)
     ```
   - Estimate m_t = mean of HMC samples
   - Update: Y_{t+δt} = Y_t + m_t · δt + √δt · ε
3. Output the final HMC sample

### Why This Works

- **No Tweedie approximation.** The MCMC uses the exact posterior gradient at each step.
- **No score function needed.** The gradient only involves the prior term (−z), the likelihood gradient (VJP through decoder), and the tilting term (quadratic, trivial).
- **Natural concentration schedule.** At small t, q_t ≈ p(z|y) (broad, hard). At large t, q_t concentrates around a point (narrow, easy). This provides automatic annealing without a noise schedule.
- **Convergence guarantees.** SLIPS converges under mild conditions on the MCMC quality at each step (Huang, Montanari, 2024).

### Scalability

- Each HMC step at each localization level: one forward + one backward through decoder
- HMC with O(d^{1/4}) leapfrog steps
- T localization levels × K HMC steps × L leapfrog steps × (1 forward + 1 backward)
- Comparable to SAL in total cost

### Novelty

SLIPS ([arXiv:2402.10758](https://arxiv.org/abs/2402.10758)) exists for unconditional sampling. It has **never been applied to inverse problems** or latent-space models. The key adaptation: the target π is p(z|y) instead of p(z), and the MCMC at each step incorporates the likelihood gradient via decoder VJPs.

### Key References

- Alaoui, Montanari, Sellke. "Sampling from the Sherrington-Kirkpatrick Gibbs Measure via Algorithmic Stochastic Localization." 2023. [arXiv:2305.10690](https://arxiv.org/abs/2305.10690) — Theoretical foundations connecting diffusion, MCMC, and stochastic localization.
- Huang, Montanari. "SLIPS: Stochastic Localization via Iterative Posterior Sampling." ICML 2024. [arXiv:2402.10758](https://arxiv.org/abs/2402.10758) — Practical SLIPS algorithm with MCMC denoiser. Code: [github.com/h2o64/slips](https://github.com/h2o64/slips).
- Montanari. "Sampling, Diffusions, and Stochastic Localization." ICML tutorial 2024. — Survey of connections between these frameworks.

---

## Idea 5: Riemannian HMC via Implicit Metric (IR-HMC)

### Core Insight

The decoder induces a Riemannian metric G(z) = J(z)^T J(z) on latent space. Directions where the decoder is insensitive (small singular values of J) need larger steps; directions where it's sensitive need smaller steps. **Use this metric as a preconditioner for HMC without ever materializing the full matrix**, via conjugate gradient solves using only VJPs.

### Formulation

Riemannian HMC with metric M(z) = I + J(z)^T J(z) / σ_n²:

```
Leapfrog step:
  p ← p − (ε/2) ∇_z H(z, p)
  z ← z + ε M(z)^{-1} p
  p ← p − (ε/2) ∇_z H(z, p)
```

where H(z, p) = −log p(z|y) + (1/2) pᵀ M(z)^{-1} p + (1/2) log det M(z).

### Implicit Metric Solve

M(z)^{-1} v requires solving M v' = v. Since M = I + J^T J/σ_n², each matrix-vector product Mv = v + J^T(Jv)/σ_n² requires:
- One forward-mode JVP: Jv (or finite-difference approximation)
- One VJP: J^T(Jv)

Solve via **conjugate gradient** (CG): converges in O(√κ) iterations where κ = cond(M). Each CG iteration: 1 JVP + 1 VJP ≈ 2 decoder evaluations.

For well-conditioned decoders (κ ~ 10–100): CG converges in ~10 iterations.

### Why This Works

- **Handles Jacobian distortion automatically.** The metric adapts step sizes to the decoder's sensitivity per direction — the core unsolved problem for latent inverse problems.
- **No full Jacobian needed.** Only JVPs and VJPs, both O(1 decoder eval).
- **Optimal MCMC scaling.** Riemannian HMC achieves dimension-independent mixing when the metric matches the target's local curvature.

### Scalability

- Per leapfrog step: ~2×CG_iters decoder evaluations for the metric solve + 1 gradient evaluation
- With CG_iters=10, L=10 leapfrog steps: ~200 decoder evaluations per HMC sample
- More expensive per step than flat HMC, but far fewer steps needed due to better geometry

### Novelty

The pullback Riemannian metric for diffusion models was studied theoretically (arXiv:2410.01950, ICML 2025). Running Langevin in the latent space of normalizing flows with Jacobian correction has been explored (Caterini et al., Machine Learning 2024). **Nobody has combined the decoder's pullback metric with implicit CG solves for calibrated posterior sampling in latent inverse problems.**

### Key References

- Girolami, Calderhead. "Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods." JRSS-B 2011. — Riemannian MCMC foundations.
- Ross, Caterini, Cresswell, Loaiza-Ganem. "Sampling in the Latent Space of Normalizing Flows with Langevin." Machine Learning 2024. [Springer](https://link.springer.com/article/10.1007/s10994-024-06623-x) — MALA in NF latent space with Jacobian correction.
- Chen, Gao, Miolane. "Score-based Pullback Riemannian Geometry." ICML 2025. [arXiv:2410.01950](https://arxiv.org/abs/2410.01950) — Riemannian metric from diffusion scores.
- Park. "Posterior Sampling via Langevin Dynamics on Generative Priors." 2024. [arXiv:2410.02078](https://arxiv.org/abs/2410.02078) — Langevin in noise space of generative models, theoretical bounds.

---

## Comparative Summary

| | NSPS (1) | LSG (2) | ST-SFE (3) | SLIPS (4) | IR-HMC (5) |
|---|---|---|---|---|---|
| Core mechanism | HMC in ε-space | Gibbs with pixel aux | Joint (z,σ) chain | Localization + MCMC | Metric-adapted HMC |
| Avoids Tweedie? | Yes | Yes | Partial | Yes | Yes |
| Annealing schedule | None | γ anneal | Self-tuning | Auto (via t) | None or SAL-style |
| Decoder access | Adjoint VJP | VJP only | VJP only | VJP only | JVP + VJP (CG) |
| Theory | Smoothness proof | PnP-DM O(1/K) | ST guarantees | SL convergence | Riemannian MCMC |
| NFEs per sample | ~20×ODE_steps | ~1000 | ~1000 | ~1000 | ~200 per step |
| Main risk | Adjoint stability | γ sensitivity | Free energy est. | MCMC quality | CG convergence |
| Novelty level | High | High | Very high | High | Moderate |

### Recommendation

- **Most practically promising:** Idea 2 (Latent Split Gibbs) — clean, convergence theory, only VJPs
- **Most intellectually surprising:** Idea 1 (Noise-Space) — reverses the conventional wisdom of where to sample
- **Most theoretically elegant:** Idea 3 (Simulated Tempering) — unifies diffusion noise levels with statistical mechanics
- **Most paradigm-breaking:** Idea 4 (SLIPS) — entirely different framework from diffusion, with formal guarantees
- **Most targeted at the core problem:** Idea 5 (Riemannian HMC) — directly addresses Jacobian distortion

---

## Additional References (Cross-cutting)

### Latent-space inverse problem methods
- Rout, Raoof, Daras, Caramanis, Dimakis, Shakkottai. "PSLD: Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models." NeurIPS 2023. [arXiv:2307.00619](https://arxiv.org/abs/2307.00619)
- Rout, Chen, Kumar, Caramanis, Shakkottai, Chu. "STSL: Beyond First-Order Tweedie." CVPR 2024. [arXiv:2312.00852](https://arxiv.org/abs/2312.00852)
- Achituve, Habi, Rosenfeld, Netzer, Diamant, Fetaya. "LD-SMC: Inverse Problem Sampling in Latent Space Using Sequential Monte Carlo." ICML 2025. [arXiv:2502.05908](https://arxiv.org/abs/2502.05908)
- Rao, Qu, Moyer. "Latent Diffusion Inversion Requires Understanding the Latent Space." 2025. [arXiv:2511.20592](https://arxiv.org/abs/2511.20592)

### SMC and particle methods
- Wu, Trippe, Naesseth, Blei, Cunningham. "Twisted Diffusion Sampler (TDS)." NeurIPS 2023. [arXiv:2306.17775](https://arxiv.org/abs/2306.17775)
- Dou, Song. "Filtering Posterior Sampling (FPS)." ICLR 2024. [OpenReview](https://openreview.net/forum?id=tplXNcHZs1)
- Cardoso et al. "MCGdiff." 2023. [arXiv:2308.07983](https://arxiv.org/abs/2308.07983)

### SVGD and particle-based methods
- Liu, Wang. "Stein Variational Gradient Descent." NeurIPS 2016. [arXiv:1608.04471](https://arxiv.org/abs/1608.04471)
- Chen, Ren, Wang, Batmanghelich. "Dual-Space Posterior Sampling (ADMM+SVGD)." 2026. [arXiv:2603.00393](https://arxiv.org/abs/2603.00393)
- Corso, Bose, Jing, Barzilay, Jaakkola. "Particle Guidance." ICLR 2024. [arXiv:2310.13102](https://arxiv.org/abs/2310.13102)

### Uncertainty quantification benchmarks
- Qiu, Yang, Liu, Wang, Shen. "Benchmarking Uncertainty Quantification of Plug-and-Play Diffusion Priors." 2026. [arXiv:2602.04189](https://arxiv.org/abs/2602.04189)
- Scope Crafts, Villa. "BIPSDA Benchmark." 2025. [arXiv:2503.03007](https://arxiv.org/abs/2503.03007)

### Complexity results
- Gupta, Chen, Chen. "Diffusion Posterior Sampling is Computationally Intractable." ICML 2024. [arXiv:2402.12727](https://arxiv.org/abs/2402.12727)

### Layerwise uncertainty propagation (background for decoder structure)
- Gast, Roth. "Lightweight Probabilistic Deep Networks (EKF Propagation)." 2018. [arXiv:1809.06009](https://arxiv.org/abs/1809.06009)
- Magris, Shabani, Iosifidis. "ADF and Smoothing with Neural Network Surrogates." Nov 2025. [arXiv:2511.09016](https://arxiv.org/abs/2511.09016)
