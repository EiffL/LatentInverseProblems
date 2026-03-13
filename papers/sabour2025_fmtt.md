# Test-time Scaling of Diffusions with Flow Maps (Sabour et al., 2025)

- **arXiv**: 2511.22688
- **Key idea**: Use trained **flow maps** X_{t,1}(z_t) → z_1 for accurate "look-ahead" during guided generation. The flow map predicts where a trajectory will land at the clean distribution, even from noisy intermediate states. Enables principled reward/likelihood evaluation throughout diffusion generation.
- **Key result**: Importance weights simplify to A_t = ∫ r(X_{s,1}(z̃_s)) ds (just accumulate reward along look-ahead trajectory). Provides exact samples via Jarzynski reweighting.
- **Why Tweedie fails at high noise**: The Tweedie denoiser (1-step mean prediction) gives blurry/inaccurate predictions at early timesteps, producing unhelpful reward gradients. Flow maps give sharp, accurate predictions.
- **Relevance to our work**: For MNISTVAE with Gaussian prior, Tweedie is already exact (prior is Gaussian), so the flow map advantage doesn't manifest. But for realistic problems with trained diffusion priors, this approach would be essential for accurate likelihood evaluation at high noise levels.
- **Our experiments**: (1) Iterative look-ahead (gradient ascent to MAP) washes out gradient signal → catastrophic over-dispersion. (2) First-order MMPS correction weakens likelihood → wrong direction. A trained flow map (fixed smooth function) avoids both failure modes.
- **Thermodynamic length Λ**: Model-agnostic diagnostic for comparing guidance strategies. Could be useful for optimizing annealing schedules.
