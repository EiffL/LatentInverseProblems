# Divide-and-Conquer Posterior Sampling for Denoising Diffusion Priors (Janati, Olsson, Moulines 2024)

- **arXiv**: 2403.11407 (NeurIPS 2024)
- **Key finding**: Decomposes the DDM reverse process into a sequence of intermediate posteriors conditioned on y. Each intermediate posterior is simpler than the full posterior. The "divide" is across noise levels; the "conquer" is MCMC at each level.
- **Relevance**: Structural analog of SAL — validates the approach of running MCMC at each noise level. Provides formal justification for the multi-level decomposition.
