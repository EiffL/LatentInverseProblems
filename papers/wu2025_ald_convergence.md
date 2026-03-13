# Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics (2025)

- **arXiv**: 2510.26324 (NeurIPS 2025, Microsoft Research)
- **Key finding**: Proves that annealed Langevin dynamics (ALD) achieves polynomial-time convergence for log-concave likelihoods with only L^4-accurate scores, whereas standard (non-annealed) Langevin requires sub-exponential score error bounds. First theoretical justification for why annealing helps with score estimation error.
- **Relevance**: Provides principled guidance on designing the annealing schedule. Confirms that multi-level annealing is theoretically necessary, not just empirically helpful.
