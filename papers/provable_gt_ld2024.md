# Provable Convergence and Limitations of Geometric Tempering for Langevin Dynamics (2024)

- **arXiv**: 2410.09697
- **Key finding**: First convergence analysis of geometric tempering π_β ∝ q^{1-β} · π^β for Langevin dynamics. GT-LD achieves exponential improvement in mixing time for multimodal distributions, but can fail for well-conditioned targets if the geometric path passes through distributions with poor functional inequalities.
- **Relevance**: Explains why our geometric β-tempering experiment (iter 4) failed — the path from broad prior to narrow posterior passes through intermediate distributions where the gradient is too weak relative to the noise.
