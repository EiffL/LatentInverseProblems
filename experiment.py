"""Scratch pad for prototyping new solvers on MNISTVAE.

Iteration 5: High-confidence comparison of MALA-SAL vs SAL.
Run with n=500 and multiple seeds to reduce variance.
"""
import time
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lip import MNISTVAE
from lip.metrics import latent_calibration_test
from lip.solvers import score_annealed_langevin, mala_sal

problem = MNISTVAE(sigma_n=0.4)


if __name__ == "__main__":
    n_cal = 500

    solvers = {
        "SAL": score_annealed_langevin,
        "MALA-SAL": mala_sal,
    }

    # Run with 3 different seeds
    for seed in [0, 42, 123]:
        key = jax.random.PRNGKey(seed)
        print(f"\n=== Seed {seed} (n={n_cal}) ===")
        for name, solver in solvers.items():
            t0 = time.time()
            r = latent_calibration_test(problem, solver, key, n=n_cal)
            t1 = time.time()
            print(f"  {name}: hpd={r['hpd_mean']:.3f}, KS={r['hpd_ks']:.3f} ({t1-t0:.0f}s)")
