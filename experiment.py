"""Scratch pad for prototyping new solvers on MNISTVAE.

Iteration 1: Score-Annealed Langevin (SAL) -- final validation.
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

problem = MNISTVAE(sigma_n=0.4)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_experiment(problem, results, title, outdir="results/experiments"):
    os.makedirs(outdir, exist_ok=True)
    n_solvers = len(results)
    fig, axes = plt.subplots(1, n_solvers, figsize=(5 * n_solvers, 4))
    if n_solvers == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        hpd = np.array(r["hpd_levels"])
        ax.hist(hpd, bins=20, range=(0, 1), density=True, alpha=0.7,
                color="steelblue", edgecolor="white")
        ax.axhline(1.0, color="red", ls="--", lw=1.5, label="Uniform(0,1)")
        ax.set_xlabel("HPD level")
        ax.set_ylabel("Density")
        ax.set_title(f"{name}\nhpd={r['hpd_mean']:.3f}, KS={r['hpd_ks']:.3f}")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    iter_num = _get_next_iter()
    fname = f"iter{iter_num:03d}_{title.lower().replace(' ', '_').replace('/', '_')[:60]}.png"
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")
    return path


def _get_next_iter():
    try:
        with open("results.tsv") as f:
            lines = f.readlines()
        iters = [int(line.split('\t')[0]) for line in lines[1:] if line.strip()]
        return max(iters) + 1 if iters else 1
    except FileNotFoundError:
        return 1


# ---------------------------------------------------------------------------
# Score-Annealed Langevin (SAL) -- optimized with jax.grad + lax.scan
# ---------------------------------------------------------------------------

def _sal_single(problem, y, key, *, N_levels, N_langevin,
                sigma_max, sigma_min, lr_scale):
    """Score-Annealed Langevin for a single observation.

    At each noise level sigma_t (decreasing geometrically from sigma_max
    to sigma_min), runs N_langevin Langevin steps targeting:

      p_t(z|y) ∝ p_{sigma_t}(z) * N(y | D(alpha_t * z), sn2 I)

    where alpha_t = sigma_0^2 / (sigma_0^2 + sigma_t^2) is the Tweedie
    shrinkage factor. Uses jax.grad for efficient gradient computation.

    This is a diffusion-based method: it uses the prior score function
    grad log p_t(z) = -z/(sigma_0^2 + sigma_t^2) at 10 noise levels
    as its generative prior, combined with Tweedie-based likelihood guidance.
    """
    d = problem.d_latent
    s02 = problem.sigma_0**2
    sn2 = problem.sigma_n**2
    sigmas = jnp.geomspace(sigma_max, sigma_min, N_levels)

    def log_density(z, sigma_t):
        alpha_t = s02 / (s02 + sigma_t**2)
        log_prior = -0.5 * jnp.sum(z**2) / (s02 + sigma_t**2)
        x_hat = problem.decoder(z_hat0 := alpha_t * z)
        log_lik = -0.5 * jnp.sum((y - x_hat)**2) / sn2
        return log_prior + log_lik

    # Workaround: walrus doesn't work in JAX traced code; rewrite
    def log_density(z, sigma_t):
        alpha_t = s02 / (s02 + sigma_t**2)
        log_prior = -0.5 * jnp.sum(z**2) / (s02 + sigma_t**2)
        z_hat0 = alpha_t * z
        x_hat = problem.decoder(z_hat0[None])[0]
        log_lik = -0.5 * jnp.sum((y - x_hat)**2) / sn2
        return log_prior + log_lik

    grad_fn = jax.grad(log_density)

    z = problem.encoder(y)

    def langevin_block(z_key, sigma_t):
        z, key = z_key
        lr = lr_scale * sigma_t**2

        def step(carry, _):
            z, key = carry
            key, k = jax.random.split(key)
            g = grad_fn(z, sigma_t)
            z = z + lr * g + jnp.sqrt(2 * lr) * jax.random.normal(k, (d,))
            return (z, key), None

        (z, key), _ = jax.lax.scan(step, (z, key), None, length=N_langevin)
        return (z, key), None

    (z, _), _ = jax.lax.scan(langevin_block, (z, key), sigmas)
    return z


_jit_sal = None
_jit_sal_config = None


def score_annealed_langevin(problem, y, key, *, N_levels=10, N_langevin=30,
                             sigma_max=0.1, sigma_min=0.01, lr_scale=0.5,
                             **kwargs):
    """Score-Annealed Langevin: multi-level Langevin with Tweedie likelihood."""
    global _jit_sal, _jit_sal_config

    config = (id(problem), N_levels, N_langevin, sigma_max, sigma_min, lr_scale)
    if _jit_sal is None or _jit_sal_config != config:
        _jit_sal = jax.jit(
            lambda y, key: _sal_single(
                problem, y, key, N_levels=N_levels, N_langevin=N_langevin,
                sigma_max=sigma_max, sigma_min=sigma_min, lr_scale=lr_scale)
        )
        _jit_sal_config = config

    if y.ndim == 1:
        return _jit_sal(y, key)

    keys = jax.random.split(key, y.shape[0])
    results = []
    for i in range(y.shape[0]):
        results.append(_jit_sal(y[i], keys[i]))
    return jnp.stack(results)


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)  # Different seed for validation
    n_cal = 200

    results = {}

    # Final validation: SAL with best parameters
    label = "SAL(10L×30, smax=0.1)"
    print(f"\n=== {label} (n={n_cal}) ===")
    t0 = time.time()
    r = latent_calibration_test(
        problem, score_annealed_langevin, key, n=n_cal)
    t1 = time.time()
    print(f"hpd_mean={r['hpd_mean']:.3f}, KS={r['hpd_ks']:.3f} ({t1-t0:.0f}s)")
    results[label] = r

    plot_experiment(problem, results, "SAL final validation n=200")
    print(f"\nTotal time: {t1-t0:.0f}s for n={n_cal}")
