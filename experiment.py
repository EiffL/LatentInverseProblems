"""Tune MAP-init MALA-SAL: more restarts, better MAP search."""
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lip import MNISTVAE
from lip.metrics import latent_calibration_test
from lip.solvers import mala_sal

problem = MNISTVAE(sigma_n=0.4)


def make_mala_sal_mapinit(n_restarts=5, N_map_steps=100, lr_map=5e-4,
                           N_levels=10, N_langevin=30, sigma_max=0.1,
                           sigma_min=0.01, lr_scale=0.5):
    """MALA-SAL with multi-restart MAP initialization."""

    def _solve_single(problem, y, key):
        d = problem.d_latent
        s02 = problem.sigma_0**2
        sn2 = problem.sigma_n**2

        # --- Phase 1: MAP search ---
        def log_posterior(z):
            log_prior = -0.5 * jnp.sum(z**2) / s02
            x_hat = problem.decoder(z[None])[0]
            log_lik = -0.5 * jnp.sum((y - x_hat)**2) / sn2
            return log_prior + log_lik

        grad_post = jax.grad(log_posterior)

        keys = jax.random.split(key, n_restarts + 2)
        key_chain = keys[0]

        # Generate inits: encoder + prior samples
        z_enc = problem.encoder(y)
        inits = [z_enc]
        for r in range(1, n_restarts):
            inits.append(s02**0.5 * jax.random.normal(keys[r + 1], (d,)))

        # Gradient ascent from each
        def grad_ascent(z0):
            def step(z, _):
                g = grad_post(z)
                return z + lr_map * g, None
            z_final, _ = jax.lax.scan(step, z0, None, length=N_map_steps)
            return z_final, log_posterior(z_final)

        best_z = z_enc
        best_lp = log_posterior(z_enc)
        for z0 in inits:
            z_f, lp_f = grad_ascent(z0)
            best_z = jnp.where(lp_f > best_lp, z_f, best_z)
            best_lp = jnp.maximum(lp_f, best_lp)

        # --- Phase 2: MALA chain ---
        sigmas = jnp.geomspace(sigma_max, sigma_min, N_levels)

        def log_density(z, sigma_t):
            alpha_t = s02 / (s02 + sigma_t**2)
            log_prior = -0.5 * jnp.sum(z**2) / (s02 + sigma_t**2)
            z_hat0 = alpha_t * z
            x_hat = problem.decoder(z_hat0[None])[0]
            log_lik = -0.5 * jnp.sum((y - x_hat)**2) / sn2
            return log_prior + log_lik

        log_and_grad = jax.value_and_grad(log_density)
        z = best_z

        def langevin_block(z_key, sigma_t):
            z, key = z_key
            lr = lr_scale * sigma_t**2

            def mala_step(carry, _):
                z, key, log_p_z, grad_z = carry
                key, k_prop, k_accept = jax.random.split(key, 3)
                noise = jax.random.normal(k_prop, (d,))
                z_prop = z + lr * grad_z + jnp.sqrt(2 * lr) * noise
                log_p_prop, grad_prop = log_and_grad(z_prop, sigma_t)
                diff_fwd = z_prop - z - lr * grad_z
                diff_bwd = z - z_prop - lr * grad_prop
                log_q_fwd = -0.5 * jnp.sum(diff_fwd**2) / (2 * lr)
                log_q_bwd = -0.5 * jnp.sum(diff_bwd**2) / (2 * lr)
                log_alpha = (log_p_prop - log_p_z) + (log_q_bwd - log_q_fwd)
                accept = jnp.log(jax.random.uniform(k_accept)) < log_alpha
                z_new = jnp.where(accept, z_prop, z)
                log_p_new = jnp.where(accept, log_p_prop, log_p_z)
                grad_new = jnp.where(accept, grad_prop, grad_z)
                return (z_new, key, log_p_new, grad_new), None

            log_p_z, grad_z = log_and_grad(z, sigma_t)
            (z, key, _, _), _ = jax.lax.scan(
                mala_step, (z, key, log_p_z, grad_z), None, length=N_langevin)
            return (z, key), None

        (z, _), _ = jax.lax.scan(langevin_block, (z, key_chain), sigmas)
        return z

    _jit_fn = [None]

    def solver(problem, y, key, **kwargs):
        if _jit_fn[0] is None:
            _jit_fn[0] = jax.jit(lambda y, key: _solve_single(problem, y, key))
        if y.ndim == 1:
            return _jit_fn[0](y, key)
        keys = jax.random.split(key, y.shape[0])
        out = []
        for i in range(y.shape[0]):
            out.append(_jit_fn[0](y[i], keys[i]))
        return jnp.stack(out)

    return solver


if __name__ == "__main__":
    configs = {
        "MALA-SAL": ("base", {}),
        "MAP5": ("map", dict(n_restarts=5, N_map_steps=100, lr_map=5e-4)),
        "MAP10": ("map", dict(n_restarts=10, N_map_steps=100, lr_map=5e-4)),
        "MAP5-lr1e-3": ("map", dict(n_restarts=5, N_map_steps=200, lr_map=1e-3)),
    }

    # Run with 3 seeds × n=200
    for seed in [0, 42, 123]:
        key = jax.random.PRNGKey(seed)
        print(f"\n=== Seed {seed} (n=200) ===")
        for label, (kind, cfg) in configs.items():
            if kind == "base":
                solver = mala_sal
            else:
                solver = make_mala_sal_mapinit(**cfg)
            t0 = time.time()
            r = latent_calibration_test(problem, solver, key, n=200)
            t1 = time.time()
            print(f"  {label}: hpd={r['hpd_mean']:.3f}, KS={r['hpd_ks']:.3f} ({t1-t0:.0f}s)")
