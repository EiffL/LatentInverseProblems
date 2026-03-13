"""Correct MMPS likelihood: broaden noise model, don't move eval point.

The MMPS approach evaluates D at the same α_t·z (Tweedie point) but
uses an inflated noise covariance:
  p(y|z_t) ≈ N(y | D(α_t·z), σ_n²I + J·Σ_t·J^T)

This naturally anneals the likelihood at higher noise levels — exactly
matching the prior broadening. This is the program.md intuition realized!

For d_latent=2 << d_pixel=784, Woodbury makes the 784×784 inverse cheap.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
from lip import MNISTVAE
from lip.metrics import latent_calibration_test

problem = MNISTVAE(sigma_n=0.4)


def make_mmps_mala_sal(N_levels=10, N_langevin=30, sigma_max=0.1,
                        sigma_min=0.01, lr_scale=0.5):
    """MALA-SAL with MMPS-broadened likelihood."""

    def _single(problem, y, key):
        d = problem.d_latent
        s02 = problem.sigma_0**2
        sn2 = problem.sigma_n**2
        sigmas = jnp.geomspace(sigma_max, sigma_min, N_levels)

        def log_density(z, sigma_t):
            alpha_t = s02 / (s02 + sigma_t**2)
            sigma_tweedie = s02 * sigma_t**2 / (s02 + sigma_t**2)  # scalar

            # Noised prior
            log_prior = -0.5 * jnp.sum(z**2) / (s02 + sigma_t**2)

            # Tweedie estimate and decoder
            z0_tweedie = alpha_t * z
            x_hat = problem.decoder(z0_tweedie[None])[0]  # (784,)
            residual = y - x_hat  # (784,)

            # Decoder Jacobian at Tweedie point
            J = problem.decoder_jacobian(z0_tweedie[None])[0]  # (784, 2)

            # MMPS likelihood: N(y | x_hat, σ_n²I + J·Σ_t·J^T)
            # Using Woodbury for (σ_n²I + J·σ_tw·J^T)^{-1}·r:
            #   = r/σ_n² - J·(I/σ_tw + J^TJ/σ_n²)^{-1}·J^T·r / σ_n⁴
            JtJ = J.T @ J  # (2, 2)
            Jt_r = J.T @ residual  # (2,)

            # Inner 2×2 matrix: I/σ_tw + J^TJ/σ_n²
            A = jnp.eye(d) / sigma_tweedie + JtJ / sn2  # (2, 2)
            A_inv = jnp.linalg.inv(A)

            # Quadratic form: r^T (σ_n²I + J σ_tw J^T)^{-1} r
            quad = (jnp.sum(residual**2) / sn2
                    - Jt_r @ A_inv @ Jt_r / sn2**2)

            # Log-determinant: log|σ_n²I + J σ_tw J^T|
            # = 784·log(σ_n²) + log|I_2 + σ_tw·J^TJ/σ_n²|
            M = jnp.eye(d) + sigma_tweedie * JtJ / sn2  # (2, 2)
            log_det_extra = jnp.log(jnp.linalg.det(M))
            # Full: 784·log(σ_n²) + log_det_extra
            # But 784·log(σ_n²) is constant, drops out of gradient

            log_lik = -0.5 * quad - 0.5 * log_det_extra

            return log_prior + log_lik

        log_and_grad = jax.value_and_grad(log_density)
        z = problem.encoder(y)

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

        (z, _), _ = jax.lax.scan(langevin_block, (z, key), sigmas)
        return z

    _jit_fn = [None]

    def solver(problem, y, key, **kwargs):
        if _jit_fn[0] is None:
            _jit_fn[0] = jax.jit(lambda y, key: _single(problem, y, key))
        if y.ndim == 1:
            return _jit_fn[0](y, key)
        keys = jax.random.split(key, y.shape[0])
        out = []
        for i in range(y.shape[0]):
            out.append(_jit_fn[0](y[i], keys[i]))
        return jnp.stack(out)

    return solver


if __name__ == "__main__":
    from lip.solvers import mala_sal
    from lip.solvers.mala_sal import _mala_sal_single

    key = jax.random.PRNGKey(42)
    n_cal = 100

    # Helper to make baseline MALA with custom sigma_max
    def _make_base(sm):
        _jf = [None]
        def solver(problem, y, key, **kw):
            if _jf[0] is None:
                _jf[0] = jax.jit(lambda y, key: _mala_sal_single(
                    problem, y, key, N_levels=10, N_langevin=30,
                    sigma_max=sm, sigma_min=0.01, lr_scale=0.5))
            if y.ndim == 1: return _jf[0](y, key)
            keys = jax.random.split(key, y.shape[0])
            return jnp.stack([_jf[0](y[i], keys[i]) for i in range(y.shape[0])])
        return solver

    print("=== MMPS-MALA vs baseline MALA at different σ_max ===\n")
    for smax in [0.1, 0.3, 0.5, 1.0]:
        print(f"σ_max = {smax}:")
        for label, solver in [
            ("  MALA", _make_base(smax)),
            ("  MMPS-MALA", make_mmps_mala_sal(sigma_max=smax)),
        ]:
            t0 = time.time()
            r = latent_calibration_test(problem, solver, key, n=n_cal)
            t1 = time.time()
            print(f"{label}: hpd={r['hpd_mean']:.3f}, KS={r['hpd_ks']:.3f} ({t1-t0:.0f}s)")
        print()
