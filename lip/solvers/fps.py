"""Filtering Posterior Sampling (FPS) -- SMC for latent inverse problems.

Adapts FPS (Dou & Song, ICLR 2024) for nonlinear latent inverse problems.
The original FPS uses VP-SDE with a "duplex" coupled measurement sequence for
linear problems (y = Ax + n). For nonlinear decoders, the duplex construction
is not possible (Remark 4.1), so we adapt:

  - VE-SDE reverse process with exact Gaussian prior score
  - Tweedie-based approximate likelihood with Jacobian correction (Woodbury)
  - Incremental importance weights with systematic resampling

Two variants:
  - fps_spf: Bootstrap particle filter (unconditional proposal + likelihood
    weights). Simpler, avoids linearization bias for nonlinear decoders.
  - fps_smc: Tailored proposal incorporating linearized measurement (Prop. B.3
    analog). Lower variance for linear problems but may introduce bias for
    nonlinear decoders.

Reference:
  Dou & Song, "Diffusion Posterior Sampling for Linear Inverse Problem
  Solving: A Filtering Perspective", ICLR 2024.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _systematic_resample(key, log_weights, K):
    """Systematic resampling from log-weights."""
    log_w = log_weights - jax.nn.logsumexp(log_weights)
    w = jnp.exp(log_w)
    cumsum = jnp.cumsum(w)
    u = (jax.random.uniform(key) + jnp.arange(K)) / K
    return jnp.clip(jnp.searchsorted(cumsum, u), 0, K - 1)


def _log_measurement_likelihood(problem, z, y, sigma):
    """Log p~(y | z_sigma) via Tweedie denoiser + Woodbury identity.

    y | z_sigma ~ N(D(z_hat0), sn2 I + c J J^T)
    where z_hat0 = E[z0|z_sigma], c = Var[z0|z_sigma], J = dD/dz|_{z_hat0}.
    Exploits d_latent << d_pixel via the matrix determinant lemma.
    """
    K, d = z.shape
    s02 = problem.sigma_0**2
    sn2 = problem.sigma_n**2

    alpha = s02 / (s02 + sigma**2)
    c = sigma**2 * s02 / (s02 + sigma**2)

    z_hat = alpha * z
    x_hat = problem.decoder(z_hat)
    r = y[None, :] - x_hat

    if c < 1e-12:
        return -0.5 * jnp.sum(r**2, axis=-1) / sn2

    J = problem.decoder_jacobian(z_hat)
    G = jnp.einsum('kpd,kpe->kde', J, J)       # J^T J, (K, d, d)
    J_r = jnp.einsum('kpd,kp->kd', J, r)       # J^T r, (K, d)

    I_d = jnp.eye(d)
    M = sn2 * I_d[None] + c * G                 # (K, d, d)
    M_inv = jnp.linalg.inv(M)
    M_inv_J_r = jnp.einsum('kde,ke->kd', M_inv, J_r)

    # Woodbury: r^T Sigma_y^{-1} r
    mahal = (jnp.sum(r**2, axis=-1)
             - c * jnp.sum(J_r * M_inv_J_r, axis=-1)) / sn2
    # log|Sigma_y| = d_pixel log(sn2) + log|I_d + (c/sn2) G|
    _, log_det_inner = jnp.linalg.slogdet(I_d[None] + (c / sn2) * G)
    log_det = problem.d_pixel * jnp.log(sn2) + log_det_inner

    return -0.5 * (mahal + log_det)


# ---------------------------------------------------------------------------
# SPF variant: unconditional proposal + likelihood reweighting
# ---------------------------------------------------------------------------

def _spf_single(problem, y, key, *, N, K, sigma_max, sigma_min):
    """Bootstrap particle filter (SPF) for a single observation.

    Unconditional reverse VE-SDE as proposal, incremental Tweedie-based
    likelihood weights, systematic resampling at every step.
    """
    sigmas = jnp.geomspace(sigma_max, sigma_min, N + 1)
    d = problem.d_latent
    s02 = problem.sigma_0**2

    key, k_init = jax.random.split(key)
    particles = jnp.sqrt(s02 + sigma_max**2) * jax.random.normal(k_init, (K, d))

    log_lik = _log_measurement_likelihood(problem, particles, y, float(sigmas[0]))

    for n in range(N):
        s_curr = float(sigmas[n])
        s_next = float(sigmas[n + 1])
        ds2 = s_curr**2 - s_next**2

        # Unconditional reverse VE-SDE
        scores = problem.score(particles, s_curr)
        key, k_noise = jax.random.split(key)
        particles = (particles + ds2 * scores
                     + jnp.sqrt(ds2) * jax.random.normal(k_noise, (K, d)))

        # Incremental likelihood weights
        log_lik_new = _log_measurement_likelihood(problem, particles, y, s_next)
        log_w = log_lik_new - log_lik

        # Resample
        key, k_res = jax.random.split(key)
        indices = _systematic_resample(k_res, log_w, K)
        particles = particles[indices]
        log_lik = log_lik_new[indices]

    key, k_sel = jax.random.split(key)
    return particles[jax.random.choice(k_sel, K)]


# ---------------------------------------------------------------------------
# FPS-SMC variant: tailored proposal with linearized decoder
# ---------------------------------------------------------------------------

def _smc_single(problem, y, key, *, N, K, sigma_max, sigma_min):
    """FPS-SMC with tailored proposal for a single observation.

    Linearizes the decoder around the Tweedie estimate to build a
    Gaussian tailored proposal (Proposition B.3 analog). Importance
    weights use the marginal likelihood at the parent particle.

    For linear problems this recovers the exact FPS-SMC of Dou & Song.
    For nonlinear decoders, linearization introduces approximation error.
    """
    sigmas = jnp.geomspace(sigma_max, sigma_min, N + 1)
    d = problem.d_latent
    s02 = problem.sigma_0**2
    sn2 = problem.sigma_n**2
    I_d = jnp.eye(d)

    key, k_init = jax.random.split(key)
    particles = jnp.sqrt(s02 + sigma_max**2) * jax.random.normal(k_init, (K, d))

    for n in range(N):
        s_curr = float(sigmas[n])
        s_next = float(sigmas[n + 1])
        ds2 = s_curr**2 - s_next**2

        # Unconditional reverse step parameters
        scores = problem.score(particles, s_curr)
        mu_rev = particles + ds2 * scores

        # Tweedie at destination
        alpha = s02 / (s02 + s_next**2)
        c = s_next**2 * s02 / (s02 + s_next**2)
        c_total = c + alpha**2 * ds2

        # Linearize decoder at predicted Tweedie estimate
        z_hat0 = alpha * mu_rev
        x_hat0 = problem.decoder(z_hat0)
        J0 = problem.decoder_jacobian(z_hat0)
        r = y[None, :] - x_hat0

        G = jnp.einsum('kpd,kpe->kde', J0, J0)
        J0_r = jnp.einsum('kpd,kp->kd', J0, r)

        # Tailored proposal: combine reverse step + linearized measurement
        # Sigma_y^{-1} J_eff = alpha J0 M^{-1} (Woodbury)
        M = sn2 * I_d[None] + c * G
        M_inv = jnp.linalg.inv(M)

        Lambda_y = alpha**2 * jnp.einsum('kde,kef->kdf', G, M_inv)
        Lambda_fps = (1.0 / ds2) * I_d[None] + Lambda_y
        Sigma_fps = jnp.linalg.inv(Lambda_fps)

        eta_y = alpha * jnp.einsum(
            'kde,ke->kd', M_inv,
            J0_r + alpha * jnp.einsum('kde,ke->kd', G, mu_rev))
        mu_fps = jnp.einsum('kde,ke->kd', Sigma_fps, mu_rev / ds2 + eta_y)

        # Sample from tailored proposal
        key, k_sample = jax.random.split(key)
        L = jnp.linalg.cholesky(Sigma_fps + 1e-10 * I_d[None])
        eps = jax.random.normal(k_sample, (K, d))
        new_particles = mu_fps + jnp.einsum('kde,ke->kd', L, eps)

        # Resampling weight: marginal likelihood at parent p(y | z_parent)
        # = N(y | D(z_hat0), sn2 I + c_total J0 J0^T)
        M_total = sn2 * I_d[None] + c_total * G
        M_total_inv = jnp.linalg.inv(M_total)
        M_total_inv_J0_r = jnp.einsum('kde,ke->kd', M_total_inv, J0_r)

        mahal = (jnp.sum(r**2, axis=-1)
                 - c_total * jnp.sum(J0_r * M_total_inv_J0_r, axis=-1)) / sn2
        _, log_det_inner = jnp.linalg.slogdet(
            I_d[None] + (c_total / sn2) * G)
        log_w = -0.5 * (mahal
                        + problem.d_pixel * jnp.log(sn2) + log_det_inner)

        # Resample
        key, k_res = jax.random.split(key)
        indices = _systematic_resample(k_res, log_w, K)
        particles = new_particles[indices]

    key, k_sel = jax.random.split(key)
    return particles[jax.random.choice(k_sel, K)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _dispatch(fn_single, problem, y, key, **kw):
    if y.ndim == 1:
        return fn_single(problem, y, key, **kw)
    keys = jax.random.split(key, y.shape[0])
    return jnp.stack([fn_single(problem, y[i], keys[i], **kw)
                      for i in range(y.shape[0])])


def fps_spf(problem, y, key, *, N=200, K=128, sigma_max=2.0, sigma_min=0.01,
            **kwargs):
    """FPS with bootstrap particle filter (SPF variant).

    Unconditional reverse VE-SDE as proposal, Tweedie-based incremental
    likelihood weights with Jacobian correction, systematic resampling.
    """
    return _dispatch(_spf_single, problem, y, key,
                     N=N, K=K, sigma_max=sigma_max, sigma_min=sigma_min)


def fps_smc(problem, y, key, *, N=200, K=64, sigma_max=2.0, sigma_min=0.01,
            **kwargs):
    """FPS-SMC with tailored proposal (linearized decoder).

    Tailored proposal incorporating linearized measurement model
    (Proposition B.3 analog), marginal likelihood resampling weights.
    Optimal for linear problems; may under-disperse for nonlinear decoders.
    """
    return _dispatch(_smc_single, problem, y, key,
                     N=N, K=K, sigma_max=sigma_max, sigma_min=sigma_min)
