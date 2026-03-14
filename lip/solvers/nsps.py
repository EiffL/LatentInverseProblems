"""NSPS: Noise-Space Posterior Sampling.

Train a RealNVP normalizing flow F on the latent prior N(0,I), then run HMC
in the flow's noise space ε where the prior is exactly N(0,I). The flow
provides exact log-density via the change-of-variables formula, giving an
unbiased MCMC sampler.

HMC target: p(ε|y) ∝ N(ε|0,I) · p(y|D(F⁻¹(ε)))

For the 2D MNISTVAE latent space, RealNVP with affine coupling layers suffices.
Both forward z→ε and inverse ε→z are exact and O(1) per layer.
"""

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Part A: RealNVP flow (pure functional, param-pytree style like lip/vae.py)
# ---------------------------------------------------------------------------

def _mlp_init(key, dims):
    """Initialize MLP parameters: list of (W, b) tuples."""
    params = []
    for i in range(len(dims) - 1):
        key, k = jax.random.split(key)
        fan_in, fan_out = dims[i], dims[i + 1]
        scale = jnp.sqrt(2.0 / fan_in)
        W = scale * jax.random.normal(k, (fan_in, fan_out))
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def _mlp_forward(params, x):
    """MLP forward: ReLU hidden layers, tanh*2 on final (bounded scale)."""
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return jnp.tanh(x @ W + b) * 2.0


def _init_flow(key, n_layers=8, hidden_dim=64):
    """Initialize RealNVP flow parameters.

    Each coupling layer has an MLP that maps 1D → 2D (scale, shift).
    Even layers: transform dim 0 conditioned on dim 1.
    Odd layers: transform dim 1 conditioned on dim 0.
    """
    layers = []
    for i in range(n_layers):
        key, k = jax.random.split(key)
        mlp_params = _mlp_init(k, [1, hidden_dim, hidden_dim, 2])
        layers.append(mlp_params)
    return layers


def _flow_forward(params, z):
    """Forward pass z → ε, accumulating log|det(dε/dz)|.

    Returns (ε, log_det).
    """
    log_det = 0.0
    for i, mlp_params in enumerate(params):
        if i % 2 == 0:
            # Transform dim 0 conditioned on dim 1
            st = _mlp_forward(mlp_params, z[1:2])
            s, t = st[0], st[1]
            z = jnp.array([z[0] * jnp.exp(s) + t, z[1]])
            log_det = log_det + s
        else:
            # Transform dim 1 conditioned on dim 0
            st = _mlp_forward(mlp_params, z[0:1])
            s, t = st[0], st[1]
            z = jnp.array([z[0], z[1] * jnp.exp(s) + t])
            log_det = log_det + s
    return z, log_det


def _flow_inverse(params, eps):
    """Inverse pass ε → z. Reverse coupling layers in reverse order."""
    z = eps
    for i in range(len(params) - 1, -1, -1):
        mlp_params = params[i]
        if i % 2 == 0:
            # Undo: z0_new = z0 * exp(s) + t => z0 = (z0_new - t) / exp(s)
            st = _mlp_forward(mlp_params, z[1:2])
            s, t = st[0], st[1]
            z = jnp.array([(z[0] - t) * jnp.exp(-s), z[1]])
        else:
            # Undo: z1_new = z1 * exp(s) + t => z1 = (z1_new - t) / exp(s)
            st = _mlp_forward(mlp_params, z[0:1])
            s, t = st[0], st[1]
            z = jnp.array([z[0], (z[1] - t) * jnp.exp(-s)])
    return z


def _flow_log_prob(params, z):
    """Log probability under the flow: log p_NF(z) = -½‖F(z)‖² + log_det."""
    eps, log_det = _flow_forward(params, z)
    return -0.5 * jnp.sum(eps**2) + log_det


# ---------------------------------------------------------------------------
# Part B: Flow training
# ---------------------------------------------------------------------------

def train_flow(key, *, n_samples=50000, n_epochs=200, lr=1e-3,
               batch_size=512, n_layers=8, hidden_dim=64):
    """Train RealNVP on N(0, I_2) samples. Returns flow_params."""
    key, k_init, k_data = jax.random.split(key, 3)
    flow_params = _init_flow(k_init, n_layers=n_layers, hidden_dim=hidden_dim)

    # Training data: samples from the prior N(0, I)
    data = jax.random.normal(k_data, (n_samples, 2))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(flow_params)

    def loss_fn(params, batch):
        log_probs = jax.vmap(lambda z: _flow_log_prob(params, z))(batch)
        return -jnp.mean(log_probs)

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    n_batches = n_samples // batch_size
    for epoch in range(n_epochs):
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, n_samples)
        for b in range(n_batches):
            batch = data[perm[b * batch_size:(b + 1) * batch_size]]
            flow_params, opt_state, loss = train_step(
                flow_params, opt_state, batch)

    return flow_params


# ---------------------------------------------------------------------------
# Part C: HMC in noise space
# ---------------------------------------------------------------------------

def _hmc_chain(key, eps_init, log_prob_fn, grad_fn, step_size, n_leapfrog,
               n_steps):
    """Run HMC chain via lax.scan. Returns final ε."""

    def hmc_step(carry, _):
        eps, key = carry
        key, k_mom, k_accept = jax.random.split(key, 3)

        # Sample momentum
        p = jax.random.normal(k_mom, eps.shape)
        log_p_old = log_prob_fn(eps)
        kinetic_old = 0.5 * jnp.sum(p**2)

        # Leapfrog integration
        def leapfrog_step(eps_p, _):
            e, p = eps_p
            g = grad_fn(e)
            p = p + (step_size / 2) * g
            e = e + step_size * p
            g = grad_fn(e)
            p = p + (step_size / 2) * g
            return (e, p), None

        (eps_new, p_new), _ = jax.lax.scan(
            leapfrog_step, (eps, p), None, length=n_leapfrog)

        # MH accept/reject
        log_p_new = log_prob_fn(eps_new)
        kinetic_new = 0.5 * jnp.sum(p_new**2)
        log_alpha = (log_p_new - kinetic_new) - (log_p_old - kinetic_old)
        accept = jnp.log(jax.random.uniform(k_accept)) < log_alpha
        eps_out = jnp.where(accept, eps_new, eps)

        return (eps_out, key), None

    (eps_final, _), _ = jax.lax.scan(hmc_step, (eps_init, key), None,
                                      length=n_steps)
    return eps_final


# ---------------------------------------------------------------------------
# Part D: Solver interface
# ---------------------------------------------------------------------------

def _nsps_single(problem, y, key, *, flow_params, n_leapfrog=20,
                 step_size=0.01, n_warmup=300, n_steps=500):
    """NSPS for a single observation."""
    sn2 = problem.sigma_n**2

    def log_posterior_eps(eps):
        z = _flow_inverse(flow_params, eps)
        x_hat = problem.decoder(z[None])[0]
        log_prior_eps = -0.5 * jnp.sum(eps**2)
        log_lik = -0.5 * jnp.sum((y - x_hat)**2) / sn2
        return log_prior_eps + log_lik

    grad_log_post = jax.grad(log_posterior_eps)

    # Initialize: encode y, map to noise space
    z_init = problem.encoder(y)
    eps_init, _ = _flow_forward(flow_params, z_init)

    # Warmup + sampling
    key, k_warmup, k_sample = jax.random.split(key, 3)
    eps = _hmc_chain(k_warmup, eps_init, log_posterior_eps, grad_log_post,
                     step_size, n_leapfrog, n_warmup)
    eps = _hmc_chain(k_sample, eps, log_posterior_eps, grad_log_post,
                     step_size, n_leapfrog, n_steps)

    return _flow_inverse(flow_params, eps)


_jit_cache = None
_jit_config = None
_flow_cache = None


def nsps(problem, y, key, *, flow_params=None, n_leapfrog=20, step_size=0.01,
         n_warmup=300, n_steps=500, flow_n_layers=8, flow_hidden_dim=64,
         flow_n_epochs=200, **kwargs):
    """NSPS posterior sampler.

    Args:
        problem: MNISTVAE problem instance.
        y: Observation(s), shape (d_pixel,) or (n, d_pixel).
        key: JAX PRNG key.
        flow_params: Pre-trained flow parameters (trained on first call if None).
        n_leapfrog: Leapfrog steps per HMC proposal.
        step_size: HMC step size.
        n_warmup: HMC warmup steps (discarded).
        n_steps: HMC sampling steps (return final sample).
        flow_n_layers: Number of RealNVP coupling layers.
        flow_hidden_dim: Hidden dimension of coupling MLPs.
        flow_n_epochs: Training epochs for the flow.

    Returns:
        z: Posterior sample(s), shape (d_latent,) or (n, d_latent).
    """
    global _jit_cache, _jit_config, _flow_cache

    # Train flow on first call (or reuse cached)
    if flow_params is None:
        flow_key = (id(problem), flow_n_layers, flow_hidden_dim, flow_n_epochs)
        if _flow_cache is None or _flow_cache[0] != flow_key:
            key, k_flow = jax.random.split(key)
            print("Training RealNVP flow on prior samples...")
            flow_params = train_flow(
                k_flow, n_layers=flow_n_layers, hidden_dim=flow_hidden_dim,
                n_epochs=flow_n_epochs)
            _flow_cache = (flow_key, flow_params)
            print("Flow training complete.")
        else:
            flow_params = _flow_cache[1]

    config = (id(problem), id(flow_params[0][0][0]),
              n_leapfrog, step_size, n_warmup, n_steps)
    if _jit_cache is None or _jit_config != config:
        _jit_cache = jax.jit(
            lambda y, key: _nsps_single(
                problem, y, key, flow_params=flow_params,
                n_leapfrog=n_leapfrog, step_size=step_size,
                n_warmup=n_warmup, n_steps=n_steps)
        )
        _jit_config = config

    if y.ndim == 1:
        return _jit_cache(y, key)

    keys = jax.random.split(key, y.shape[0])
    results = []
    for i in range(y.shape[0]):
        results.append(_jit_cache(y[i], keys[i]))
    return jnp.stack(results)
