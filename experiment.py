"""NSPS experiment: train flow, verify correctness, run calibration test."""

import time
import jax
import jax.numpy as jnp
import numpy as np
from lip import MNISTVAE
from lip.metrics import latent_calibration_test
from lip.solvers.nsps import (
    _init_flow, _flow_forward, _flow_inverse, _flow_log_prob, train_flow, nsps,
)

problem = MNISTVAE(sigma_n=0.4)
key = jax.random.PRNGKey(42)

# --- 1. Train the flow ---
print("=== Training RealNVP flow ===")
key, k_flow = jax.random.split(key)
t0 = time.time()
flow_params = train_flow(k_flow, n_epochs=200)
print(f"Training time: {time.time() - t0:.1f}s")

# --- 2. Verify flow correctness ---
print("\n=== Flow verification ===")

# Invertibility: flow_inverse(flow_forward(z)) ≈ z
key, k_test = jax.random.split(key)
z_test = jax.random.normal(k_test, (10, 2))
max_err = 0.0
for i in range(10):
    eps, ld = _flow_forward(flow_params, z_test[i])
    z_rec = _flow_inverse(flow_params, eps)
    err = float(jnp.max(jnp.abs(z_test[i] - z_rec)))
    max_err = max(max_err, err)
print(f"Invertibility max error: {max_err:.2e} (should be < 1e-6)")

# Log-det vs numerical Jacobian
fwd_single = lambda z: _flow_forward(flow_params, z)[0]
for i in range(3):
    _, ld_analytic = _flow_forward(flow_params, z_test[i])
    J = jax.jacfwd(fwd_single)(z_test[i])
    ld_numerical = jnp.log(jnp.abs(jnp.linalg.det(J)))
    print(f"  log_det analytic={float(ld_analytic):.4f}, "
          f"numerical={float(ld_numerical):.4f}, "
          f"diff={float(jnp.abs(ld_analytic - ld_numerical)):.2e}")

# Flow quality: check that samples from the flow look Gaussian
key, k_sample = jax.random.split(key)
eps_samples = jax.random.normal(k_sample, (5000, 2))
z_flow = jax.vmap(lambda e: _flow_inverse(flow_params, e))(eps_samples)
print(f"\nFlow sample stats (should be ≈ N(0,1)):")
print(f"  mean: {np.array(jnp.mean(z_flow, axis=0))}")
print(f"  std:  {np.array(jnp.std(z_flow, axis=0))}")

# --- 3. Calibration test ---
print("\n=== Calibration test (n=200) ===")

key, k_nsps, k_mala = jax.random.split(key, 3)

# NSPS
t0 = time.time()
r_nsps = latent_calibration_test(
    problem, nsps, k_nsps, n=200, flow_params=flow_params)
t_nsps = time.time() - t0
print(f"NSPS:     hpd={r_nsps['hpd_mean']:.3f}, KS={r_nsps['hpd_ks']:.3f} ({t_nsps:.0f}s)")

# MALA-SAL baseline
from lip.solvers import mala_sal
t0 = time.time()
r_mala = latent_calibration_test(problem, mala_sal, k_mala, n=200)
t_mala = time.time() - t0
print(f"MALA-SAL: hpd={r_mala['hpd_mean']:.3f}, KS={r_mala['hpd_ks']:.3f} ({t_mala:.0f}s)")

# --- 4. Summary ---
print(f"\n{'Method':<12} {'HPD mean':>9} {'KS stat':>9} {'Time':>8}")
print("-" * 42)
print(f"{'NSPS':<12} {r_nsps['hpd_mean']:9.3f} {r_nsps['hpd_ks']:9.3f} {t_nsps:7.0f}s")
print(f"{'MALA-SAL':<12} {r_mala['hpd_mean']:9.3f} {r_mala['hpd_ks']:9.3f} {t_mala:7.0f}s")
print(f"{'(calibrated)':<12} {'0.500':>9} {'→ 0':>9}")
