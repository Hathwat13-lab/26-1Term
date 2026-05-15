import pickle
from pathlib import Path

import jax
import jax.numpy as jnp


FEATURE_MIN = jnp.array([0.05, 0.1, 0.0])
FEATURE_MAX = jnp.array([1.0, 2.0, 1.0 / 6.0])
TARGET_MEAN = -1.35
TARGET_SCALE = 0.75
CRITICAL_H = 1.0


def normalize_x(x):
    return 2.0 * (x - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN) - 1.0


def critical_features(x):
    """Features capturing the exact non-analytic structure of the 1D TFIM free energy."""
    T = x[..., 0]
    h = x[..., 1]
    inv_L = x[..., 2]
    dh = h - CRITICAL_H
    
    thermal_rounding = 1.0 * T + 1.2 * inv_L + 1e-3
    r2 = dh * dh + thermal_rounding * thermal_rounding
    
    return jnp.stack(
        [
            dh,
            dh * dh,
            thermal_rounding,
            r2 * jnp.log(r2 + 1e-8),  # CORRECT FSS singular part for Free Energy
            jax.nn.softplus(5.0 * dh),
            jax.nn.softplus(-5.0 * dh),
        ],
        axis=-1,
    )


def features(x):
    return jnp.concatenate([normalize_x(x), critical_features(x)], axis=-1)


def init_mlp(key, widths=(9, 80, 80, 80, 1)):
    params = []
    keys = jax.random.split(key, len(widths) - 1)
    for k, fan_in, fan_out in zip(keys, widths[:-1], widths[1:]):
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        w = jax.random.uniform(k, (fan_in, fan_out), minval=-limit, maxval=limit)
        b = jnp.zeros((fan_out,))
        params.append({"w": w, "b": b})
    return params


def mlp_raw(params, x):
    z = features(x)
    for layer in params[:-1]:
        y = z @ layer["w"] + layer["b"]
        z = y * jax.nn.sigmoid(y)
    z = z @ params[-1]["w"] + params[-1]["b"]
    return z[..., 0]


def free_energy(params, x):
    return TARGET_MEAN + TARGET_SCALE * mlp_raw(params, x)


def free_energy_scalar(params, T, h, inv_L):
    x = jnp.array([T, h, inv_L])
    return free_energy(params, x)


def observables_scalar(params, x):
    T, h, inv_L = x

    def f_local(y):
        return free_energy_scalar(params, y[0], y[1], y[2])

    grad_f = jax.grad(f_local)(x)
    hess = jax.hessian(f_local)(x)
    F = f_local(x)
    M = -grad_f[1]
    Cv = -T * hess[0, 0]
    S = -grad_f[0]
    chi = -hess[1, 1]
    # Return order: [F, Cv, chi, M, S] to keep Cv and chi at indices 1 and 2 for 3_train.py
    return jnp.array([F, Cv, chi, M, S])


batched_free_energy = jax.jit(jax.vmap(free_energy, in_axes=(None, 0)))
batched_observables = jax.jit(jax.vmap(observables_scalar, in_axes=(None, 0)))


def save_params(path, params, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"params": params, "metadata": metadata}, f)


def load_params(path):
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    return payload["params"], payload.get("metadata", {})
