import pickle
from pathlib import Path

import jax
import jax.numpy as jnp


FEATURE_MIN = jnp.array([0.05, 0.1, 0.0])
FEATURE_MAX = jnp.array([1.0, 2.0, 1.0 / 6.0])
TARGET_MEAN = -1.35
TARGET_SCALE = 0.75


def normalize_x(x):
    return 2.0 * (x - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN) - 1.0


def init_mlp(key, widths=(3, 64, 64, 64, 1)):
    params = []
    keys = jax.random.split(key, len(widths) - 1)
    for k, fan_in, fan_out in zip(keys, widths[:-1], widths[1:]):
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        w = jax.random.uniform(k, (fan_in, fan_out), minval=-limit, maxval=limit)
        b = jnp.zeros((fan_out,))
        params.append({"w": w, "b": b})
    return params


def mlp_raw(params, x):
    z = normalize_x(x)
    for layer in params[:-1]:
        z = jax.nn.gelu(z @ layer["w"] + layer["b"])
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

    hess = jax.hessian(f_local)(x)
    F = f_local(x)
    Cv = -T * hess[0, 0]
    chi = -hess[1, 1]
    return jnp.array([F, Cv, chi])


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
