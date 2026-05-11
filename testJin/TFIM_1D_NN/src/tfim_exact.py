from functools import partial

import jax
import jax.numpy as jnp


EPS = 1e-9


def dispersion(k, h):
    arg = 1.0 + h * h - 2.0 * h * jnp.cos(k) + EPS
    return 2.0 * jnp.sqrt(arg)


@partial(jax.jit, static_argnames=("num_k",))
def thermodynamic_free_energy(T, h, num_k=4096):
    k = jnp.linspace(0.0, jnp.pi, num_k)
    eps_k = dispersion(k, h)
    x = eps_k / (2.0 * T)
    return -T * jnp.mean(jnp.logaddexp(x, -x))


@partial(jax.jit, static_argnames=("L",))
def finite_free_energy(T, h, L):
    """Finite-L free-energy density from the TFIM free-fermion spectrum.

    The memo calls for ED labels. For L=14,16 full finite-temperature ED is
    expensive because it needs the full 2**L spectrum, so this uses the exact
    NN TFIM fermionic spectrum with L discrete momenta.
    """
    k = (jnp.arange(L) + 0.5) * jnp.pi / L
    eps_k = dispersion(k, h)
    x = eps_k / (2.0 * T)
    return -T * jnp.mean(jnp.logaddexp(x, -x))


def _finite_scalar(T, h, L):
    return finite_free_energy(T, h, L)


def _thermo_scalar(T, h):
    return thermodynamic_free_energy(T, h)


def make_observable_fns(f_scalar, static_L=None):
    if static_L is None:
        scalar = lambda T, h: f_scalar(T, h)
    else:
        scalar = lambda T, h: f_scalar(T, h, static_L)

    d2T = jax.grad(jax.grad(scalar, argnums=0), argnums=0)
    d2h = jax.grad(jax.grad(scalar, argnums=1), argnums=1)

    def cv(T, h):
        return -T * d2T(T, h)

    def chi(T, h):
        return -d2h(T, h)

    return jax.jit(scalar), jax.jit(cv), jax.jit(chi)


def vectorized_finite_observables(L):
    f, cv, chi = make_observable_fns(_finite_scalar, static_L=L)
    return (
        jax.jit(jax.vmap(f, in_axes=(0, 0))),
        jax.jit(jax.vmap(cv, in_axes=(0, 0))),
        jax.jit(jax.vmap(chi, in_axes=(0, 0))),
    )


thermo_f, thermo_cv, thermo_chi = make_observable_fns(_thermo_scalar)
v_thermo_f = jax.jit(jax.vmap(thermo_f, in_axes=(0, 0)))
v_thermo_cv = jax.jit(jax.vmap(thermo_cv, in_axes=(0, 0)))
v_thermo_chi = jax.jit(jax.vmap(thermo_chi, in_axes=(0, 0)))
