"""The three target centerline shapes.

All targets use the normalized position along the beam

    xi = (x - xmin) / (xmax - xmin),   0 <= xi <= 1,

and give the desired vertical displacement of the centerline. Every target
starts at 0 (xi = 0, the clamp) and ends at -A (xi = 1, the tip):

    linear : y = -A * xi
    cosine : y = -(A/2) * (1 - cos(pi * xi))
    sigmoid: s(xi) = 1 / (1 + exp(-beta * (xi - 0.5)))
             y = -A * (s(xi) - s(0)) / (s(1) - s(0))

The functions accept NumPy or JAX arrays and return JAX arrays.

    from target_profiles import get_target
    y = get_target("cosine", xi, A=0.015)
"""
import jax.numpy as jnp

import _paths  # noqa: F401  (double precision)


def linear_target(xi, A, beta=None):
    return -A * xi


def cosine_target(xi, A, beta=None):
    return -(A / 2.0) * (1.0 - jnp.cos(jnp.pi * xi))


def sigmoid_target(xi, A, beta=8.0):
    s = lambda u: 1.0 / (1.0 + jnp.exp(-beta * (u - 0.5)))
    return -A * (s(xi) - s(0.0)) / (s(1.0) - s(0.0))


TARGETS = {
    "linear": linear_target,
    "cosine": cosine_target,
    "sigmoid": sigmoid_target,
}


def get_target(name, xi, A, beta=8.0):
    """Evaluate target `name` ("linear", "cosine" or "sigmoid") at positions xi."""
    if name not in TARGETS:
        raise ValueError(f"unknown target {name!r}; choose from {list(TARGETS)}")
    return TARGETS[name](jnp.asarray(xi), A, beta)
