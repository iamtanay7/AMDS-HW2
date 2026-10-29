"""Model layout for the JAX mass-spring-damper simulator.

Mirrors msd_prepare.m. The topology/boundary data (``Model``) is kept separate
from the differentiable design variables (``params``), so that jax.grad only
sees what we want it to see:

    model  : Model NamedTuple of fixed arrays (nodes, edges, L0, mass, BCs, loads)
    params : dict pytree of per-spring design variables
             {"k": (M,), "c": (M,), "a": (M,)}   a = activation in [0, 1]

Edges are stored 0-based. Use ``one_based=True`` when passing MATLAB edges.
"""

from math import ceil, floor, sqrt
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp


class Model(NamedTuple):
    X0: jnp.ndarray        # (N, D) initial node coordinates
    edge_i: jnp.ndarray    # (M,)   int32, first node of each spring
    edge_j: jnp.ndarray    # (M,)   int32, second node of each spring
    L0: jnp.ndarray        # (M,)   rest lengths
    mass: jnp.ndarray      # (N, 1) nodal mass
    inv_mass: jnp.ndarray  # (N, 1)
    fixed: jnp.ndarray     # (N, D) bool mask of fixed DOFs
    F_const: jnp.ndarray   # (N, D) force + mass * gravity
    tiny: jnp.ndarray      # ()     length guard, max(1e-12, 1e-12*mean(L0))

    @property
    def N(self):
        return self.X0.shape[0]

    @property
    def D(self):
        return self.X0.shape[1]

    @property
    def M(self):
        return self.edge_i.shape[0]


def prepare(nodes, edges, k, c, mass, L0=None, fixed=None, force=None,
            gravity=None, a=None, one_based=False, dtype=jnp.float64):
    """Validate inputs and build (model, params). Equivalent to msd_prepare.m."""
    X0 = np.asarray(nodes, dtype=np.float64)
    N, D = X0.shape
    assert D in (2, 3), "Only 2-D and 3-D systems are supported."

    edges = np.asarray(edges, dtype=np.int64)
    if one_based:
        edges = edges - 1
    assert edges.ndim == 2 and edges.shape[1] == 2, "edges must be M-by-2."
    assert edges.min() >= 0 and edges.max() < N, "Invalid node index in edges."
    M = edges.shape[0]

    k = np.broadcast_to(np.asarray(k, np.float64).ravel(), (M,))
    c = np.broadcast_to(np.asarray(c, np.float64).ravel(), (M,))
    a = np.ones(M) if a is None else np.broadcast_to(np.asarray(a, np.float64).ravel(), (M,))
    mass = np.broadcast_to(np.asarray(mass, np.float64).ravel(), (N,))
    assert np.all(mass > 0), "All nodal masses must be positive."

    fixed = np.zeros((N, D), bool) if fixed is None else np.asarray(fixed, bool)
    force = np.zeros((N, D)) if force is None else np.asarray(force, np.float64)
    gravity = np.zeros(D) if gravity is None else np.asarray(gravity, np.float64).ravel()
    assert fixed.shape == (N, D), "fixed must be N-by-D."
    assert force.shape == (N, D), "force must be N-by-D."
    assert gravity.size == D, "gravity must have D entries."

    i, j = edges[:, 0], edges[:, 1]
    lengths = np.linalg.norm(X0[j] - X0[i], axis=1)
    assert np.all(lengths > 0), "Zero-length edges are not allowed."
    if L0 is None:
        L0 = lengths
    else:
        L0 = np.broadcast_to(np.asarray(L0, np.float64).ravel(), (M,))
        assert np.all(L0 > 0), "All rest lengths must be positive."

    model = Model(
        X0=jnp.asarray(X0, dtype),
        edge_i=jnp.asarray(i, jnp.int32),
        edge_j=jnp.asarray(j, jnp.int32),
        L0=jnp.asarray(L0, dtype),
        mass=jnp.asarray(mass[:, None], dtype),
        inv_mass=jnp.asarray(1.0 / mass[:, None], dtype),
        fixed=jnp.asarray(fixed),
        F_const=jnp.asarray(force + mass[:, None] * gravity[None, :], dtype),
        tiny=jnp.asarray(max(1e-12, 1e-12 * float(np.mean(L0))), dtype),
    )
    params = {
        "k": jnp.asarray(k, dtype),
        "c": jnp.asarray(c, dtype),
        "a": jnp.asarray(a, dtype),
    }
    return model, params


def make_grid2d(nx, ny, k, c, mass_value, dtype=jnp.float64):
    """Triangulated rectangular grid, identical to msd_make_grid2d.m.

    Node and edge ordering follow MATLAB's column-major meshgrid/reshape, so
    node n here is node n+1 in MATLAB. Left edge fixed, total downward load of
    1 spread over the right edge.
    """
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    nodes = np.column_stack([xx.ravel(order="F"), yy.ravel(order="F")]).astype(float)
    node_id = np.arange(nx * ny).reshape(nx, ny).T      # (ny, nx), MATLAB id - 1

    def pairs(a, b):
        return np.column_stack([a.ravel(order="F"), b.ravel(order="F")])

    edges = np.vstack([
        pairs(node_id[:, :-1], node_id[:, 1:]),           # horizontal
        pairs(node_id[:-1, :], node_id[1:, :]),           # vertical
        pairs(node_id[:-1, :-1], node_id[1:, 1:]),        # diagonal
        pairs(node_id[1:, :-1], node_id[:-1, 1:]),        # anti-diagonal
    ])

    fixed = np.zeros(nodes.shape, bool)
    fixed[nodes[:, 0] == 0, :] = True
    force = np.zeros(nodes.shape)
    right = nodes[:, 0] == nx - 1
    force[right, 1] = -1.0 / max(1, right.sum())

    return prepare(nodes, edges, k, c, mass_value, fixed=fixed, force=force,
                   gravity=[0.0, 0.0], dtype=dtype)


def make_grid3d(nx, ny, nz, k, c, mass_value, dtype=jnp.float64):
    """Braced 3-D lattice cantilever (no MATLAB counterpart yet).

    Every node connects to its 13 forward neighbours in {-1,0,1}^3 (axis,
    face-diagonal and body-diagonal springs). The x=0 face is fixed and a total
    load of 1 in -z is spread over the x=nx-1 face.
    """
    g = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij"), axis=-1)
    nodes = g.reshape(-1, 3).astype(float)
    node_id = np.arange(nx * ny * nz).reshape(nx, ny, nz)

    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]
    edges = []
    for dx, dy, dz in offsets:
        src = node_id[max(0, -dx):nx - max(0, dx),
                      max(0, -dy):ny - max(0, dy),
                      max(0, -dz):nz - max(0, dz)]
        dst = node_id[max(0, dx):nx + min(0, dx) or None,
                      max(0, dy):ny + min(0, dy) or None,
                      max(0, dz):nz + min(0, dz) or None]
        edges.append(np.column_stack([src.ravel(), dst.ravel()]))
    edges = np.vstack(edges)

    fixed = np.zeros(nodes.shape, bool)
    fixed[nodes[:, 0] == 0, :] = True
    force = np.zeros(nodes.shape)
    tip = nodes[:, 0] == nx - 1
    force[tip, 2] = -1.0 / max(1, tip.sum())

    return prepare(nodes, edges, k, c, mass_value, fixed=fixed, force=force,
                   gravity=[0.0, 0.0, 0.0], dtype=dtype)


def grid_size_for(target_nodes):
    """Same (nx, ny) sizing rule as benchmark_fast_msd.m."""
    ny = max(4, int(floor(sqrt(target_nodes / 4) + 0.5)))  # MATLAB round()
    nx = max(4, int(ceil(target_nodes / ny)))
    return nx, ny


def num_steps(tspan, dt):
    """MATLAB fast solver convention: nSteps = ceil(T/dt), dt = T/nSteps."""
    t0, tf = tspan
    n = int(ceil((tf - t0) / dt))
    return n, (tf - t0) / n
