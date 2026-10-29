"""Per-step physics, mirroring msd_simulate_fast_double.m.

For edge e = (i, j):
    r    = x_j - x_i              (MATLAB: R = B*X)
    u    = v_j - v_i              (MATLAB: U = B*V)
    L    = |r|,  n = r / L
    beta = u . n
    s    = a * [ spring_law(L - L0) + damper_law(beta) ]
    f_i  = +s n,  f_j = -s n      (MATLAB: Fnode = -B'*Fedge)

Geometry is fully nonlinear (L and n recomputed every step). The constitutive
laws are pluggable: each is a function ``law(x, params) -> scalar per edge``
that may read any per-spring entry of ``params``. The defaults reproduce the
MATLAB linear law exactly. ``a`` is a per-spring activation (1 = intact,
0 = removed) for differentiable fracture / topology studies.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Constitutive laws: (stretch or beta, params) -> axial force per edge
# ---------------------------------------------------------------------------

def linear_spring(stretch, params):
    return params["k"] * stretch


def linear_damper(beta, params):
    return params["c"] * beta


def cubic_spring(stretch, params):
    """Example nonlinear law: k*s + k3*s^3 (needs params["k3"])."""
    return params["k"] * stretch + params["k3"] * stretch ** 3


# ---------------------------------------------------------------------------
# Forces and time step
# ---------------------------------------------------------------------------

def edge_forces(X, V, model, params, spring_law=linear_spring,
                damper_law=linear_damper):
    """Return per-edge force vectors Fedge (M, D), applied +Fedge to node i."""
    i, j = model.edge_i, model.edge_j
    R = X[j] - X[i]
    U = V[j] - V[i]

    # sqrt(max(|r|^2, tiny^2)) == max(|r|, tiny) but with a finite gradient at
    # r = 0. MATLAB uses the unclamped L in (L - L0); this only differs when
    # |r| < tiny (~1e-12), i.e. a collapsed spring.
    L = jnp.sqrt(jnp.maximum(jnp.sum(R * R, axis=1), model.tiny ** 2))
    n = R / L[:, None]
    beta = jnp.sum(U * n, axis=1)

    s = spring_law(L - model.L0, params) + damper_law(beta, params)
    if "a" in params:
        s = params["a"] * s
    return n * s[:, None]


def nodal_forces(Fedge, model):
    """Fnode = -B' * Fedge + F_const, via scatter-add (no explicit B)."""
    N = model.N
    F = jax.ops.segment_sum(Fedge, model.edge_i, num_segments=N)
    F = F - jax.ops.segment_sum(Fedge, model.edge_j, num_segments=N)
    return F + model.F_const


def accelerations(X, V, model, params, spring_law=linear_spring,
                  damper_law=linear_damper):
    Fedge = edge_forces(X, V, model, params, spring_law, damper_law)
    A = nodal_forces(Fedge, model) * model.inv_mass
    return jnp.where(model.fixed, 0.0, A)


def semi_implicit_euler_step(state, model, params, dt, spring_law=linear_spring,
                             damper_law=linear_damper):
    """One step of exactly the MATLAB fast-solver update:

        A(fixed) = 0;  V = V + dt*A;  V(fixed) = 0;
        X = X + dt*V;  X(fixed) = Xfixed(fixed)
    """
    X, V = state
    A = accelerations(X, V, model, params, spring_law, damper_law)
    V = jnp.where(model.fixed, 0.0, V + dt * A)
    X = jnp.where(model.fixed, model.X0, X + dt * V)
    return X, V


# ---------------------------------------------------------------------------
# Stability helper
# ---------------------------------------------------------------------------

def stable_dt_estimate(model, params, X=None, iters=60, spring_law=linear_spring,
                       seed=0):
    """Undamped explicit-stability estimate dt_crit ~ 2/sqrt(lambda_max).

    lambda_max of M^-1 K (tangent stiffness at X, default X0) is found by power
    iteration using jax.jvp, so it works for any spring_law. Damping lowers the
    true limit somewhat, so use a safety factor (e.g. 0.8-0.9).
    """
    X = model.X0 if X is None else X
    zero_V = jnp.zeros_like(X)
    no_damp = lambda beta, p: jnp.zeros_like(beta)

    def acc(Xq):
        return accelerations(Xq, zero_V, model, params, spring_law, no_damp)

    def matvec(w):
        return -jax.jvp(acc, (X,), (w,))[1]

    w = jax.random.normal(jax.random.PRNGKey(seed), X.shape, X.dtype)
    w = jnp.where(model.fixed, 0.0, w)
    lam = jnp.asarray(0.0, X.dtype)
    for _ in range(iters):
        Kw = matvec(w)
        lam = jnp.vdot(w, Kw) / jnp.vdot(w, w)
        w = Kw / jnp.linalg.norm(Kw)
    return 2.0 / jnp.sqrt(lam), lam
