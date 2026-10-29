"""Force-based link fracture for the cantilever demo.

Failure criterion, checked for every intact link at every time step:

    |f_axial| > F_fail          f_axial = k * (L - L0)   (elastic link force)

Threshold options (config.fracture_threshold_mode):
    "fixed"     : F_fail = config.fracture_force for every link
    "stiffness" : F_fail = config.fracture_force * (k / k_ref) ** p,
                  with p = config.fracture_k_exponent and k_ref the median k.

Note: p = 1 turns the force criterion into a fixed-elongation criterion,
because |k (L - L0)| > c * k  <=>  |L - L0| > c, the same c for every link.
It is not a strain criterion: diagonals are sqrt(2) times longer than sides,
so they fail at 1/sqrt(2) times the strain. (A strain criterion would need
F_fail proportional to k * L0.) Use 0 <= p < 1 for a criterion that is
genuinely force-based.

A failed link gets activation a = 0. The core simulator scales both the
spring force and the damping force by a, so a failed link carries no force
at all. It stays in the arrays, so array sizes never change. Failure is
permanent.
"""
import jax
import jax.numpy as jnp

import _paths  # noqa: F401
from physics import edge_forces, nodal_forces


def failure_threshold(k, config):
    """Per-link failure force F_fail [N] for link stiffnesses k."""
    if config.fracture_threshold_mode == "fixed":
        return config.fracture_force * jnp.ones_like(k)
    if config.fracture_threshold_mode == "stiffness":
        k_ref = jnp.median(k)
        return config.fracture_force * (k / k_ref) ** config.fracture_k_exponent
    raise ValueError("fracture_threshold_mode must be 'fixed' or 'stiffness'")


def axial_spring_force(X, model, k):
    """Elastic axial force of every link, k * (L - L0). Positive = tension."""
    R = X[model.edge_j] - X[model.edge_i]
    L = jnp.sqrt(jnp.sum(R * R, axis=1))
    return k * (L - model.L0)


def simulate_fracture(beam, k, c, config, dt, n_steps, n_frames=200):
    """Ramp load F(t) = load_rate * t on beam.load_nodes and let links fail.

    Returns a dict of NumPy-friendly arrays:
        times     [n_frames]          time of each saved frame
        X         [n_frames x N x 2]  node positions
        load      [n_frames]          applied force magnitude
        n_failed  [n_frames]          number of failed links
        fail_time [M]                 time each link failed (inf if intact)
        threshold [M]                 failure force of each link
    """
    model = beam.model
    alpha = config.alpha_damping
    rate = config.fracture_load_rate
    threshold = failure_threshold(k, config)
    stride = max(1, n_steps // n_frames)

    def step(state, n):
        X, V, a, fail_time = state
        t = n * dt

        # 1. Check failure with the current geometry.
        f = axial_spring_force(X, model, k)
        stressed = (f > threshold) if config.fracture_tension_only else (jnp.abs(f) > threshold)
        breaks = stressed & (a > 0)
        a = jnp.where(breaks, 0.0, a)
        fail_time = jnp.where(breaks, t, fail_time)

        # 2. Advance one step; failed links (a = 0) carry no spring or damper force.
        F = nodal_forces(edge_forces(X, V, model, {"k": k, "c": c, "a": a}), model)
        F = F + rate * t * beam.load_nodes
        A = jnp.where(model.fixed, 0.0, F * model.inv_mass - alpha * V)
        V = jnp.where(model.fixed, 0.0, V + dt * A)
        X = jnp.where(model.fixed, model.X0, X + dt * V)
        return (X, V, a, fail_time)

    @jax.jit
    def run():
        def chunk(state, n0):
            state, _ = jax.lax.scan(lambda s, n: (step(s, n), None), state,
                                    n0 + jnp.arange(stride))
            X, V, a, fail_time = state
            return state, (X, jnp.sum(a == 0))

        X0 = model.X0
        state = (X0, jnp.zeros_like(X0), jnp.ones(beam.n_links), jnp.full(beam.n_links, jnp.inf))
        state, (X_frames, n_failed) = jax.lax.scan(chunk, state, stride * jnp.arange(n_steps // stride))
        return X_frames, n_failed, state[3]

    X_frames, n_failed, fail_time = run()
    times = dt * stride * (1 + jnp.arange(n_steps // stride))
    return {"times": times, "X": X_frames, "load": rate * times, "n_failed": n_failed,
            "fail_time": fail_time, "threshold": threshold}


def run_fracture_demo(config=None):
    """The cantilever fracture demo: build the beam, run it, summarize the failures.

    Uniform beam (config.fracture_n_cells_x x fracture_n_cells_y cells, modulus
    config.fracture_E), left edge clamped, F(t) = load_rate * t downward at the
    upper-right node. Returns a dict with
        beam, out                 the beam and the simulate_fracture output
        first_failure_time [s]    None if no link failed
        first_failure_load [N]    load_rate * first_failure_time
        n_failed                  number of failed links at the end
        first_links               indices of the link(s) that failed first
        last_failure_time [s]
        tip_deflection_before [m] upper-right node, last frame before failure
        before, after             frame indices: just before the first failure,
                                  and after fracture (free end dropped 0.3 L)
    Plot it with plotting.plot_fracture_demo(result).
    """
    import numpy as np
    from beam_setup import build_beam
    from config import CONFIG
    from materials import link_damping, material_to_link_stiffness
    from model import num_steps

    config = CONFIG if config is None else config
    beam = build_beam(config, config.fracture_n_cells_x, config.fracture_n_cells_y,
                      load="top_right")
    k = material_to_link_stiffness(np.full(beam.n_cells, config.fracture_E), beam)
    c = link_damping(k, config)
    n_steps, dt = num_steps((0.0, config.fracture_sim_time), config.dt)
    out = simulate_fracture(beam, k, c, config, dt, n_steps)

    times = np.asarray(out["times"])
    fail_time = np.asarray(out["fail_time"])
    X = np.asarray(out["X"])
    broken = np.isfinite(fail_time)
    tip = beam.model.X0.shape[0] - 1                    # upper-right node
    result = {"beam": beam, "out": out, "config": config, "first_failure_time": None,
              "first_failure_load": None, "n_failed": 0, "first_links": np.array([], int),
              "last_failure_time": None, "tip_deflection_before": None}
    if broken.any():
        t1 = fail_time[broken].min()
        before = max(0, np.searchsorted(times, t1) - 1)    # last frame before failure
        result.update(first_failure_time=t1, first_failure_load=config.fracture_load_rate * t1,
                      n_failed=int(broken.sum()),
                      tip_deflection_before=X[before, tip, 1] - beam.nodes[tip, 1],
                      first_links=np.where(fail_time == t1)[0],
                      last_failure_time=fail_time[broken].max())
    else:
        before = len(times) - 1
    # "After fracture" frame: once the free end has dropped 30% of the beam
    # length (after that the detached piece just keeps moving), else the last frame.
    drop = beam.nodes[tip, 1] - X[:, tip, 1]
    late = np.where((times > times[before]) & (drop > 0.3 * beam.length))[0]
    result["before"] = int(before)
    result["after"] = int(late[0]) if len(late) else len(times) - 1
    return result
