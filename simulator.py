"""Time stepping for the HW2 beam (you should not need to modify this file).

Each step uses the same semi-implicit Euler update as the core simulator
(msd_jax, which matches the MATLAB fast solver), plus two things HW2 needs:

  * a time-varying load:  F_ext(t) = load_factor(t) * beam.load_nodes
  * nodal drag -alpha * m * v (the alpha*M Rayleigh term) so the beam
    settles to static equilibrium within sim_time.

    a = (F_links + F_ext(t)) / m - alpha * v      (a = 0 on clamped DOFs)
    v <- v + dt * a
    x <- x + dt * v

Link forces come from physics.edge_forces/nodal_forces in msd_jax, so spring
geometry is fully nonlinear (lengths and directions updated every step).
"""
import jax
import jax.numpy as jnp

import _paths  # noqa: F401
from physics import edge_forces, nodal_forces


def smooth_ramp(t, ramp_time):
    """0 at t=0, rising smoothly to 1 at t=ramp_time, then constant 1."""
    tau = jnp.clip(t / ramp_time, 0.0, 1.0)
    return 0.5 * (1.0 - jnp.cos(jnp.pi * tau))


def make_step(beam, dt, alpha, load_factor):
    """Return step(state, n, params) -> new state, advancing from t = n*dt."""
    model = beam.model

    def step(state, n, params):
        X, V = state
        F = nodal_forces(edge_forces(X, V, model, params), model)
        F = F + load_factor(n * dt) * beam.load_nodes
        A = F * model.inv_mass - alpha * V
        A = jnp.where(model.fixed, 0.0, A)
        V = jnp.where(model.fixed, 0.0, V + dt * A)
        X = jnp.where(model.fixed, model.X0, X + dt * V)
        return (X, V)

    return step


def make_simulator(beam, dt, n_steps, config):
    """Build jitted simulation functions for the shape-matching load case.

    Returns a dict with
      "final"   : final(k, c)              -> (X, V) at t = sim_time
      "history" : history(k, c, n_frames)  -> X at n_frames evenly spaced times
    k and c are per-link arrays of length beam.n_links.
    """
    load_factor = lambda t: smooth_ramp(t, config.ramp_time)
    step = make_step(beam, dt, config.alpha_damping, load_factor)
    X0 = beam.model.X0

    @jax.jit
    def final(k, c):
        params = {"k": k, "c": c}
        # jax.checkpoint: during backpropagation, recompute each step instead of
        # storing its intermediates, so memory stays small for long runs.
        body = jax.checkpoint(lambda s, n: (step(s, n, params), None))
        state, _ = jax.lax.scan(body, (X0, jnp.zeros_like(X0)), jnp.arange(n_steps))
        return state

    def history(k, c, n_frames=50):
        stride = max(1, n_steps // n_frames)

        @jax.jit
        def run(k, c):
            params = {"k": k, "c": c}

            def chunk(s, n0):
                s, _ = jax.lax.scan(lambda s, n: (step(s, n, params), None), s,
                                    n0 + jnp.arange(stride))
                return s, s[0]
            _, frames = jax.lax.scan(chunk, (X0, jnp.zeros_like(X0)),
                                     stride * jnp.arange(n_steps // stride))
            return frames
        times = dt * stride * (1 + jnp.arange(n_steps // stride))
        return times, run(k, c)

    return {"final": final, "history": history}
