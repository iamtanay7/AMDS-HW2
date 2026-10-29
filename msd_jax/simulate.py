"""Fixed-step rollouts with lax.scan and optional rematerialization.

Checkpoint modes (affect reverse-mode memory only, not results):

  "none"    : plain scan. Backprop stores every per-step intermediate
              (R, U, L, n, beta, Fedge, ...) for all steps: O(T * M * D).
  "step"    : jax.checkpoint around each step. Stores only (X, V) per step:
              O(T * N * D); recomputes each step's internals in the backward.
  "segment" : two-level (sqrt) checkpointing. Steps are grouped into segments
              of length S (default ceil(sqrt(T))); only segment-boundary states
              are kept, and each segment is replayed in the backward pass.
              Memory O((T/S + S) * N * D) at roughly one extra forward.
"""

from functools import partial
from math import ceil, sqrt

import jax
import jax.numpy as jnp

from physics import linear_damper, linear_spring, semi_implicit_euler_step


def _scan_steps(state, n, step_fn, remat_step):
    if n == 0:
        return state
    body = jax.checkpoint(step_fn) if remat_step else step_fn
    state, _ = jax.lax.scan(lambda s, _: (body(s), None), state, None, length=n)
    return state


@partial(jax.jit, static_argnames=("n_steps", "spring_law", "damper_law",
                                   "checkpoint", "segment_len"))
def rollout(params, model, n_steps, dt, spring_law=linear_spring,
            damper_law=linear_damper, checkpoint="segment", segment_len=None):
    """Integrate from (X0, V=0) for n_steps; return final (X, V)."""
    step_fn = lambda s: semi_implicit_euler_step(s, model, params, dt,
                                                 spring_law, damper_law)
    state = (model.X0, jnp.zeros_like(model.X0))

    if checkpoint == "none":
        return _scan_steps(state, n_steps, step_fn, remat_step=False)
    if checkpoint == "step":
        return _scan_steps(state, n_steps, step_fn, remat_step=True)
    if checkpoint != "segment":
        raise ValueError(f"unknown checkpoint mode {checkpoint!r}")

    S = segment_len or max(1, ceil(sqrt(n_steps)))
    n_seg, rem = divmod(n_steps, S)

    @jax.checkpoint
    def segment(s):
        return _scan_steps(s, S, step_fn, remat_step=False)

    state, _ = jax.lax.scan(lambda s, _: (segment(s), None), state, None,
                            length=n_seg)
    return _scan_steps(state, rem, step_fn, remat_step=True)


@partial(jax.jit, static_argnames=("n_steps", "stride", "spring_law",
                                   "damper_law"))
def trajectory(params, model, n_steps, dt, stride=1, spring_law=linear_spring,
               damper_law=linear_damper):
    """Forward-only rollout that also returns X every `stride` steps.

    Returns (X_final, V_final, X_hist) with X_hist of shape
    (n_steps // stride, N, D). Intended for plotting and objectives on small
    problems; use rollout() for large gradient runs.
    """
    step_fn = lambda s: semi_implicit_euler_step(s, model, params, dt,
                                                 spring_law, damper_law)

    def chunk(s, _):
        s = _scan_steps(s, stride, step_fn, remat_step=False)
        return s, s[0]

    state = (model.X0, jnp.zeros_like(model.X0))
    state, X_hist = jax.lax.scan(chunk, state, None, length=n_steps // stride)
    state = _scan_steps(state, n_steps % stride, step_fn, remat_step=False)
    return state[0], state[1], X_hist
