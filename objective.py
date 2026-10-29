"""The HW2 shape-matching objective, shared by every optimization method.

    J(design) = mean over centerline nodes of (y_centerline - y_target)^2     [m^2]

y_centerline is the vertical displacement of each centerline node at the end
of the simulation (t = sim_time). It is compared with the target at that
node's *undeformed* normalized position xi. Because nodes also move a little
in x, xi is a material label, not the deformed x-position.

Function-evaluation convention (every method is charged the same way):
    problem.evaluate(design)          1 evaluation
    problem.evaluate_batch(designs)   len(designs) evaluations
    problem.value_and_grad(z)         1 evaluation
Every call is logged in problem.log, so convergence curves are built
identically for all methods.

Example:
    from objective import ShapeProblem
    problem = ShapeProblem("cosine")
    b = np.ones(problem.n_design)            # all stiff
    J = problem.evaluate(b)
    J, dJdz = problem.value_and_grad(0.5 * np.ones(problem.n_design))
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

import _paths  # noqa: F401
from beam_setup import build_beam, choose_time_step
from config import CONFIG
from materials import (design_size, design_to_modulus, link_damping,
                       material_to_link_stiffness)
from simulator import make_simulator
from target_profiles import get_target


class ShapeProblem:
    """Cantilever shape-matching problem for one target ("linear", "cosine", "sigmoid")."""

    def __init__(self, target_name, config=CONFIG, design_level="cell"):
        self.config = config
        self.target_name = target_name
        self.design_level = design_level
        self.beam = build_beam(config)
        self.n_design = design_size(self.beam, design_level)

        k_stiffest, _ = self.link_properties(np.ones(self.n_design))
        self.dt, self.n_steps, self.dt_limit = choose_time_step(self.beam, k_stiffest, config)
        self.sim = make_simulator(self.beam, self.dt, self.n_steps, config)

        self.A = config.target_amplitude_ratio * self.beam.length
        self.y_target = get_target(target_name, self.beam.xi, self.A, config.sigmoid_beta)

        self._J = jax.jit(self.objective)
        self._value_and_grad = jax.jit(jax.value_and_grad(self.objective))
        self.reset_counter()

    # ------------------------------------------------------------------
    # Physics: design -> link properties -> final shape -> J
    # ------------------------------------------------------------------
    def link_properties(self, z):
        """Per-link stiffness k and damping c for design z (binary or continuous)."""
        E = design_to_modulus(jnp.asarray(z, jnp.float64), self.config)
        k = material_to_link_stiffness(E, self.beam, self.design_level)
        return k, link_damping(k, self.config)

    def final_positions(self, z):
        """Node positions [N x 2] at t = sim_time for design z (not counted)."""
        k, c = self.link_properties(z)
        X, V = self.sim["final"](k, c)
        return X

    def centerline_deflection(self, X):
        """Vertical displacement of the centerline nodes, ordered from clamp to tip."""
        return X[self.beam.center_nodes, 1] - self.beam.model.X0[self.beam.center_nodes, 1]

    def objective(self, z):
        """J(z) = mean((y_centerline - y_target)^2). Differentiable in z."""
        y = self.centerline_deflection(self.final_positions(z))
        return jnp.mean((y - self.y_target) ** 2)

    # ------------------------------------------------------------------
    # Counted evaluations (use these inside optimizers)
    # ------------------------------------------------------------------
    def reset_counter(self):
        """Start a new run: zero the evaluation count, clock and log."""
        self.n_evals = 0
        self.wall_time = 0.0
        self.log = []   # one entry per evaluation: (J, is_binary_design, wall_time)

    def _record(self, values, designs, elapsed):
        for J, d in zip(values, designs):
            binary = bool(np.all((d == 0) | (d == 1)))
            self.n_evals += 1
            self.wall_time += elapsed / len(values)
            self.log.append((float(J), binary, self.wall_time))

    def evaluate(self, design):
        """J for one design. Counts as 1 function evaluation."""
        design = np.asarray(design, dtype=float)
        t = time.perf_counter()
        J = float(self._J(design))
        self._record([J], [design], time.perf_counter() - t)
        return J

    def evaluate_batch(self, designs):
        """J for each row of `designs` [n x n_design]. Counts as n evaluations.

        (A plain loop: on CPU, jax.vmap over this small simulation was
        measured to be slower per design than evaluating one at a time.)
        """
        designs = np.atleast_2d(np.asarray(designs, dtype=float))
        return np.array([self.evaluate(d) for d in designs])

    def value_and_grad(self, z):
        """(J, dJ/dz) for a continuous design z in [0,1]. Counts as 1 evaluation."""
        z = np.asarray(z, dtype=float)
        t = time.perf_counter()
        J, g = self._value_and_grad(z)
        J, g = float(J), np.asarray(g)
        self._record([J], [z], time.perf_counter() - t)
        return J, g

    # ------------------------------------------------------------------
    def rms_error_ratio(self, J):
        """sqrt(J) / A: RMS centerline error as a fraction of the tip target."""
        return float(np.sqrt(J) / self.A)

    def describe(self):
        b, c = self.beam, self.config
        return (f"{self.target_name} target: {b.n_cells_x}x{b.n_cells_y} cells, "
                f"{b.model.N} nodes, {b.n_links} links, {self.n_design} design variables "
                f"({self.design_level}); A = {self.A * 1e3:.2f} mm; "
                f"dt = {self.dt:.3g} s ({self.dt / self.dt_limit:.2f} of stability limit), "
                f"{self.n_steps} steps")
