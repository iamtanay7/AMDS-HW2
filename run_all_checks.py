"""Basic checks for the HW2 package.

    python run_all_checks.py

Takes about 1-2 minutes on a laptop CPU. Checks for optimizers you have not
written yet are reported as TODO rather than FAIL. (In the student package the
optimizers are written in HW2_starter.ipynb, so from the command line those
checks always show TODO; the notebook runs the same checks on your functions.)

From the notebook:
    from run_all_checks import run_check, check_random_search
    run_check("random search", check_random_search, random_search, problem)
"""
import time
from dataclasses import replace

import numpy as np
import jax
import jax.numpy as jnp

import _paths  # noqa: F401
from config import CONFIG

CHECKS = []


def check(name):
    """Decorator that registers a check function under a readable name."""
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def run_check(name, fn, *args):
    """Run one check fn(*args), print PASS / FAIL / TODO, and return the status."""
    t = time.perf_counter()
    try:
        msg = fn(*args) or ""
        status = "PASS"
    except NotImplementedError as e:
        status, msg = "TODO", str(e) or "not implemented yet"
    except AssertionError as e:
        status, msg = "FAIL", str(e)
    except Exception as e:  # noqa: BLE001
        status, msg = "FAIL", f"{type(e).__name__}: {e}"
    print(f"[{status}] {name} ({time.perf_counter() - t:.1f} s) {msg}", flush=True)
    return status


def run_checks():
    """Run every registered check and print PASS / FAIL / TODO for each."""
    return [run_check(name, fn) for name, fn in CHECKS]


# ---------------------------------------------------------------------------
@check("HW2 time stepper reproduces the core (MATLAB-matching) solver")
def _():
    from beam_setup import build_beam
    from simulator import make_step
    from simulate import rollout
    beam = build_beam(CONFIG)
    rng = np.random.default_rng(0)
    k = jnp.asarray(rng.uniform(1e3, 5e3, beam.n_links))
    c = 2e-6 * k                  # same order as CONFIG.beta_damping
    n, dt = 300, 2e-5
    # Same physics: constant full load, no nodal drag.
    step = make_step(beam, dt, alpha=0.0, load_factor=lambda t: 1.0)
    state = (beam.model.X0, jnp.zeros_like(beam.model.X0))
    state, _ = jax.lax.scan(lambda s, i: (step(s, i, {"k": k, "c": c}), None), state, jnp.arange(n))
    core_model = beam.model._replace(F_const=beam.load_nodes)
    X_core, _ = rollout({"k": k, "c": c}, core_model, n, dt)
    err = float(jnp.max(jnp.abs(state[0] - X_core)))
    assert err < 1e-14, f"max difference {err:.2e} m"
    return f"max position difference {err:.1e} m"


@check("Targets start at 0 and end at -A")
def _():
    from target_profiles import TARGETS, get_target
    A = 0.015
    for name in TARGETS:
        y = np.asarray(get_target(name, np.array([0.0, 0.5, 1.0]), A, CONFIG.sigmoid_beta))
        assert abs(y[0]) < 1e-15 and abs(y[-1] + A) < 1e-15, f"{name}: y(0)={y[0]}, y(1)={y[-1]}"
        assert np.all(np.diff(np.asarray(get_target(name, np.linspace(0, 1, 101), A))) <= 0), \
            f"{name} is not monotone"
    return "linear, cosine, sigmoid OK"


# Settling check. Its purpose: J must be the static (equilibrium) shape of any design
# an optimizer could prefer, never a snapshot of a beam still in motion. So every
# design must give a finite result, and every design that scores reasonably well
# (RMS error below A) must have settled at t = T. Very soft designs (e.g. all soft)
# droop far past the target and may still creep slowly at T; that is acceptable,
# because they score so badly that no optimizer would choose them. They are reported
# as "poor (not settled)" instead of failing the check.
SETTLED_SPEED = 1e-2      # [m/s] max node speed at T to count as settled
POOR_RMS_RATIO = 1.0      # RMS error / A above which a design counts as clearly poor


@check("Beams are finite; every design that scores well has settled at t = T")
def _():
    from objective import ShapeProblem
    p = ShapeProblem("cosine")
    n = p.n_design
    rng = np.random.default_rng(1)
    designs = {"all soft": np.zeros(n), "all stiff": np.ones(n), "gray z=0.5": 0.5 * np.ones(n),
               "random 50% stiff": (rng.random(n) < 0.5).astype(float),
               "random 80% stiff": (rng.random(n) < 0.8).astype(float)}
    tips, notes = {}, []
    for name, z in designs.items():
        k, c = p.link_properties(z)
        X, V = p.sim["final"](k, c)
        assert bool(jnp.all(jnp.isfinite(X))), f"{name} beam is not finite"
        speed = float(jnp.abs(V).max())
        ratio = p.rms_error_ratio(float(p.objective(z)))
        assert speed < SETTLED_SPEED or ratio > POOR_RMS_RATIO, (
            f"{name}: scores RMS/A = {ratio:.2f} but is still moving ({speed * 1e3:.0f} mm/s) at t = T")
        if speed >= SETTLED_SPEED:
            notes.append(f"{name} poor (RMS/A {ratio:.1f}, not settled)")
        tips[name] = float(p.centerline_deflection(X)[-1])
    assert tips["all soft"] < tips["random 50% stiff"] < tips["random 80% stiff"] < tips["all stiff"] < 0, \
        f"unexpected ordering {tips}"
    assert tips["all soft"] / tips["all stiff"] > 4, "soft and stiff beams deform too similarly"
    assert p.dt < 0.8 * p.dt_limit, "time step too close to the stability limit"
    return (f"tip [mm]: soft {tips['all soft'] * 1e3:.1f}, stiff {tips['all stiff'] * 1e3:.1f}; "
            f"dt = {p.dt / p.dt_limit:.2f} x limit" + ("; " + ", ".join(notes) if notes else ""))


@check("Gradient dJ/dz matches finite differences")
def _():
    from objective import ShapeProblem
    p = ShapeProblem("sigmoid")
    rng = np.random.default_rng(2)
    z = rng.uniform(0.2, 0.8, p.n_design)
    J, g = p.value_and_grad(z)
    d = rng.standard_normal(p.n_design)
    h = 1e-5
    fd = (float(p.objective(z + h * d)) - float(p.objective(z - h * d))) / (2 * h)
    ad = float(np.dot(g, d))
    rel = abs(fd - ad) / abs(fd)
    assert rel < 1e-5, f"directional derivative AD {ad:.6e} vs FD {fd:.6e}"
    return f"relative error {rel:.1e}"


@check("Fracture: force criterion, failed links carry no force")
def _():
    from fracture import run_fracture_demo
    from physics import edge_forces
    r = run_fracture_demo(CONFIG)
    assert r["first_failure_time"] is not None, "no link failed"
    assert r["first_failure_time"] > 0.2 * CONFIG.fracture_sim_time, "fracture starts too early"
    assert r["n_failed"] < 0.5 * r["beam"].n_links, "too much of the beam failed"
    # Force-based: halving E halves the stiffness but should NOT change the load
    # at which the first link fails (an elongation criterion would halve it).
    r2 = run_fracture_demo(replace(CONFIG, fracture_E=CONFIG.fracture_E / 2))
    ratio = r2["first_failure_load"] / r["first_failure_load"]
    assert abs(ratio - 1) < 0.05, f"first-failure load changed by x{ratio:.2f} when E halved"
    # A failed link (a = 0) produces exactly zero spring and damping force.
    beam = r["beam"]
    failed = np.isfinite(np.asarray(r["out"]["fail_time"]))
    X = r["out"]["X"][-1]
    V = jnp.asarray(np.random.default_rng(3).standard_normal(X.shape))
    k = jnp.ones(beam.n_links) * 1e3
    F = edge_forces(X, V, beam.model, {"k": k, "c": 0.1 * k, "a": jnp.asarray(~failed, float)})
    assert float(jnp.abs(F[failed]).max()) == 0.0, "a failed link still carries force"
    return (f"first failure t = {r['first_failure_time']:.3f} s at F = {r['first_failure_load']:.2f} N; "
            f"{r['n_failed']} links failed; same load with E/2 (x{ratio:.3f})")


def _small_problem():
    from objective import ShapeProblem
    return ShapeProblem("cosine")


def check_random_search(random_search, problem=None):
    """Your random_search: uses exactly max_evals, returns a binary best design
    whose J is the minimum of the log, and is reproducible for a fixed seed."""
    p = problem or _small_problem()
    r = random_search(p, max_evals=20, seed=0)
    assert len(r["log"]) == 20, f"used {len(r['log'])} evaluations, expected 20"
    assert set(np.unique(r["best_design"])) <= {0.0, 1.0}, "best design is not binary"
    assert abs(r["best_J"] - min(J for J, _, _ in r["log"])) < 1e-20, "best_J is not the minimum"
    r2 = random_search(p, max_evals=20, seed=0)
    assert r2["best_J"] == r["best_J"], "same seed gave a different result"
    return f"best J = {r['best_J']:.3e}"


def check_genetic_algorithm(genetic_algorithm, problem=None):
    """Your genetic_algorithm: stays within budget, improves on the initial
    population, returns a binary design."""
    p = problem or _small_problem()
    r = genetic_algorithm(p, max_evals=200, seed=0, pop_size=20)
    assert len(r["log"]) <= 200, f"used {len(r['log'])} > 200 evaluations"
    initial_best = min(J for J, _, _ in r["log"][:20])
    assert r["best_J"] < initial_best, "GA did not improve on the initial population"
    assert set(np.unique(r["best_design"])) <= {0.0, 1.0}, "best design is not binary"
    return f"initial best {initial_best:.3e} -> {r['best_J']:.3e}"


def check_gradient_optimize(gradient_optimize, problem=None):
    """Your gradient_optimize: stays within budget, lowers J, returns a valid
    thresholded (0/1) design."""
    p = problem or _small_problem()
    r = gradient_optimize(p, max_evals=60, seed=0)
    assert len(r["log"]) <= 60, f"used {len(r['log'])} > 60 evaluations"
    first = r["log"][0][0]
    assert min(J for J, _, _ in r["log"]) < first, "objective did not decrease"
    b = np.asarray(r["best_design"])
    assert b.shape == (p.n_design,) and set(np.unique(b)) <= {0.0, 1.0}, "invalid thresholded design"
    return f"J {first:.3e} -> best thresholded {r['best_J']:.3e}"


def check_convergence(random_search, problem=None):
    """Repeated runs give non-increasing best-so-far curves of the right shape."""
    import os
    import tempfile
    import matplotlib.pyplot as plt
    from convergence import plot_convergence, run_trials
    target = problem.target_name if problem is not None else "linear"
    data = run_trials(random_search, target, n_seeds=3, budget=10, verbose=False,
                      problem=problem)
    assert data["curves"].shape == (3, 10), "wrong curve shape"
    assert np.all(np.diff(data["curves"], axis=1) <= 0), "best-so-far curve increases"
    path = os.path.join(tempfile.gettempdir(), "hw2_check_convergence.png")
    plt.close(plot_convergence({"random": data}, target, path))
    return f"plot written to {path}"


def _method(name):
    from convergence import get_method
    return get_method(name)


@check("Random search: runs, counts evaluations, keeps best")
def _():
    return check_random_search(_method("random"))


@check("Genetic algorithm improves on its initial population")
def _():
    return check_genetic_algorithm(_method("genetic"))


@check("Gradient method improves J and returns a valid thresholded design")
def _():
    return check_gradient_optimize(_method("gradient"))


@check("Repeated-run convergence statistics and plot")
def _():
    return check_convergence(_method("random"))


@check("CPU timing")
def _():
    from objective import ShapeProblem
    p = ShapeProblem("linear")
    z = 0.5 * np.ones(p.n_design)
    p.evaluate(z), p.value_and_grad(z)            # compile
    t = time.perf_counter(); p.evaluate(z); t_f = time.perf_counter() - t
    t = time.perf_counter(); p.value_and_grad(z); t_g = time.perf_counter() - t
    return f"forward {t_f * 1e3:.0f} ms, value_and_grad {t_g * 1e3:.0f} ms ({p.n_steps} steps)"


if __name__ == "__main__":
    results = run_checks()
    counts = {s: results.count(s) for s in ("PASS", "FAIL", "TODO")}
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['TODO']} TODO")
    raise SystemExit(1 if counts["FAIL"] else 0)
