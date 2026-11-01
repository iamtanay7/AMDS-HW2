"""Repeated-run convergence experiments and plots.

    python convergence.py --target cosine                  # all methods, 10 seeds
    python convergence.py --target linear --methods random genetic --seeds 5 --budget 500
    python convergence.py --plot-only                      # re-plot saved results

For every run, the best-so-far curve is

    best[i] = lowest J among the BINARY designs evaluated in evaluations 1..i

with i counted over ALL function evaluations (problem.log). Continuous designs
from the gradient method are charged evaluations but cannot set the
best-so-far value, because they are not buildable soft/stiff designs.

Across seeds we plot the mean curve and a shaded band. With fewer than 10 runs
(the assignment uses 5) the band is the min-max range over the runs; with many
runs it can be the 2.5-97.5 percentile band. Either way it shows run-to-run
spread; it is not a confidence interval of the mean.

In the notebook, pass your own optimizer functions to run_trials:
    data = run_trials(random_search, "cosine", n_seeds=5, budget=3000, problem=problem)
"""
import argparse
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

import _paths  # noqa: F401
from config import CONFIG
from objective import ShapeProblem

RESULTS_DIR = "results"
COLORS = {"random": "tab:gray", "genetic": "tab:blue", "gradient": "tab:red"}


METHOD_NAMES = {"random_search": "random", "genetic_algorithm": "genetic",
                "gradient_optimize": "gradient"}


def get_method(name):
    """Look up an optimizer module by name (instructor tree / command line).
    Each optimizer has signature f(problem, max_evals, seed).
    In the student package the optimizers live in HW2_starter.ipynb instead;
    pass the functions themselves to run_trials."""
    import importlib
    modules = {"random": ("random_search", "random_search"),
               "genetic": ("evolutionary", "genetic_algorithm"),
               "gradient": ("gradient_opt", "gradient_optimize")}
    if name not in modules:
        raise ValueError(f"unknown method {name!r}")
    module, function = modules[name]
    try:
        return getattr(importlib.import_module(module), function)
    except ModuleNotFoundError as e:
        if e.name != module:
            raise
        raise NotImplementedError(
            f"TODO: write {function} in HW2_starter.ipynb and pass it to run_trials") from None


def best_so_far_curve(log, budget):
    """Best binary J after each of `budget` evaluations (NaN before the first
    binary design; the last value is carried forward if the run used fewer)."""
    curve = np.full(budget, np.nan)
    best = np.inf
    for i, (J, is_binary, _) in enumerate(log[:budget]):
        if is_binary and J < best:
            best = J
        curve[i] = best if np.isfinite(best) else np.nan
    if 0 < len(log) < budget:
        curve[len(log):] = curve[len(log) - 1]
    return curve


def summarize(curves):
    """Mean, 2.5th and 97.5th percentile over runs (rows) at every evaluation."""
    curves = np.asarray(curves)
    with warnings.catch_warnings():
        # Columns that are all NaN (no binary design evaluated yet, e.g. the
        # gradient method's first evaluation) stay NaN and are not plotted.
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmean(curves, axis=0),
                np.nanpercentile(curves, 2.5, axis=0),
                np.nanpercentile(curves, 97.5, axis=0))


def run_trials(method, target, n_seeds, budget, config=CONFIG, verbose=True,
               method_name=None, problem=None):
    """Run one optimizer for seeds 0..n_seeds-1. Returns curves, final J, wall times
    and the best design of every run.

    method  : an optimizer function f(problem, max_evals, seed=...) or the name
              "random", "genetic" or "gradient" (looked up with get_method)
    problem : optional ShapeProblem for `target` to reuse (saves compile time)
    """
    if isinstance(method, str):
        method_name, method = method, get_method(method)
    elif method_name is None:
        method_name = METHOD_NAMES.get(method.__name__, method.__name__)
    if problem is None:
        problem = ShapeProblem(target, config)
    curves, finals, walls, designs = [], [], [], []
    for seed in range(n_seeds):
        t = time.perf_counter()
        result = method(problem, budget, seed=seed)
        walls.append(time.perf_counter() - t)
        curves.append(best_so_far_curve(result["log"], budget))
        finals.append(result["best_J"])
        designs.append(result["best_design"])
        if verbose:
            print(f"  {target:8s} {method_name:8s} seed {seed:2d}: best J = {result['best_J']:.3e} "
                  f"(rms/A = {problem.rms_error_ratio(result['best_J']):.3f}), "
                  f"{problem.n_evals} evals, {walls[-1]:.1f} s", flush=True)
    return {"curves": np.array(curves), "final_J": np.array(finals),
            "wall_time": np.array(walls), "designs": np.array(designs)}


def save_results(target, method_name, data, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    np.savez(os.path.join(results_dir, f"{target}_{method_name}.npz"), **data)


def has_results(target, method_name, n_seeds, budget, results_dir=RESULTS_DIR):
    """True if results_dir holds a finished run set for this target and method
    with exactly n_seeds runs x budget evaluations (used to resume after a
    disconnect without recomputing finished work)."""
    path = os.path.join(results_dir, f"{target}_{method_name}.npz")
    if not os.path.exists(path):
        return False
    with np.load(path) as data:
        return data["curves"].shape == (n_seeds, budget)


def load_results(target, results_dir=RESULTS_DIR):
    """Saved results for one target: {method_name: data}, for every method found."""
    out = {}
    for method_name in COLORS:
        path = os.path.join(results_dir, f"{target}_{method_name}.npz")
        if os.path.exists(path):
            with np.load(path) as data:
                out[method_name] = dict(data)
    return out


def plot_convergence(results, target, filename=None, band="auto"):
    """One figure: mean best-so-far curve and a band per method.

    band = "minmax"    : best and worst run
    band = "percentile": 2.5-97.5 percentile of the runs (meaningful only with many runs)
    band = "auto"      : min-max for fewer than 10 runs (the assignment uses 5), else percentile
    Saves it to filename (default results/convergence_<target>.png) and returns it."""
    if band == "auto":
        band = "minmax" if min(len(d["curves"]) for d in results.values()) < 10 else "percentile"
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method_name, data in results.items():
        curves = np.asarray(data["curves"])
        mean, lo, hi = summarize(curves)
        if band == "minmax":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi = np.nanmin(curves, axis=0), np.nanmax(curves, axis=0)
        evals = np.arange(1, len(mean) + 1)
        color = COLORS.get(method_name)
        ax.plot(evals, mean, color=color, label=f"{method_name} (mean of {len(curves)} run{'s' if len(curves) != 1 else ''})")
        ax.fill_between(evals, lo, hi, color=color, alpha=0.2)
    ax.set_yscale("log")
    budget = max(len(np.asarray(d["curves"])[0]) for d in results.values())
    runs = sorted({len(d["curves"]) for d in results.values()})
    ax.set_xlabel("Function evaluations", fontsize=11)
    ax.set_ylabel("Best-so-far objective J  [m$^2$]", fontsize=11)
    ax.set_xlim(0, budget)
    ax.tick_params(labelsize=9, labelbottom=True, labelleft=True)
    label = "2.5-97.5 percentile of runs" if band == "percentile" else "min-max over runs"
    ax.set_title(f"Target: {target}\nBudget: {budget} evaluations    Runs: {'/'.join(map(str, runs))}\n"
                 f"(line: mean best-so-far; shaded: {label})", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    filename = filename or os.path.join(RESULTS_DIR, f"convergence_{target}.png")
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig.savefig(filename, dpi=130, bbox_inches="tight")
    return fig


def final_statistics(results, A):
    """Per method: statistics of the final best binary J of each run, as RMS error / A."""
    rows = {}
    for method_name, data in results.items():
        r = np.sqrt(np.asarray(data["final_J"])) / A
        rows[method_name] = {
            "runs": len(r), "budget": np.asarray(data["curves"]).shape[1],
            "mean": r.mean(), "std": r.std(ddof=1) if len(r) > 1 else 0.0,
            "best": r.min(), "worst": r.max(),
            "seconds_per_run": float(np.mean(data["wall_time"]))}
    return rows


def summary_table(results, target, A):
    """Text table of final_statistics (RMS error / A at the end of each run)."""
    lines = [f"{target} target, RMS error / A at the end of each run",
             f"{'method':9s} {'runs':>4s} {'mean':>7s} {'std':>7s} {'best':>7s} {'worst':>7s} {'s/run':>6s}"]
    for method_name, s in final_statistics(results, A).items():
        lines.append(f"{method_name:9s} {s['runs']:4d} {s['mean']:7.4f} {s['std']:7.4f} "
                     f"{s['best']:7.4f} {s['worst']:7.4f} {s['seconds_per_run']:6.1f}")
    return "\n".join(lines)


def write_summary_csv(all_results, A, filename):
    """One CSV row per (target, method) with the final_statistics columns."""
    import csv
    columns = ["runs", "budget", "mean", "std", "best", "worst", "seconds_per_run"]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "method"] + [f"rms_over_A_{c}" if c not in
                        ("runs", "budget", "seconds_per_run") else c for c in columns])
        for target, results in all_results.items():
            for method_name, s in final_statistics(results, A).items():
                writer.writerow([target, method_name] + [
                    s[c] if c in ("runs", "budget") else f"{s[c]:.6g}" for c in columns])
    return filename


def outputs_up_to_date(target, results_dir, submission_dir):
    """True if submission_dir already holds the plots for exactly the results
    saved in results_dir (save_target_outputs copies the .npz files there)."""
    import filecmp
    names = [f"{target}_{m}.npz" for m in COLORS
             if os.path.exists(os.path.join(results_dir, f"{target}_{m}.npz"))]
    if not names or not os.path.exists(os.path.join(submission_dir, f"convergence_{target}.png")):
        return False
    copies = os.path.join(submission_dir, "results")
    return all(os.path.exists(os.path.join(copies, n)) and
               filecmp.cmp(os.path.join(results_dir, n), os.path.join(copies, n), shallow=False)
               for n in names)


def save_target_outputs(target, problem, results_dir, submission_dir):
    """Write everything for one target into submission_dir, right after it has run:
    the convergence plot, the best design of each method, and a copy of the raw
    results (.npz). Returns the loaded results."""
    import shutil
    from plotting import plot_design_result
    results = load_results(target, results_dir)
    if not results:
        return results
    os.makedirs(os.path.join(submission_dir, "results"), exist_ok=True)
    plt.close(plot_convergence(results, target,
                               os.path.join(submission_dir, f"convergence_{target}.png")))
    for method_name, data in results.items():
        best_run = int(np.argmin(data["final_J"]))
        plot_design_result(problem, data["designs"][best_run],
                           os.path.join(submission_dir, f"best_{target}_{method_name}.png"),
                           title=f"{target} target, best {method_name} design (run {best_run})")
        shutil.copy(os.path.join(results_dir, f"{target}_{method_name}.npz"),
                    os.path.join(submission_dir, "results"))
    return results


def main():
    import matplotlib
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", nargs="+", default=["linear", "cosine", "sigmoid"])
    parser.add_argument("--methods", nargs="+", default=["random", "genetic", "gradient"])
    parser.add_argument("--seeds", type=int, default=CONFIG.n_seeds)
    parser.add_argument("--budget", type=int, default=CONFIG.eval_budget)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    for target in args.target:
        if not args.plot_only:
            for method_name in args.methods:
                data = run_trials(method_name, target, args.seeds, args.budget)
                save_results(target, method_name, data)
        results = load_results(target)
        if results:
            plt.close(plot_convergence(results, target))
            print(f"Wrote {os.path.join(RESULTS_DIR, f'convergence_{target}.png')}")
            for method_name, data in results.items():
                print(f"  {method_name:8s}: final best J mean {np.mean(data['final_J']):.3e}, "
                      f"median {np.median(data['final_J']):.3e}, "
                      f"mean wall time {np.mean(data['wall_time']):.1f} s/run")


if __name__ == "__main__":
    main()
