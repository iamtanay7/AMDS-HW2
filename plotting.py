"""Plotting helpers for beams and designs (you should not need to modify this file)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


def draw_links(ax, X, beam, failed=None, k=None, title=""):
    """Draw every link of the beam at node positions X [N x 2].

    failed : optional boolean array [M]; failed links are drawn red and dashed.
    k      : optional link stiffness [M]; stiffer links are drawn darker.
    """
    X = np.asarray(X)
    segments = np.stack([X[beam.edges[:, 0]], X[beam.edges[:, 1]]], axis=1)
    if failed is None:
        failed = np.zeros(len(segments), bool)
    failed = np.asarray(failed)
    intact = ~failed
    if k is None:
        colors = "0.35"
    else:
        k = np.asarray(k)
        shade = (k - k.min()) / max(np.ptp(k), 1e-30)
        colors = plt.cm.Greys(0.35 + 0.6 * shade[intact])
    ax.add_collection(LineCollection(segments[intact], colors=colors, linewidths=0.8))
    if failed.any():
        ax.add_collection(LineCollection(segments[failed], colors="red",
                                         linewidths=1.8, linestyles="--"))
    fixed = np.asarray(beam.model.fixed[:, 0])
    ax.plot(X[fixed, 0], X[fixed, 1], "s", color="k", ms=3)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


def draw_design(ax, beam, design, title=""):
    """Show a cell-level binary/continuous design as pixels (dark = stiff)."""
    design = np.asarray(design)
    h = beam.cell_size
    for value, (cx, cy) in zip(design, beam.cell_centers):
        ax.add_patch(Rectangle((cx - h / 2, cy - h / 2), h, h,
                               color=plt.cm.Greys(0.1 + 0.8 * float(value))))
    ax.set_xlim(0, beam.length)
    ax.set_ylim(0, beam.cell_size * beam.n_cells_y)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


def plot_design_result(problem, design, filename=None, title=None):
    """Three panels: the design (pixels), the deformed beam, and the centerline
    vs the target. Uses problem.final_positions, so it is not counted as an
    evaluation."""
    design = np.asarray(design, dtype=float)
    X = np.asarray(problem.final_positions(design))
    k, _ = problem.link_properties(design)
    beam = problem.beam
    y = np.asarray(problem.centerline_deflection(X))
    J = float(np.mean((y - np.asarray(problem.y_target)) ** 2))

    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    if problem.design_level == "cell":
        draw_design(axes[0], beam, design, "design (dark = stiff)")
    else:
        draw_links(axes[0], beam.nodes, beam, k=k, title="design (darker link = stiffer)")
    draw_links(axes[1], X, beam, k=k, title="deformed beam at t = sim_time")
    axes[1].plot(*beam.nodes[beam.center_nodes].T, ":", color="tab:blue", lw=1)
    xi = np.asarray(beam.xi)
    axes[2].plot(xi, 1e3 * np.asarray(problem.y_target), "k--", label=f"{problem.target_name} target")
    axes[2].plot(xi, 1e3 * y, "o-", color="tab:red", ms=3, label="beam centerline")
    axes[2].set_xlabel("xi = normalized position along the beam")
    axes[2].set_ylabel("vertical displacement [mm]")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_title(f"J = {J:.3e} m^2   (RMS error = {problem.rms_error_ratio(J) * 100:.1f}% of A)",
                      fontsize=10)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=120)
        plt.close(fig)
    return fig


def plot_fracture_demo(result, filename=None):
    """Four panels for fracture.run_fracture_demo(): initial beam, just before
    the first failure, after fracture (failed links red), and load and number
    of failed links vs time. Returns the figure; saves it if filename is given."""
    beam, out, config = result["beam"], result["out"], result["config"]
    times = np.asarray(out["times"])
    X = np.asarray(out["X"])
    before, after = result["before"], result["after"]
    failed_after = np.asarray(out["fail_time"]) <= times[after]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    lim = dict(xlim=(-0.005, beam.length + 0.01),
               ylim=(-0.45 * beam.length, beam.nodes[:, 1].max() + 0.008))
    draw_links(axes[0, 0], beam.nodes, beam, title="Initial beam (t = 0)")
    draw_links(axes[0, 1], X[before], beam,
               title=f"Just before first failure (t = {times[before]:.3f} s, "
                     f"F = {config.fracture_load_rate * times[before]:.2f} N)")
    draw_links(axes[1, 0], X[after], beam, failed=failed_after,
               title=f"After fracture (t = {times[after]:.3f} s): "
                     f"{int(failed_after.sum())} failed links (red)")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set(**lim)

    ax = axes[1, 1]
    ax.plot(times, np.asarray(out["load"]), color="tab:blue")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("applied load F(t) [N]", color="tab:blue")
    ax2 = ax.twinx()
    ax2.step(times, np.asarray(out["n_failed"]), color="tab:red", where="post")
    ax2.set_ylabel("failed links", color="tab:red")
    if result["first_failure_time"] is not None:
        ax.axvline(result["first_failure_time"], color="0.5", ls="--", lw=1,
                   label="first failure")
        ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Load and failed links vs time")
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=120)
    return fig
