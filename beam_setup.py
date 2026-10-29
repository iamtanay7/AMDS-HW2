"""Build the 2-D cantilever lattice used throughout HW2.

The beam is a grid of square cells. Every cell has 4 side links (shared with
neighbouring cells) and 2 diagonal links:

        o---o---o---o        o = node (lumped mass)
        |\\ /|\\ /|\\ /|       left column of nodes is clamped
        | X | X | X |       load is applied at the right end
        |/ \\|/ \\|/ \\|
        o---o---o---o

Numbering (0-based, column-major like MATLAB):
    node (ix, iy)  ->  ix * (n_cells_y + 1) + iy
    cell (ix, iy)  ->  ix * n_cells_y + iy
ix runs along the length (x), iy through the height (y).
"""
import numpy as np
import jax.numpy as jnp

import _paths  # noqa: F401  (makes the msd_jax simulator importable)
from model import prepare, num_steps
from physics import stable_dt_estimate


class Beam:
    """Container for everything that describes one lattice beam.

    Main attributes (shapes in brackets, N nodes, M links, C cells):
        model          core simulator Model (nodes, links, mass, fixed DOFs)
        nodes          [N x 2] undeformed node coordinates
        edges          [M x 2] node indices of each link
        link_type      [M]     "horizontal", "vertical" or "diagonal"
        cell_to_link   [M x C] link stiffness = thickness * cell_to_link @ E_cell
        link_weight    [M]     row sums of cell_to_link (used for link-level designs)
        center_nodes   [n]     node indices on the beam centerline, left to right
        xi             [n]     normalized x-position of each centerline node, 0..1
        load_nodes     [N x 2] force on every node at full load
        dt, n_steps            time step and number of steps
    """


def node_index(ix, iy, n_cells_y):
    return ix * (n_cells_y + 1) + iy


def cell_index(ix, iy, n_cells_y):
    return ix * n_cells_y + iy


def build_lattice(n_cells_x, n_cells_y, length, height):
    """Return nodes, edges, link types and the cell->link matrix (all NumPy)."""
    hx = length / n_cells_x
    hy = height / n_cells_y
    if abs(hx - hy) > 1e-12 * max(hx, hy):
        raise ValueError("Cells must be square: beam_length/n_cells_x must equal "
                         "beam_height/n_cells_y (the E -> k mapping assumes it).")

    nodes = []
    for ix in range(n_cells_x + 1):
        for iy in range(n_cells_y + 1):
            nodes.append([ix * hx, iy * hy])
    nodes = np.array(nodes)

    edges, types, cells_of_edge = [], [], []

    def add(a, b, kind, cells):
        edges.append([a, b])
        types.append(kind)
        cells_of_edge.append(cells)

    n = lambda ix, iy: node_index(ix, iy, n_cells_y)
    c = lambda ix, iy: cell_index(ix, iy, n_cells_y)

    # Horizontal links: shared by the cell below and the cell above.
    for ix in range(n_cells_x):
        for iy in range(n_cells_y + 1):
            cells = [c(ix, j) for j in (iy - 1, iy) if 0 <= j < n_cells_y]
            add(n(ix, iy), n(ix + 1, iy), "horizontal", cells)
    # Vertical links: shared by the cell to the left and the cell to the right.
    for ix in range(n_cells_x + 1):
        for iy in range(n_cells_y):
            cells = [c(i, iy) for i in (ix - 1, ix) if 0 <= i < n_cells_x]
            add(n(ix, iy), n(ix, iy + 1), "vertical", cells)
    # Two diagonals inside each cell.
    for ix in range(n_cells_x):
        for iy in range(n_cells_y):
            add(n(ix, iy), n(ix + 1, iy + 1), "diagonal", [c(ix, iy)])
            add(n(ix, iy + 1), n(ix + 1, iy), "diagonal", [c(ix, iy)])

    # Each cell gives 3/8 * E * t to each of its 6 links (see materials.py).
    n_cells = n_cells_x * n_cells_y
    cell_to_link = np.zeros((len(edges), n_cells))
    for e, cells in enumerate(cells_of_edge):
        for cell in cells:
            cell_to_link[e, cell] = 3.0 / 8.0

    return nodes, np.array(edges), np.array(types), cell_to_link


def lumped_masses(n_cells_x, n_cells_y, cell_size, thickness, density):
    """Each cell's mass rho*t*h^2 is split equally between its 4 corner nodes."""
    mass = np.zeros((n_cells_x + 1) * (n_cells_y + 1))
    m_cell = density * thickness * cell_size ** 2
    for ix in range(n_cells_x):
        for iy in range(n_cells_y):
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                mass[node_index(ix + dx, iy + dy, n_cells_y)] += m_cell / 4
    return mass


def build_beam(config, n_cells_x=None, n_cells_y=None, load="right_edge"):
    """Create the clamped cantilever described by `config`.

    load = "right_edge": total config.tip_load shared by the right-edge nodes
                         (the shape-matching problems).
    load = "top_right" : unit force on the upper-right node only
                         (the fracture demo scales it by load_rate * t).
    """
    ncx = config.n_cells_x if n_cells_x is None else n_cells_x
    ncy = config.n_cells_y if n_cells_y is None else n_cells_y
    if ncy % 2 != 0:
        raise ValueError("n_cells_y must be even so a row of nodes lies on the centerline.")
    h = config.beam_height / ncy
    length = h * ncx       # keeps cells square if n_cells_x is overridden

    nodes, edges, types, cell_to_link = build_lattice(ncx, ncy, length, config.beam_height)
    mass = lumped_masses(ncx, ncy, h, config.thickness, config.density)

    fixed = np.zeros(nodes.shape, bool)
    fixed[np.isclose(nodes[:, 0], 0.0), :] = True              # clamp left edge

    load_nodes = np.zeros(nodes.shape)
    if load == "right_edge":
        right = [node_index(ncx, iy, ncy) for iy in range(ncy + 1)]
        share = np.ones(len(right))
        share[[0, -1]] = 0.5                  # end nodes carry half (trapezoid rule)
        load_nodes[right, 1] = -config.tip_load * share / share.sum()
    elif load == "top_right":
        load_nodes[node_index(ncx, ncy, ncy), 1] = -1.0
    else:
        raise ValueError(f"unknown load {load!r}")

    # Placeholder stiffness/damping; the real values come from the design.
    model, _ = prepare(nodes, edges, k=1.0, c=0.0, mass=mass, fixed=fixed)

    beam = Beam()
    beam.config = config
    beam.n_cells_x, beam.n_cells_y = ncx, ncy
    beam.n_cells = ncx * ncy
    beam.cell_size = h
    beam.length = length
    beam.model = model
    beam.nodes = nodes
    beam.edges = edges
    beam.link_type = types
    beam.n_links = len(edges)
    beam.cell_to_link = jnp.asarray(cell_to_link)
    beam.link_weight = jnp.asarray(cell_to_link.sum(axis=1))
    beam.center_nodes = np.array([node_index(ix, ncy // 2, ncy) for ix in range(ncx + 1)])
    x_center = nodes[beam.center_nodes, 0]
    beam.xi = jnp.asarray((x_center - x_center.min()) / (x_center.max() - x_center.min()))
    beam.y_center0 = float(nodes[beam.center_nodes[0], 1])
    beam.load_nodes = jnp.asarray(load_nodes)
    beam.cell_centers = np.array([[(ix + 0.5) * h, (iy + 0.5) * h]
                                  for ix in range(ncx) for iy in range(ncy)])
    return beam


def choose_time_step(beam, k_stiffest, config):
    """dt and number of steps for config.sim_time.

    k_stiffest is the link stiffness of the all-stiff design. Stiffening any
    link can only raise the highest natural frequency, so this design has the
    smallest stable time step of all designs (binary or continuous).
    If config.dt is None, dt = dt_safety * that limit.
    Returns (dt, n_steps, dt_limit).
    """
    params = {"k": jnp.asarray(k_stiffest), "c": jnp.zeros(beam.n_links)}
    dt_limit, _ = stable_dt_estimate(beam.model, params, iters=200)
    dt_limit = float(dt_limit)
    dt = config.dt if config.dt is not None else config.dt_safety * dt_limit
    n_steps, dt = num_steps((0.0, config.sim_time), dt)
    return dt, n_steps, dt_limit
