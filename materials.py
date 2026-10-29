"""Soft/stiff materials and the mapping from Young's modulus E to link stiffness k.

Mapping (2-D lattice of square cells with both diagonals)
---------------------------------------------------------
Give a side link stiffness k_s and a diagonal link stiffness k_d, and stretch
the lattice with a uniform small strain. Its stored energy per unit area
equals that of a 2-D isotropic continuum (plane stress, thickness t) exactly
when

    k_d = k_s / 2,   E * t = (4/3) * k_s,   Poisson ratio nu = 1/3.

So each cell of modulus E contributes 3/8 * E * t to each of its 6 links:
diagonals belong to one cell (k_d = 3/8 E t), and interior side links are
shared by two cells (k_s = 3/4 E t). A side link on the outer boundary has
only one cell and gets 3/8 E t, which is what makes a uniform beam match the
continuum. The cell size h cancels, so k does not depend on the mesh size.

Limitations, stated honestly:
  * nu is fixed at 1/3 (the in-plane, plane-stress value). Central-force
    springs force C12 = C66 (the Cauchy relation), so 1/3 is the only Poisson
    ratio for which this lattice is isotropic. Read as plane strain, the same
    stiffness corresponds to a 3-D solid with nu = 1/4 and modulus 15/16 E.
  * The equivalence is exact only for small, uniform strain. In bending, a
    beam only 4 cells high is about 8% stiffer than a continuum beam of the
    same E (this converges to the continuum value as the mesh is refined).
    Large rotations are simulated exactly by the spring network, but the
    lattice is not a finite-element model and its large-strain behaviour is
    that of springs.
  * Treat E_soft and E_stiff as effective lattice moduli. They behave like
    Young's moduli of a nu = 1/3 material, not like measured data for a
    particular rubber or plastic.

Design variables
----------------
  binary design  b in {0, 1}: 0 = soft, 1 = stiff        (random search, GA)
  continuous     z in [0, 1]: E = E_soft + z (E_stiff - E_soft)  (gradients)
A design has one entry per cell ("pixel") by default, or one per link when
design_level = "link".
"""
import jax.numpy as jnp

import _paths  # noqa: F401  (double precision)


def design_to_modulus(z, config):
    """Linear interpolation E = E_soft + z * (E_stiff - E_soft). Works for 0/1 too."""
    return config.E_soft + z * (config.E_stiff - config.E_soft)


def threshold_design(z):
    """Continuous -> binary: z < 0.5 -> 0 (soft), z >= 0.5 -> 1 (stiff)."""
    return (jnp.asarray(z) >= 0.5).astype(jnp.float64)


def material_to_link_stiffness(E, beam, design_level="cell"):
    """Link stiffness k [N/m] from moduli E [Pa].

    design_level = "cell": E has one value per cell  (length beam.n_cells)
    design_level = "link": E has one value per link  (length beam.n_links);
                           each link uses its own E with the same geometric
                           factor it would have in a uniform beam.
    """
    t = beam.config.thickness
    if design_level == "cell":
        return t * (beam.cell_to_link @ E)
    if design_level == "link":
        return t * beam.link_weight * E
    raise ValueError(f"design_level must be 'cell' or 'link', not {design_level!r}")


def link_damping(k, config):
    """Stiffness-proportional link damping c = beta * k (the beta*K Rayleigh term)."""
    return config.beta_damping * k


def design_size(beam, design_level="cell"):
    return beam.n_cells if design_level == "cell" else beam.n_links
