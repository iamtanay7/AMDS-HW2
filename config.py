"""All physical and numerical parameters for HW2, in one place.

Units are SI (m, kg, s, N, Pa). Change values here, not inside other files.

To try a different value without editing this file:

    from config import CONFIG
    from dataclasses import replace
    my_config = replace(CONFIG, E_stiff=20e6)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---------------- Beam geometry (2-D lattice, left edge clamped) -------------
    n_cells_x: int = 20          # cells along the beam length
    n_cells_y: int = 4           # cells through the height (must be even -> centerline row)
    beam_length: float = 0.10    # [m]
    beam_height: float = 0.02    # [m]
    thickness: float = 0.01      # [m] out-of-plane thickness (used in E -> k)

    # ---------------- Materials ------------------------------------------------
    # Effective (lattice-equivalent) Young's moduli. See materials.py.
    E_soft: float = 1.0e6        # [Pa]
    E_stiff: float = 10.0e6      # [Pa]
    density: float = 1000.0      # [kg/m^3], same for both materials

    # ---------------- Damping (Rayleigh: C = alpha*M + beta*K) -----------------
    # alpha_damping is a drag on every node, -alpha*m*v, that lets the beam settle
    # to static equilibrium quickly. beta_damping gives each link c = beta*k.
    alpha_damping: float = 100.0  # [1/s]
    beta_damping: float = 2.0e-6  # [s]

    # ---------------- Shape-matching load case ------------------------------------
    # A total downward force tip_load is shared by the right-edge nodes. It ramps
    # smoothly from 0 to full over ramp_time, then is held until sim_time.
    tip_load: float = 1.0        # [N]
    ramp_time: float = 0.05      # [s]
    sim_time: float = 0.20       # [s]
    dt: float = 3.2e-5           # [s] = 0.69 x stability limit; None = automatic
    dt_safety: float = 0.7      # used only when dt is None

    # ---------------- Target centerline shapes ------------------------------------
    target_amplitude_ratio: float = 0.15   # A = ratio * beam_length
    sigmoid_beta: float = 8.0

    # ---------------- Fracture demo -----------------------------------------------
    fracture_n_cells_x: int = 12
    fracture_n_cells_y: int = 4
    fracture_E: float = 5.0e6              # uniform material for the demo [Pa]
    fracture_load_rate: float = 20.0       # [N/s], F(t) = load_rate * t at upper-right node
    fracture_sim_time: float = 0.25        # [s]
    fracture_threshold_mode: str = "fixed"  # "fixed" or "stiffness"
    fracture_force: float = 4.0            # [N] fixed threshold, or value at k = k_ref
    fracture_k_exponent: float = 0.5       # "stiffness" mode: F_fail = F * (k/k_ref)^p
    fracture_tension_only: bool = False    # True: only tensile links can fail

    # ---------------- Optimization defaults ------------------------------------
    eval_budget: int = 1000      # function evaluations per run
    n_seeds: int = 10            # repeated independent runs


CONFIG = Config()
