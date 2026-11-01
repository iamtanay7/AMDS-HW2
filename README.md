# MCEN 4155/5155 HW2: Optimizing a soft/stiff cantilever

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamtanay7/AMDS-HW2/blob/master/HW2_starter.ipynb)

You are given a mass-spring-damper simulator of a 2-D cantilever beam made of
square cells. Each cell is either **soft** or **stiff**. Your job is to choose
the material of every cell so that the bent beam's centerline matches a target
shape, and to compare three optimizers: random search, a genetic algorithm,
and a gradient-based method.

**Everything you do happens in one notebook: `HW2_starter.ipynb`.** The `.py`
files are the simulator behind it; you do not need to edit them.

---

Choose **ONE** of the two options below (not both). Both use the same notebook and
the same simulator files from this repository. **Option 1 (Colab) is recommended**: it
needs no Python installation on your computer.

This assignment uses Python and JAX, whereas earlier work in the course used MATLAB.
You are not expected to arrive as a Python expert; additional office hours are
available as needed (see Canvas). Recommended resource: the official [Python tutorial](https://docs.python.org/3/tutorial/).
[NumPy for MATLAB users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)
is a handy side-by-side table of MATLAB and NumPy commands.

## 1. OPTION 1: Google Colab (recommended)

1. Click the HW2 Colab link (on Canvas, or the "Open in Colab" badge above) while
   signed in to your Google account.
2. Choose *File → Save a copy in Drive*. This saves a copy of the notebook into
   **your Google Drive** (online, in *My Drive → Colab Notebooks*), **not** onto your
   PC. Colab opens the copy in a new tab: work only in that copy (the original link
   is read-only). Colab saves your copy automatically.
3. Run the setup cell (section 2) first, every session. It downloads this
   repository into `hw2_backend/` and checks the packages; Colab already has JAX,
   NumPy and Matplotlib, so usually nothing is installed.
4. Work through the notebook from top to bottom.
5. Use `QUICK_MODE = True` (section 13) while you debug your optimizers.
6. For the final experiment, set `QUICK_MODE = False` and run the targets one at a
   time (`TARGETS_TO_RUN = ["linear"]`, then `["cosine"]`, then `["sigmoid"]`).
   Finished work is skipped when you run the cell again.
7. Download `HW2_submission.zip` (section 15) and your notebook before ending the
   session.

Colab's working files are temporary: when the runtime disconnects, files such as
`results/` and `submission/` are deleted (your notebook in Drive is safe). To keep
results across sessions, set `USE_DRIVE = True` in the setup cell; they are then
saved in the Drive folder `HW2_outputs`. Leave the runtime type on CPU; a GPU does
not help.

## 2. OPTION 2: your own computer (Jupyter or VS Code)

You need Python 3.10 or newer. You do **not** need to know Git.

**Get the files (once).**

1. Open the course GitHub repository link (on Canvas) in your web browser.
2. Either:
   * **No Git needed:** click the green **Code** button, then **Download ZIP**. Extract
     the ZIP file (Windows: right-click → *Extract All*; macOS: double-click).
   * **If you have Git:** `git clone <repository link>` in a terminal.
3. You now have one folder (e.g. `AMDS-HW2-master`) that contains
   `HW2_starter.ipynb` next to the simulator `.py` files.

**Install the packages (once).** Open a terminal *in that folder* (Windows: open the
folder in File Explorer, type `powershell` in the address bar and press Enter;
macOS: right-click the folder → *New Terminal at Folder*), then run:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

This installs JAX, NumPy, Matplotlib and JupyterLab into `.venv`. Everything runs
on a normal laptop CPU; no GPU is needed.

**Each time you work on the homework**, open a terminal in that folder and run:

```bash
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
jupyter lab
```

Open `HW2_starter.ipynb` and run the cells in order (`Shift + Enter`), or
*Kernel → Restart Kernel and Run All Cells*. Always open the notebook from inside
this folder and do not move it out: the setup cell finds the simulator files next to
it. (VS Code: *File → Open Folder* on this folder, open the notebook, and pick the
`.venv` Python as the kernel.)

Tip (Git users): save your work under a new name in the same folder
(e.g. `HW2_yourname.ipynb`), so `git pull` can update the course files without
touching your notebook. Your outputs (`results/`, `submission/`,
`HW2_submission.zip`) are ignored by git.

## 3. What is in the notebook

| Section | What happens | You edit? | Who |
| --- | --- | --- | --- |
| 1-3 | introduction, **setup cell** (run first), the parameters in `config.py`; set `COURSE` | `COURSE` only | all |
| 4 | build the 20 x 4 cell beam: clamp, load, centerline | no | all |
| 5 | fracture demo (worked example) | no | all |
| 6 | material assignment: E → link stiffness and damping | no | all |
| 7 | the three target shapes | no | linear, cosine: all; sigmoid: 5155 |
| 8 | the objective J and the function-evaluation convention | no | all |
| 9 | **random search** | ✏️ yes | all |
| 10 | **genetic algorithm** | ✏️ yes | all |
| 11 | **gradient-based optimization** (JAX) | ✏️ yes | 5155 required, 4155 optional |
| 12 | continuous-to-binary projection analysis | add your own comparisons | 5155 required, 4155 optional |
| 13-14 | repeated runs (resumable) and convergence plots | set `QUICK_MODE = False` for the final run | all |
| 15 | checklist; writes `summary.csv` and `HW2_submission.zip` | no | all |

Before you write any code, *Restart Kernel and Run All Cells* already runs to the
end: unfinished optimizers print `TODO: complete ...` instead of crashing.

**Expected output of the first sections** (the last digit may differ; times
depend on your computer):

Section 4:
```
cosine target: 20x4 cells, 105 nodes, 344 links, 80 design variables (cell); A = 15.00 mm; dt = 3.2e-05 s (0.69 of stability limit), 6251 steps
centerline: 21 nodes at y = 10 mm
```
Section 5:
```
First failure time    : 0.1325 s
Load at first failure : 2.649 N
First link(s) to fail : [4] (horizontal)
Number of failed links: 15 of 208
```
Section 8:
```
all soft : J = 3.139e-03 m^2, RMS error = 373.5% of A
all stiff: J = 4.791e-05 m^2, RMS error =  46.1% of A
random   : J = 1.880e-03 m^2, RMS error = 289.0% of A
```
Errors above 100% are expected here: the soft material is 500 times softer than
the stiff one, so mostly-soft designs droop far past the 15 mm target. Good designs
need a careful layout of the two materials, which is what the optimizers are for.

Sections 9-11 print `[TODO]` until you have written each optimizer, then `[PASS]`.

**Run times.** `QUICK_MODE = True` (2 runs x 100 evaluations) takes about 2-5
minutes for all targets; use it while debugging. The assignment setting
(`QUICK_MODE = False`: 5 runs x 3000 evaluations per method and target) costs
about 6 minutes per run of all three methods on a laptop (random search ~1 min,
GA ~1 min, gradient method ~3.7 min) and roughly 8 minutes on free Colab. That
is about 30 minutes per target on a laptop and 40 minutes on Colab for 5155
(three targets), and about a third of that for 4155 (random search and GA only,
two targets). Run one target per sitting (`TARGETS_TO_RUN`); finished work is
saved immediately and skipped when you run the cell again, so an interrupted run
can simply be restarted. The very first simulation in a session takes a few
extra seconds to compile.

## 4. The problem in brief

* **Beam:** 20 x 4 square cells (100 mm x 20 mm, 10 mm thick), left edge
  clamped. A total downward force of 1 N on the right-edge nodes ramps up over
  0.05 s and is held until T = 0.2 s, when the beam has settled.
* **Design:** one value per cell (80 in total), numbered `ix * 4 + iy`.
  Binary `b`: 0 = soft (E = 0.02 MPa), 1 = stiff (E = 10 MPa),
  both effective lattice moduli (contrast 1:500). Continuous `z` in
  [0, 1] for the gradient method: `E = E_soft + z (E_stiff - E_soft)`. Always
  clip z to [0, 1].
* **Centerline:** the row of 21 nodes at mid-height. Their vertical
  displacement at t = T is compared with the target at each node's undeformed
  position ξ = x₀/L.
* **Objective:** `J = mean((y_centerline - y_target)^2)` [m²].
* **Targets** (A = 0.15 L = 15 mm, β = 8): linear `-A ξ`; cosine
  `-(A/2)(1 - cos πξ)`; sigmoid `-A (s(ξ) - s(0))/(s(1) - s(0))` with
  `s(u) = 1/(1 + exp(-β(u - 0.5)))`. The cosine and sigmoid need an S-shaped
  (inflected) centerline, which only a heterogeneous layout can produce; the sigmoid
  is the hardest. Compare the methods with each
  other, not with zero error.
* **Evaluation counting:** `evaluate` = 1, `evaluate_batch` = 1 per design,
  `value_and_grad` = 1. Convergence plots show the best-so-far J of a *binary*
  design vs. function evaluations, with the mean over the runs and the min-max range
  over the runs (run-to-run spread, not a confidence interval).

## 5. Files

| File | Purpose |
| --- | --- |
| `HW2_starter.ipynb` | **the homework**: everything you run and write |
| `config.py` | every physical and numerical parameter |
| `objective.py` | `ShapeProblem`: design → simulation → J, with evaluation counting |
| `materials.py` | E → link stiffness, z → E, thresholding |
| `beam_setup.py` | builds the lattice: nodes, links, masses, clamp, load, centerline |
| `simulator.py` | time stepping (semi-implicit Euler) |
| `target_profiles.py` | the three target shapes |
| `fracture.py` | force-based link fracture and the fracture demo |
| `plotting.py` | drawing beams, designs and the fracture demo |
| `convergence.py` | repeated runs, best-so-far curves, convergence plots |
| `run_all_checks.py` | sanity checks of the simulator |
| `_paths.py` | lets the files find `msd_jax/` and turns on double precision |
| `msd_jax/` | core mass-spring-damper simulator (same physics as the course MATLAB code) |

Do not edit the `.py` files. To try other parameter values, make a modified copy
of the configuration in the notebook:
`replace(CONFIG, tip_load=0.8)` (see notebook section 3).

`python run_all_checks.py` checks the simulator from the terminal. Its output is
`6 passed, 0 failed, 4 TODO`; the four TODO lines are the optimizers, which the
command line cannot see because they live in the notebook. The notebook runs
the same four checks on your functions.

## 6. Physics notes (short)

* **Links.** Every link is an axial spring plus a damper (c = β k). Lengths and
  directions are recomputed every step, so large rotations are handled.
* **Material to stiffness.** E is an *effective lattice modulus*. Each cell
  gives 3/8 x E x thickness to each of its 4 sides and 2 diagonals; a side shared
  by two cells gets both contributions. Under a small, uniform strain the network
  then stores exactly the energy of an isotropic elastic sheet with modulus E and
  Poisson ratio 1/3. It is still a coarse spring lattice (4 cells high), not a
  finite-element model: in bending it is about 8% stiffer than a continuum beam
  with the same E.
* **Settling.** A drag term (`alpha_damping`) lets the beam settle, so the shape
  at t = 0.2 s is the static equilibrium shape.
* **Time step.** The method is explicit, so dt must stay below a stability limit
  set by the stiffest design. `dt = 3.2e-5 s` is 0.69 of that limit. If you
  change E_stiff, the density or the mesh, use `dt=None` so it is chosen
  automatically.
* **Fracture demo.** A link fails when its elastic force |k (L − L0)| exceeds
  F_fail (a force criterion). A failed link keeps its place in the arrays but
  carries no spring or damping force. The optional stiffness-scaled threshold
  F_fail ∝ k is a fixed-elongation rule; a fixed-strain rule would need
  F_fail ∝ k · L0.

## 7. Troubleshooting

* **`ModuleNotFoundError: jax`** (or numpy, matplotlib): the notebook is not
  using the `.venv` Python. Start `jupyter lab` from the activated environment,
  or pick the `.venv` kernel in VS Code.
* **A cell is very slow or the computer swaps.** Each simulation needs about
  200-400 MB. On an 8 GB laptop, close other heavy programs (browsers with many
  tabs) during the full run.
* **Something looks broken after many edits.** *Kernel → Restart Kernel and Run
  All Cells*: the notebook is designed to run top to bottom from a clean start.
