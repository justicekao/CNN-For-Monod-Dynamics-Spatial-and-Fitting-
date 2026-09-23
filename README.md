# monod-ml-handoff

Handoff repo for two follow-on projects extending a validated Monod-kinetics
microbial population model:

1. **`curriculum_nn/`** — a curriculum-trained neural network that fits Monod
   parameter matrices to complex, high-dimensional multi-strain systems,
   trained by starting on low-dimensional systems and progressively scaling
   up.
2. **`spatial_pde/`** — a spatial (reaction-diffusion PDE) extension of the
   Monod model for a gut microbiome spatial simulation.

**Start here: [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md)**, then
[`docs/model_equations.md`](docs/model_equations.md) for the shared math,
then whichever of [`docs/curriculum_nn_spec.md`](docs/curriculum_nn_spec.md)
or [`docs/spatial_pde_spec.md`](docs/spatial_pde_spec.md) applies.

## What's already validated (don't redo this work)

`shared/monod_core/` contains two working, validated Python reference models
and one MATLAB reference GUI — see `PROJECT_BRIEF.md` for what each one is and
why it's there. These remain ground truth for "what the biology is"; the
configurable kinetics engine below is the refactor of them called for in
`PROJECT_BRIEF.md`'s next-steps section, validated to reproduce both scripts'
exact math (see `shared/tests/test_kinetics.py`).

## What's implemented so far

- **`shared/monod_core/kinetics.py`** — a single configurable reaction-kinetics
  engine (`MonodConfig`) covering the modified-Monod variants a research group
  actually needs to switch between, instead of hand-editing an ODE per variant:
  - toxin saturation terms sharing ONE uptake-capacity denominator with
    metabolites, sharing a denominator only among themselves, or fully
    independent (`ToxinMode`);
  - metabolites sharing one denominator vs. saturating independently, per
    strain (`share_metabolite_uptake`);
  - contact-based, non-depleting transfer/transconjugation between any strain
    triple (`TransferEvent`), generalizing the conjugation model's donor +
    recipient → product term;
  - an optional per-strain lag/"readiness" state `q` that can inherit another
    strain's `q` (`LagConfig`, reproducing the Transconjugant-inherits-
    Recipient's-`q` mechanism exactly).
  - `shared/monod_core/well_mixed.py` runs any `MonodConfig` as a well-mixed
    (single-compartment) ODE, log-space integrated as in both reference
    scripts.
- **`spatial_pde/`** — a working reaction-diffusion-advection PDE solver on
  **1D, 2D, or 3D unstructured meshes**, not just a straight gut-axis line:
  - `src/mesh.py` builds finite-volume geometry from any 1D-line/2D-triangle/
    3D-tetrahedron simplicial mesh, including `Mesh.from_meshio(...)` to
    import a real/arbitrary geometry (a segmented gut scan, a bespoke
    experimental vessel) built in Gmsh/CAD/any tool `meshio` reads. Built-in
    generators cover common cases without needing an external mesh file:
    `generate_line` (optional variable cross-sectional area profile),
    `generate_rectangle`, `generate_box`, and `generate_tube` (a genuinely
    3D swept, variable-radius lumen approximating a real gut's changing
    diameter — see its docstring/`docs/spatial_pde_spec.md` for the
    convex-hull caveat on sharply pinched profiles).
  - `src/transport.py` + `src/boundary.py`: dimension-agnostic diffusion
    (two-point flux) and advection (upwind) operators with per-boundary-tag
    Dirichlet/Neumann(no-flux default)/Outflow conditions.
  - `src/solve.py`: operator-split time integration (reaction in log-space via
    `shared/monod_core`, transport implicit in linear space), reusing the
    exact same kinetics as the well-mixed case per grid cell. Supports both
    first-order (Lie, default) and second-order (Strang) splitting —
    `test_splitting_accuracy.py` confirms Strang is measurably more accurate
    at matched timestep.
  - Validated per `docs/spatial_pde_spec.md`'s validation plan: zero-diffusion
    reduces exactly to the well-mixed ODE, pure diffusion conserves mass on
    1D/2D/3D meshes (including the tube) alike (`spatial_pde/tests/`).
  - **Known v1 limitations** (see `docs/spatial_pde_spec.md` "Status" section
    for the reasoning): two-point-flux diffusion assumes a reasonably
    orthogonal/Delaunay-quality mesh; `generate_tube`'s convex-hull
    construction can't represent a sharply pinched (non-convex) profile
    (use `Mesh.from_meshio` + an external mesher for that); the lag state
    `q` is cell-autonomous and does not itself diffuse.
- **`curriculum_nn/`** — a v1 curriculum-trained parameter-fitting pipeline,
  for **Stage 0-1 only** (pure Monod, no transfer/lag — see
  `docs/curriculum_nn_spec.md`'s "Status" section for what's not done yet):
  - `src/model.py`'s `MonodParameterGNN` is a permutation-EQUIVARIANT
    bipartite message-passing network over strain/metabolite(/toxin) nodes —
    tested to give identical (correctly-permuted) predictions under any
    relabeling of strain or metabolite order, and to run on any (S, M, T)
    with the same weights, which is what lets one model train on Stage 0's
    small systems and keep training, unmodified, on Stage 1's larger ones.
  - `src/data_gen.py` generates synthetic (trajectory, ground-truth params)
    data via a generalized equilibrium-solving trick (any topology, not a
    hand-derived formula per case — see the spec doc) so training data has
    real, reachable equilibria instead of mostly-degenerate random draws.
  - `src/curriculum.py` + `src/train.py` + `src/eval.py`: the training loop
    (one model/optimizer reused across stages), CLI
    (`python -m curriculum_nn.src.train`), and both metrics the spec calls
    for (parameter recovery error, trajectory reconstruction error).
  - **Honest first-experiment result**: the pipeline runs end to end
    (data → train → eval, loss decreases, curriculum-transfer check runs),
    but parameter-recovery accuracy at a quick/small scale is still ~40-60%
    relative error — a working, testable pipeline, not yet an accurate
    fitting tool. See `docs/curriculum_nn_spec.md`'s "Status" section for
    what's worth validating next before investing further.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest   # runs shared/tests + spatial_pde/tests + curriculum_nn/tests
python -m curriculum_nn.src.train --stages stage0 stage1 --n-per-tier 40
```

## Status

`shared/monod_core/` (configurable kinetics) and `spatial_pde/` (mesh-general
PDE solver) are implemented and tested. `curriculum_nn/` has a working,
tested v1 pipeline for Stage 0-1, with accuracy/scale-up work still open
(see its spec doc's "Status" section).
