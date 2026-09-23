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
    generators (`generate_line` — with an optional variable cross-sectional
    area profile to cheaply approximate a real gut's changing diameter,
    `generate_rectangle`, `generate_box`) cover common cases without needing
    an external mesh file.
  - `src/transport.py` + `src/boundary.py`: dimension-agnostic diffusion
    (two-point flux) and advection (upwind) operators with per-boundary-tag
    Dirichlet/Neumann(no-flux default)/Outflow conditions.
  - `src/solve.py`: operator-split time integration (reaction in log-space via
    `shared/monod_core`, transport implicit in linear space), reusing the
    exact same kinetics as the well-mixed case per grid cell.
  - Validated per `docs/spatial_pde_spec.md`'s validation plan: zero-diffusion
    reduces exactly to the well-mixed ODE, pure diffusion conserves mass on
    1D/2D/3D meshes alike (`spatial_pde/tests/`).
  - **Known v1 limitations** (see `docs/spatial_pde_spec.md` "Status" section
    for the reasoning): first-order (Lie) operator splitting, not Strang;
    two-point-flux diffusion assumes a reasonably orthogonal/Delaunay-quality
    mesh; the lag state `q` is cell-autonomous and does not itself diffuse.
- **`curriculum_nn/`** — still spec-only, no implementation (unstarted).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest   # runs shared/tests + spatial_pde/tests
# add torch/jax + graph-nn library of choice once curriculum_nn's
# architecture is chosen (see docs/curriculum_nn_spec.md).
```

## Status

`shared/monod_core/` (configurable kinetics) and `spatial_pde/` (mesh-general
PDE solver) are implemented and tested. `curriculum_nn/` is still spec-only.
