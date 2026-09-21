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
why it's there. These are ground truth for the reaction kinetics; both new
projects should import/reuse them (after the refactor described in
`PROJECT_BRIEF.md`'s next-steps section) rather than reimplementing the Monod
equations from scratch.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib
# add torch/jax + graph-nn library of choice for curriculum_nn,
# and any additional PDE/sparse-linear-algebra deps for spatial_pde,
# once an architecture/solver approach is chosen (see the spec docs).
```

## Status

Skeleton + specs only — no implementation yet. This repo is meant to be
picked up and implemented from here.
