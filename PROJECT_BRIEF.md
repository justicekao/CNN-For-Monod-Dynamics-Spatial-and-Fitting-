# Monod-ML: Neural Fitting + Spatial Extension — Handoff Brief

This repo hands off two follow-on projects that build on an already-validated
ODE core (shared-denominator Monod kinetics for multi-strain / multi-metabolite
/ multi-toxin microbial population dynamics). Both projects should treat the
files in `shared/monod_core/` as ground truth for "what the model is" — do not
redesign the reaction kinetics, only extend how they're fit or where they're
solved.

Read `docs/model_equations.md` first. It is the single source of truth for the
math. Then read whichever of `docs/curriculum_nn_spec.md` or
`docs/spatial_pde_spec.md` matches the task you're picking up.

## Why these two projects, and how they relate

Project A (`curriculum_nn/`) is an *inverse-problem* tool: given a time series
of population/metabolite/toxin trajectories, infer the underlying Monod
parameter matrices (growth rates, half-saturations, consumption rates, etc.).
The MATLAB reference app (`shared/monod_core/FitMicrobialSystemApp.m`) already
does this via nonlinear least squares (`lsqnonlin`) with multi-restart and
sparsity pruning — it works, but it's slow (one dataset at a time, restarted
from scratch) and struggles as the number of strains/metabolites/toxins grows,
because the parameter space grows roughly as O(S×M) for each matrix and
`lsqnonlin` has no way to transfer what it learned from an easy (low-S/M)
problem to a hard (high-S/M) one. The neural network is meant to *amortize*
this: learn a fast, direct (or lightly-refined) mapping from trajectories to
parameters, trained with a curriculum that starts on small systems (2 strains,
3 metabolites — exactly `standalone_two_strain_system.py`'s scale) and
progressively scales up dimensionality, reusing what was learned at each
smaller scale to warm-start or regularize the next.

Project B (`spatial_pde/`) is a *forward* extension: take the validated
reaction kinetics (the growth/uptake/mortality/transfer terms already in
`conjugation_unified_model.py` and `standalone_two_strain_system.py`) and place
them inside a 1D (or later, more complex-geometry) spatial domain representing
the gut, adding diffusion/advection so that strains, metabolites and toxins
can vary along the gut axis instead of being single well-mixed pools. This is
a reaction-diffusion PDE system where the reaction term is exactly the
existing ODE right-hand side.

These two projects can proceed independently, but they share `shared/monod_core/`
(pure functions for the reaction kinetics) so that Project A's synthetic
training data generator and Project B's local reaction term are guaranteed to
be evaluating the *same* biology.

## Repo layout

```
monod-ml-handoff/
├── PROJECT_BRIEF.md              <- this file
├── docs/
│   ├── model_equations.md        <- the shared math (READ FIRST)
│   ├── curriculum_nn_spec.md     <- Project A spec
│   └── spatial_pde_spec.md       <- Project B spec
├── shared/
│   ├── monod_core/                <- reference implementations (ground truth)
│   │   ├── standalone_two_strain_system.py   (minimal 2-strain/3-metab/1-toxin ODE)
│   │   ├── conjugation_unified_model.py      (4-population, plasmid-transfer ODE,
│   │   │                                       validated against real CFU/mL data)
│   │   └── FitMicrobialSystemApp.m           (MATLAB GUI: full explicit
│   │                                           multi-strain/metabolite/toxin fitter
│   │                                           with hidden dims + sparsity — this
│   │                                           defines the FULL parameter-matrix
│   │                                           shape the NN should eventually match)
│   └── notebooks/                 <- exploratory notebooks go here
├── curriculum_nn/                 <- Project A
│   ├── src/
│   ├── tests/
│   └── data/                      <- generated synthetic training data (gitignored)
└── spatial_pde/                   <- Project B
    ├── src/
    ├── tests/
    └── data/
```

## Ground-truth reference files (do not modify without reason)

- `shared/monod_core/standalone_two_strain_system.py` — minimal, dependency-free
  (numpy/scipy/matplotlib) 2-strain / 3-metabolite / 1-toxin shared-denominator
  Monod ODE. This is the cleanest starting point for Project A's smallest
  curriculum stage and for Project B's local reaction kernel.
- `shared/monod_core/conjugation_unified_model.py` — a 4-population system
  (Recipient, Donor, Transconjugant_Recipient, plus resource pools per
  population) with plasmid-conjugation transfer and a lag/"readiness" state
  mechanism (`q`), validated to reproduce two independent real experimental
  CFU/mL datasets from initial conditions alone, using one shared set of
  intrinsic parameters. This is the most biologically-detailed validated
  example and a good mid-size curriculum stage / stress test for both
  projects (it has non-Monod-standard terms: lag state, contact-based
  non-depleting transfer, mass-conserving state transitions — Project A's
  network needs to either handle these or the curriculum needs to scope them
  out explicitly; see the spec).
- `shared/monod_core/FitMicrobialSystemApp.m` — MATLAB reference implementation
  of the FULL general system: S strains, M metabolites, T toxins, with
  parameter matrices `r, k, P, K, c, Z, s, delta, g, m_supply, D_dilution,
  m_toxin_supply, d_toxin_decay`, hidden (unobserved) strains/metabolites/
  toxins, sparsity pruning per matrix, and multi-dataset joint fitting. This
  is the target complexity level Project A's curriculum should be scaling
  *toward* — not necessarily matching every MATLAB feature (hidden dims,
  joint multi-dataset fitting, freeze/zero constraints) on day one, but the
  parameter-matrix shapes and naming here should be treated as canonical.

## Immediate next steps for whichever agent picks this up

1. Read `docs/model_equations.md`.
2. Pick a project (or ask the user which to start with if unspecified).
3. Read that project's spec doc.
4. Start with `shared/monod_core/` as a library: refactor the pure reaction-rate
   functions (currently copy-pasted across the two reference scripts) into a
   small shared module (e.g. `shared/monod_core/kinetics.py`) that both
   `curriculum_nn/` and `spatial_pde/` import, rather than each project
   reimplementing the Monod term. This refactor is low-risk, high-leverage,
   and should happen before either project's core work starts.
