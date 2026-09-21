# Project A: Curriculum-trained NN for Monod parameter fitting

## Goal

Given a simulated (or eventually real) multi-strain/metabolite/toxin
trajectory, predict the Monod parameter matrices that generated it (see
`model_equations.md` §3 for the target matrix set: `r, k, c, P, K, s, Z,
delta, m_supply, D_dilution, m_toxin_supply, d_toxin_decay`), faster than the
MATLAB app's per-dataset `lsqnonlin` restart loop, and with the ability to
scale to higher dimensionality (more strains/metabolites/toxins) than
`lsqnonlin` handles gracefully.

The core idea requested: **train on small systems first, then use what was
learned to bootstrap fitting on progressively larger systems** — a curriculum
over dimensionality (S, M, T), not just over difficulty within a fixed
dimensionality.

## Why this is non-trivial (the actual research problem)

A parameter matrix like `r` is S×M — its *shape* changes as the curriculum
scales up. A standard fixed-input/fixed-output-size neural network can't
directly transfer weights from an S=2,M=3 problem to an S=5,M=8 problem. This
is the central design question this project needs to answer; some directions
worth evaluating (in rough order of how directly they solve the stated
transfer goal):

1. **Set/permutation-invariant architecture** (e.g. a Deep Sets / Graph
   Neural Network over strains and resources as nodes, with edges = matrix
   entries). A GNN naturally handles variable S, M, T because it operates on
   however many nodes/edges are present — this is likely the strongest fit
   for "train small, deploy large" and should be the default direction unless
   there's a concrete reason to deviate.
2. **Padding/masking to a fixed maximum size** with an explicit "active
   strain/resource" mask, trained across a distribution of active sizes. Simpler
   to implement than (1) but wastes capacity and caps the max system size
   in advance.
3. **Meta-learning / fine-tuning transfer**: train a base network per
   dimensionality tier, initialize the next tier's network from the previous
   tier's weights (where shapes allow, e.g. via padding), fine-tune on the
   new tier. Closer to literal "use lower-dimensional solutions to train for
   larger systems" as stated, but doesn't solve the shape-mismatch problem as
   cleanly as (1).

Recommendation: prototype with (1) (set/graph-based, size-agnostic) since it
most directly satisfies the stated goal without needing an explicit
curriculum-transfer mechanism bolted on — the same network handles all
sizes. If a GNN underperforms or is too slow to prototype, fall back to (2)+(3)
combined. Whoever picks this up should treat the architecture choice as an
open decision to validate empirically, not a given — note this to the user
early with results from a first small experiment before committing further
engineering effort.

## Curriculum stages (dimensionality ladder)

Suggested progression, each stage validated (loss on held-out synthetic data,
and ideally a sanity check against `lsqnonlin` on the same data) before moving
to the next:

1. **Stage 0**: 2 strains, 3 metabolites, 0-1 toxin, no lag/transfer
   mechanisms — i.e. exactly `standalone_two_strain_system.py`'s scope, pure
   §1-2 Monod form. This is the cleanest, fastest-to-generate stage.
2. **Stage 1**: 2-4 strains, 3-6 metabolites, 1-2 toxins, still no lag/transfer
   — scale up S, M, T within the pure Monod form.
3. **Stage 2**: introduce pairwise transfer (`Z` matrix / conjugation-style
   terms), still no lag state — validate against `conjugation_unified_model.py`
   structurally (same transfer mechanism) but without its lag/`q` state.
4. **Stage 3** (stretch goal, may be out of scope for v1): introduce the lag
   state `q` as an additional predicted quantity (`Q_RATE`,
   `Q_REFERENCE_DENSITY` per strain) — needed to fully explain
   `conjugation_unified_model.py`-style data. Flag this explicitly as
   optional/future work if time-constrained; the pure-Monod stages (0-2) are
   the core deliverable.
5. **Stage 4+**: push S, M, T toward the full generality of
   `FitMicrobialSystemApp.m` (hidden/unobserved strains-metabolites-toxins,
   sparsity, joint multi-dataset fitting) — long-term target, not a v1
   requirement.

## Synthetic training data generation

- Build a generator (in `curriculum_nn/src/`) that samples random-but-valid
  parameter matrices at a given (S, M, T) and simulates trajectories using the
  shared kinetics module (see `PROJECT_BRIEF.md` step 4 — refactor
  `shared/monod_core/` into an importable module first).
- "Valid" matters: uniform-random sampling over parameter ranges will mostly
  produce either trivial extinction or numerically stiff/unstable
  trajectories (see `model_equations.md` §4). Reuse the validated equilibrium-
  solving approach from `conjugation_unified_model.py`
  (`_solve_equilibrium()`) to sample parameters that are *guaranteed* to have
  a real, reachable equilibrium at a chosen target population scale, then
  perturb from there — this avoids wasting most of the training budget on
  degenerate simulations.
- Store (trajectory, ground-truth parameters, dimensionality metadata) tuples;
  keep an explicit train/val/test split *by dimensionality tier* so the model's
  ability to generalize to larger, unseen-during-training system sizes can be
  measured directly (this is the actual claim being tested — "lower-dimensional
  solutions help train for larger systems" — so held-out generalization to a
  size the model was never directly trained on, only bootstrapped toward, is
  the key metric).

## Evaluation

- Primary metric: parameter recovery error (per-matrix relative error, and
  matrix-sparsity-pattern recovery — i.e. did it correctly identify which
  entries are ~zero) against known synthetic ground truth.
- Secondary metric: trajectory reconstruction error (simulate forward with
  predicted parameters, compare to input trajectory) — useful since parameter
  estimation from time series is often non-identifiable (multiple parameter
  sets can produce near-identical trajectories); this metric catches "wrong
  parameters that still fit the data," which parameter-error alone won't.
- Benchmark against the MATLAB `lsqnonlin` approach on at least the smallest
  curriculum stage, both for accuracy and wall-clock time, to demonstrate the
  amortization benefit motivating this project.
- Eventually: real data. `Master_s_experimental_Data.xlsx`, `507RawDataLaurenL.xlsx`,
  and `NSERC.xlsx` (uploaded alongside this brief but not copied into the repo —
  check with the user before committing raw experimental data to a repo) are
  candidate real-world validation sets once the synthetic-data pipeline works.

## Suggested file layout inside `curriculum_nn/`

```
curriculum_nn/
├── src/
│   ├── data_gen.py       # synthetic (params -> trajectory) generator per stage
│   ├── model.py          # the NN architecture (GNN/set-based recommended)
│   ├── curriculum.py     # stage progression / training loop / transfer logic
│   ├── train.py          # CLI entry point
│   └── eval.py           # parameter-recovery + trajectory-reconstruction metrics
├── tests/
│   └── test_data_gen.py  # at minimum: generated trajectories are finite,
│                          # bounded, and reproduce known equilibria
└── data/                  # generated datasets (gitignore the large files;
                            # keep a small fixture set for tests)
```
