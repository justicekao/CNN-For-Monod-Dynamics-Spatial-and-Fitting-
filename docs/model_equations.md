# Model equations (shared source of truth)

Both sub-projects extend the same core reaction kinetics. This doc defines
that core once so `curriculum_nn/` and `spatial_pde/` don't drift apart.

## 1. Base form: shared-denominator Monod growth

For one strain with growth rate `r_a` and half-saturation `K_a` on each of
several resources `a` (metabolites, and optionally a toxin as a
negative-growth "resource"), per-capita growth is:

```
growth = sum_a [ r_a * (x_a / K_a) ] / [ 1 + sum_a (x_a / K_a) ]
```

i.e. all resources in the set compete for **one shared uptake capacity**
(the `1 + sum(...)` denominator is shared). This is the "statistical
mechanics" grounding from the source paper
(`Gut_Microbiome_Toxin_Monod_Aug_12.pdf` — kept for reference, not copied into
this repo; re-derive from there if the exact derivation is needed). Per-resource
consumption follows the same saturation term:

```
uptake_a = c_a * (x_a / K_a) / [ 1 + sum_a (x_a / K_a) ]
```

`standalone_two_strain_system.py` implements a variant with a `SHARE_UPTAKE`
flag: metabolites can either share one denominator (`SHARE_UPTAKE=True`, the
form above) or each saturate independently (`SHARE_UPTAKE=False`,
`denom = 1 + x_a/K_a` computed separately per resource). A toxin is always
handled with its own independent denominator (its kill effect doesn't compete
with metabolites for uptake capacity) — see `_strain_growth_and_uptake()`.

## 2. Population and resource ODEs

For strain `i` with population `N_i`:

```
dN_i/dt = N_i * (growth_i - mortality_i) + (population-level transfer terms)
```

For a metabolite or toxin pool `x_a`, supplied externally and diluted/decayed,
drawn down by every strain that consumes it:

```
dx_a/dt = supply_a - dilution_a * x_a - sum_i [ N_i * uptake_{i,a} ]
```

(sign flips for toxins that strains *secrete* rather than consume — see
`s_secretion` in the MATLAB app.)

## 3. Full general form (target complexity — `FitMicrobialSystemApp.m`)

For `S` strains, `M` metabolites, `T` toxins, the MATLAB reference defines
these per-(strain, resource) matrices, which `curriculum_nn/` should treat as
the canonical parameter set to eventually predict in full:

| Matrix | Shape | Meaning |
|---|---|---|
| `r` | S×M | growth rate of strain i on metabolite a |
| `k` | S×M | half-saturation of strain i on metabolite a |
| `c` | S×M | consumption rate of strain i on metabolite a |
| `P` | S×T | kill rate of toxin t on strain i |
| `K` | S×T | half-saturation of strain i's toxin-t response |
| `s` | S×T | secretion rate of toxin t by strain i |
| `Z` | S×S | pairwise transconjugation/transfer rate (strain i → strain j) |
| `delta` | S×1 | mortality rate of strain i |
| `g` | (see .m) | growth-modifying term, check `FitMicrobialSystemApp.m` for exact use |
| `m_supply` | M×1 | external supply rate of metabolite a |
| `D_dilution` | M×1 | dilution/removal rate of metabolite a |
| `m_toxin_supply` | T×1 | external supply rate of toxin t |
| `d_toxin_decay` | T×1 | decay rate of toxin t |

The MATLAB `explicitSystemODE` right-hand side (already extracted, don't
re-derive from scratch):

```matlab
X_s = repmat(x', p.S, 1) ./ (max(1e-4, p.k) + 1e-4);
denom_x = 1 + sum(X_s, 2);
Y_s = repmat(tox', p.S, 1) ./ (max(1e-4, p.K) + 1e-4);
denom_y = 1 + sum(Y_s, 2);
dN = N .* (sum(p.r .* X_s, 2) ./ denom_x - sum(p.P .* Y_s, 2) ./ denom_y - p.delta) + conjugationInflow;
dx = p.m_supply - p.D_dilution .* x - sum((p.c_consumption .* X_s ./ denom_x) .* repmat(N, 1, p.M), 1)';
dtox = p.m_toxin_supply - p.d_toxin_decay .* tox + sum((p.s_secretion .* Y_s ./ denom_y) .* repmat(N, 1, p.T), 1)';
```

Notes for reimplementation in Python/NN context:
- Matrices are stored log10-transformed with an offset internally in the
  MATLAB app (`Mats.r = 10.^(reshape(...) - 1)`, etc.) — worth keeping a
  positivity-preserving parameterization (log-space or softplus) in the NN
  output layer too, both for numerical stability and because it mirrors how
  the ODE itself is integrated (see §4).
- Sparsity: the MATLAB app zeros all but the top-`w_vec(i)` largest-magnitude
  entries per matrix. This is a strong prior (most strain/resource pairs
  don't actually interact) that the NN curriculum should probably exploit —
  e.g. an L1 penalty or explicit top-k masking on predicted matrices — rather
  than trying to learn dense matrices from scratch at high dimension.
- `conjugationInflow` and `Z` generalize the ad-hoc `TRANSFER_RATE * Nd * Nr`
  term in `conjugation_unified_model.py` to arbitrary strain pairs.

## 4. Numerical integration notes (apply to both projects)

- Integrate in **log-space** (`d(logN)/dt = dN/dt / N`) so populations and
  concentrations can never go negative and stiffness near zero is avoided.
  Both reference scripts do this; `spatial_pde/` should too, per grid point.
- At realistic population scales (~1e8 cfu/mL), consumption/transfer rate
  constants must be scaled down hard (~1e-8 to 1e-13 in the validated
  examples) or the system becomes numerically stiff/unstable. This isn't a
  solver bug — it reflects genuine boom-then-bust dynamics when a population
  starts far from its true equilibrium. Use `scipy.integrate.solve_ivp` with
  `method="LSODA"` and tight tolerances (`rtol=1e-6..1e-9`, `atol=1e-9..1e-12`)
  as the two reference scripts do.
- Two populations with *identical* growth/mortality parameters, one
  continuously converting into the other (e.g. Recipient → Transconjugant),
  exhibit near-neutral competition and converge extremely slowly. If
  Project A's synthetic data generator produces near-degenerate parameter
  sets like this, expect the corresponding trajectories to be uninformative
  for fitting (flat, slowly-drifting) — consider excluding or down-weighting
  such cases in training data, or explicitly testing whether the NN can
  detect this degeneracy (it's a real edge case the MATLAB app's
  `lsqnonlin`-based fitting would also struggle with).

## 5. Non-standard mechanisms present in the validated examples

These aren't part of the "pure" Monod form in §1–3, but they're validated
against real data (`conjugation_unified_model.py`) and Project A/B need to
decide explicitly whether to support them or scope them out:

- **Lag/readiness state `q`**: a per-population state in [0,1], `dq/dt =
  Q_RATE*(1-q)`, multiplying realized growth (`growth_effective = growth * q`)
  and partially damping mortality (`mort_eff = mortality*(MORT_FLOOR +
  (1-MORT_FLOOR)*q)`). `q0` is deterministically computed from the
  population's own initial density (`q0 = min(1, N0/Q_REFERENCE_DENSITY)`),
  not a free parameter — this is what let one parameter set reproduce two
  experimental datasets differing only in initial conditions. If Project A's
  curriculum includes systems calibrated this way, `q` and `Q_RATE`/
  `Q_REFERENCE_DENSITY` need to be part of the parameter vector it predicts
  (or the curriculum should stay in "pure Monod, no lag" territory and treat
  lag as future work — recommend the latter for the first curriculum stages,
  see `curriculum_nn_spec.md`).
- **Contact-based, non-depleting transfer** (plasmid conjugation): adds
  `rate * N_donor * N_recipient` to a product population's growth term
  without subtracting from either parent population (donor and recipient
  aren't consumed by the act of conjugating). This generalizes to the `Z`
  matrix / `conjugationInflow` in the MATLAB app.
- **Private vs. shared resource pools**: in `conjugation_unified_model.py`
  each population (Recipient, Donor, Transconjugant) has its *own* metabolite
  pool (`Ar`, `Ad`, `At`) rather than all strains competing for one shared
  pool. This was a deliberate calibration choice (prevents one strain's
  growth from crashing another's population when independently matching
  target dynamics) — not always biologically motivated, so treat it as a
  configurable topology (shared pool vs. per-strain pool vs. arbitrary
  strain→pool assignment graph) rather than baking in one assumption.
