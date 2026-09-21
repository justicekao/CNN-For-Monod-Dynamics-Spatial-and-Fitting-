# Project B: Spatial Monod PDE (gut microbiome spatial model)

## Goal

Extend the validated well-mixed (ODE) Monod system into a spatial
reaction-diffusion PDE representing a 1D gut-axis geometry (extensible later
to more realistic geometry), so strain, metabolite and toxin distributions can
vary along the gut rather than being single lumped pools.

## Core formulation

For each state variable (strain population density `N_i(x,t)`, metabolite
concentration `x_a(x,t)`, toxin concentration `tox_t(x,t)`) along a 1D spatial
coordinate `x` (position along the gut, e.g. 0 = stomach/proximal end, L =
distal/colon end):

```
∂N_i/∂t = D_Ni * ∂²N_i/∂x²  -  v * ∂N_i/∂x  +  R_Ni(N, x_state, tox; x, t)
∂x_a/∂t  = D_xa * ∂²x_a/∂x²  -  v * ∂x_a/∂x   +  R_xa(N, x_state, tox; x, t)
∂tox_t/∂t = D_tt * ∂²tox_t/∂x² - v * ∂tox_t/∂x + R_tt(N, x_state, tox; x, t)
```

- `D_*` are diffusion coefficients (motility of bacteria/metabolites/toxins
  through gut contents — likely small for bacteria relative to metabolites,
  worth making this a tunable per-species parameter, not a shared constant).
- `v` is an advective/flow term (peristalsis/gut transit moving contents from
  proximal to distal) — likely dominates over diffusion for the overall
  transport direction; consider whether diffusion is even necessary for a
  first version or whether pure advection + local reaction is a reasonable
  simplification to start from.
- `R_*` is **exactly the local reaction term from the existing ODE model**
  (`model_equations.md` §1-2, or §5 if lag/transfer mechanisms are in scope)
  evaluated using the local state at each point `x` — this is the part that
  must come from `shared/monod_core/`, not be reimplemented here.
- Supply/dilution terms in the original ODE (`m_supply`, `D_dilution`) may
  need reinterpretation spatially: is metabolite supply localized (e.g. food
  intake at the proximal end, host-secreted mucins along the epithelium) or
  uniform along the domain? This is a modeling decision to flag to the user
  rather than assume — start with uniform supply as the simplest default and
  note the assumption explicitly in code comments and any results.

## Boundary conditions

Needs an explicit decision, flagged to the user if not already specified:
- Proximal end (x=0): likely a Dirichlet or flux boundary condition
  representing continuous input (food/metabolite intake, possibly bacterial
  inoculation).
- Distal end (x=L): likely an outflow (zero-gradient / advective outflow)
  condition representing excretion.
- No-flux (zero-gradient) is the standard default for a state with no known
  input/output at a boundary; use that unless there's a reason otherwise.

## Numerical approach

- **Method of lines**: discretize space with a finite-difference (or finite-
  volume, to guarantee mass conservation, which matters here since the model
  has explicit supply/dilution/consumption balance) grid, turning the PDE
  system into a large ODE system (one ODE per state variable per grid point),
  then integrate in time with the same `solve_ivp`/LSODA approach the
  existing ODE reference scripts use.
- **Positivity**: the existing ODE core integrates in log-space specifically
  to guarantee positivity (see `model_equations.md` §4). Diffusion operators
  don't preserve log-space linearity as cleanly as pure reaction terms do
  (the diffusion term becomes nonlinear in log-space), so this needs a
  concrete decision: either (a) integrate the diffusion term in linear space
  and the reaction term's positivity-sensitive parts separately via operator
  splitting (reaction step in log-space, diffusion step in linear space,
  alternating each timestep), or (b) integrate everything in linear space
  with a positivity-preserving scheme (e.g. clip at zero, or a flux-limited
  finite-volume scheme) and accept the reduced stiffness robustness. Option
  (a) (operator splitting) is recommended as the more direct extension of the
  already-validated log-space ODE approach — flag which was chosen and why
  when this is implemented, since it's a real design choice, not a detail.
- **Stiffness**: the well-mixed ODE core is already numerically delicate at
  realistic population scales (see `model_equations.md` §4 — real boom-bust
  dynamics, not just a solver artifact). Adding a spatial dimension multiplies
  the state count by the number of grid points, so expect this to be
  significantly more expensive and possibly stiffer; start with a coarse grid
  (e.g. 10-20 points) and a short domain/time before scaling up, and profile
  before optimizing.

## Validation plan

1. **Zero-diffusion, zero-advection sanity check**: with `D_* = 0` and `v = 0`
   at every grid point, the spatial model at any single grid point should
   reduce exactly to the well-mixed ODE result from `shared/monod_core/` — use
   this as the first automated test (`spatial_pde/tests/`), since it directly
   verifies the reaction term was ported in correctly.
2. **Mass conservation check**: with no supply/dilution/consumption (i.e. only
   diffusion/advection), total mass integrated across the domain should be
   constant (up to boundary flux) — a standard PDE correctness check.
3. **Qualitative gut biology check**: once the above pass, run a scenario with
   spatially-varying initial conditions (e.g. bacteria inoculated near the
   proximal end) and confirm the model produces biologically sensible
   spreading/gradient behavior along the gut axis before attempting any
   quantitative comparison to real spatial microbiome data (if available;
   none of the uploaded experimental spreadsheets appear to be spatially
   resolved — confirm with the user before assuming any are).

## Suggested file layout inside `spatial_pde/`

```
spatial_pde/
├── src/
│   ├── grid.py            # 1D spatial discretization utilities
│   ├── reaction.py        # thin wrapper calling into shared/monod_core
│   ├── transport.py       # diffusion + advection operators
│   ├── solve.py           # method-of-lines integration (operator splitting)
│   └── boundary.py        # boundary condition implementations
├── tests/
│   ├── test_zero_diffusion_matches_ode.py   # validation plan #1
│   └── test_mass_conservation.py            # validation plan #2
└── data/                   # any spatial reference data, if/when available
```
