# Project B: Spatial Monod PDE (gut microbiome spatial model)

## Status: implemented (v1) in `spatial_pde/src/`

The finite-volume mesh/transport/reaction/solver framework described below is
implemented and tested (`spatial_pde/tests/`) — see the end of this file for
what's done, the modeling choices actually made (vs. left open below), and
known v1 limitations. The rest of this document is kept as the original
design spec; read it for the reasoning behind those choices.

## Goal

Extend the validated well-mixed (ODE) Monod system into a spatial
reaction-diffusion PDE over **1D, 2D, or 3D geometry** — not just a straight
gut-axis line — so strain, metabolite and toxin distributions can vary in
space rather than being single lumped pools. A 1D line with a variable
cross-sectional area profile cheaply approximates a real gut's changing
diameter; a full unstructured 2D/3D mesh (imported from Gmsh/CAD/a segmented
scan, or a structured generator) handles genuinely non-tube geometry, like a
bespoke experimental vessel or a branched/non-uniform gut segment.

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

## File layout inside `spatial_pde/` (as implemented)

```
spatial_pde/
├── src/
│   ├── mesh.py             # 1D/2D/3D finite-volume mesh geometry + generators +
│   │                        # Mesh.from_meshio() for real/arbitrary geometry
│   ├── boundary.py          # Dirichlet / Neumann(no-flux default) / Outflow
│   ├── transport.py         # dimension-agnostic diffusion (TPFA) + advection (upwind)
│   ├── reaction.py          # thin wrapper batching shared/monod_core across mesh cells
│   └── solve.py             # operator-split time integration (FieldSpec, SpatialConfig)
├── tests/
│   ├── test_zero_diffusion_matches_ode.py   # validation plan #1
│   ├── test_mass_conservation.py            # validation plan #2 (1D/2D/3D)
│   └── test_mesh_import.py                  # Mesh.from_meshio() geometry correctness
└── data/                   # any spatial reference data, if/when available
```

(The original plan called the mesh module `grid.py`, scoped to 1D; it became
`mesh.py` once genuine 2D/3D unstructured-mesh support was in scope, since
"grid" undersold what it now does.)

## Status: implementation notes and known v1 limitations

Modeling decisions the sections above left open, as actually resolved:

- **Numerical approach**: option (a), operator splitting — reaction
  integrated in log-space per cell (reusing `shared/monod_core` unmodified,
  batched across cells), diffusion+advection integrated implicitly
  (backward Euler) in linear space. `solve_spatial(..., splitting=...)`
  supports both first-order (`"lie"`: reaction then transport, each over the
  full timestep — cheaper, one transport solve/step) and second-order
  (`"strang"`: half-transport / full-reaction / half-transport — costs one
  extra transport solve/step, but is measurably more accurate at the same dt;
  see `test_splitting_accuracy.py`). Default is `"lie"`; use `"strang"`
  whenever time-accuracy at a practical (not vanishingly small) dt matters.
- **Discretization**: finite-volume (not finite-difference), on a simplicial
  mesh (1D line segments, 2D triangles, or 3D tetrahedra) — chosen
  specifically because it generalizes to unstructured/imported geometry and
  guarantees exact mass conservation for the diffusion/advection terms
  (verified in `test_mass_conservation.py`), which finite-difference on a
  non-uniform grid does not.
- **Diffusive flux**: a two-point flux approximation (TPFA) — flux between
  two cells is `D * face_area / centroid_distance * (u_i - u_j)`. This
  assumes the line between adjacent cell centroids is reasonably
  perpendicular to their shared face, which holds for the built-in
  generators and for Delaunay-quality unstructured meshes. A badly skewed
  imported mesh would need a non-orthogonal flux correction this v1 doesn't
  implement — worth checking mesh quality (or adding that correction) before
  trusting results on a very irregular real-geometry import.
- **Boundary conditions**: implemented per-boundary-tag, per-field
  (`spatial_pde/src/boundary.py`): `Dirichlet(value)`, `Neumann(flux=0.0)`
  (no-flux is the default, per this doc's own recommendation above), and
  `Outflow()` (zero-gradient/advective exit). A `Dirichlet` boundary where
  the local velocity points inward is treated as an inflow at that
  concentration (the proximal-end "continuous intake" case above); an
  `Outflow`/`Neumann` boundary with unexpected inflow is treated as
  zero-concentration inflow, since no value was specified for what's
  entering.
- **The lag/readiness state `q`** (docs/model_equations.md §5) is treated as
  a per-cell, cell-autonomous property: it evolves via the local reaction
  step only and is NOT itself transported between cells. This matches the
  validated well-mixed model (`q` lives per-population, not per-place); a
  group wanting readiness/adaptation to spread spatially (e.g. a
  quorum-sensing-like signal) would need to add `q` as its own transported
  field, which this module does not currently do.
- **Geometry**: `Mesh.from_meshio()` is the general path for real/arbitrary
  shapes; it does not infer boundary semantics (which face is "proximal
  inlet" vs. "gut wall") from an imported mesh's own physical-group tags, on
  the theory that no single convention generalizes across meshing tools —
  instead you supply a `boundary_tag_fn(face_centroid) -> tag_name`
  classifier, which is usually a one-line function based on position. The
  built-in generators (`generate_line`, `generate_rectangle`, `generate_box`,
  `generate_tube`) apply sensible default tags automatically.
- **`generate_tube(length, radius_profile, ...)`**: a genuinely 3D swept
  tube (solid, varying-radius lumen cross-section) for approximating a real
  gut's changing diameter in full 3D, tagged `proximal`/`distal`/`wall`.
  Built via Delaunay tetrahedralization of a structured point cloud, which
  means it fills exactly the CONVEX HULL of that point cloud — an accurate
  mesh for a monotonic or gently-varying radius profile, but a sharply
  pinched profile (a true sphincter narrowing toward zero) gets "bridged
  over" rather than represented, since a convex hull can't hold a
  concavity. For anatomically precise or sharply non-convex gut geometry,
  build the mesh externally (Gmsh, or any `meshio`-readable tool) and use
  `Mesh.from_meshio()` instead, which has no such limitation. See
  `test_tube_mesh.py` for the volume-convergence and boundary-tagging
  checks (a coarse angular resolution under-estimates volume the way an
  inscribed hexagon under-estimates its circle's area — that's expected
  discretization error, not a bug, and it shrinks with `n_theta`).
