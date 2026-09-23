"""
solve.py -- the spatial PDE solver: operator splitting between the local
Monod reaction term (shared/monod_core, via reaction.py) and spatial
diffusion+advection (transport.py), on any Mesh (1D/2D/3D alike).

Per docs/spatial_pde_spec.md's "Numerical approach": diffusion is
integrated in LINEAR space (transport.py's operators act on real
concentrations) while the reaction step is integrated in LOG space
(reaction.py, via the shared kinetics module) -- this is the
"operator splitting" option that doc recommends as the more direct
extension of the already-validated log-space well-mixed ODE approach.

solve_spatial()'s `splitting` argument picks first-order (Lie: reaction
then transport, each over the full timestep) or second-order (Strang:
half-transport, full-reaction, half-transport) splitting -- see that
function's docstring for the accuracy/cost tradeoff.

The lag/readiness state q (if present) is treated as a per-cell,
cell-autonomous property -- it does not diffuse or advect between cells,
only evolves via the local reaction step. This matches the validated
well-mixed model (q lives per-population, not per-place) and is a
modeling choice worth flagging if a group later wants "readiness" to
spread spatially (e.g. quorum-sensing-like signaling) -- that would need
q added as its own transported field, which this module does not do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import factorized

from shared.monod_core.kinetics import MonodConfig
from .boundary import BoundaryConditions
from .mesh import Mesh
from .reaction import MeshState, reaction_substep
from .transport import VelocityField, build_transport_operator


@dataclass
class FieldSpec:
    """Per-field (per-species) transport parameters."""

    diffusion: object = 0.0          # scalar or (n_cells,) array
    velocity: VelocityField = None
    bc: BoundaryConditions = field(default_factory=BoundaryConditions)


@dataclass
class SpatialConfig:
    mesh: Mesh
    reaction: MonodConfig
    strain_fields: List[FieldSpec]
    metabolite_fields: List[FieldSpec]
    toxin_fields: List[FieldSpec]

    def __post_init__(self):
        S, M, T = self.reaction.S, self.reaction.M, self.reaction.T
        if len(self.strain_fields) != S:
            raise ValueError(f"expected {S} strain_fields, got {len(self.strain_fields)}")
        if len(self.metabolite_fields) != M:
            raise ValueError(f"expected {M} metabolite_fields, got {len(self.metabolite_fields)}")
        if len(self.toxin_fields) != T:
            raise ValueError(f"expected {T} toxin_fields, got {len(self.toxin_fields)}")

    def all_field_specs(self) -> List[FieldSpec]:
        return self.strain_fields + self.metabolite_fields + self.toxin_fields


@dataclass
class SpatialResult:
    t: np.ndarray
    N: np.ndarray     # (n_steps+1, n_cells, S)
    x: np.ndarray      # (n_steps+1, n_cells, M)
    tox: np.ndarray     # (n_steps+1, n_cells, T)
    q: Optional[np.ndarray]  # (n_steps+1, n_cells, S) or None


def _precompute_implicit_transport(spatial: SpatialConfig, dt: float):
    """One implicit (backward-Euler) linear solve per field, factorized
    once and reused every timestep since dt/D/velocity/bcs are fixed."""
    n = spatial.mesh.n_cells
    d_inv = diags(1.0 / spatial.mesh.cell_volumes)
    solvers, rhs_const = [], []
    for spec in spatial.all_field_specs():
        M_op, b = build_transport_operator(spatial.mesh, spec.diffusion, spec.velocity, spec.bc)
        A = (identity(n, format="csc") - dt * (d_inv @ M_op).tocsc())
        solvers.append(factorized(A))
        rhs_const.append(dt * (d_inv @ b))
    return solvers, rhs_const


def _transport_substep(state: MeshState, solvers, rhs_const, config: MonodConfig) -> MeshState:
    S, M, T = config.S, config.M, config.T
    fields_old = (
        [state.N[:, i] for i in range(S)]
        + [state.x[:, a] for a in range(M)]
        + [state.tox[:, t] for t in range(T)]
    )
    fields_new = [solver(u_old + c) for solver, c, u_old in zip(solvers, rhs_const, fields_old)]

    n_cells = state.N.shape[0]
    N_new = np.stack(fields_new[:S], axis=1) if S else np.zeros((n_cells, 0))
    x_new = np.stack(fields_new[S:S + M], axis=1) if M else np.zeros((n_cells, 0))
    tox_new = np.stack(fields_new[S + M:S + M + T], axis=1) if T else np.zeros((n_cells, 0))
    return MeshState(N=N_new, x=x_new, tox=tox_new, q=state.q)


def solve_spatial(spatial: SpatialConfig, initial_state: MeshState, t_span, dt: float,
                   save_every: int = 1, splitting: str = "lie") -> SpatialResult:
    """splitting="lie" (default): reaction(dt) then transport(dt) each step
    -- first-order accurate in time, cheaper (one transport solve per step).
    splitting="strang": transport(dt/2), reaction(dt), transport(dt/2) --
    second-order accurate in time (the local splitting error is O(dt^3) per
    step vs. O(dt^2) for Lie), at roughly double the transport-solve cost.
    Prefer "strang" whenever time-accuracy at a practical (not vanishingly
    small) dt matters; "lie" is fine once dt is already small relative to
    both the reaction and transport timescales.
    """
    if splitting not in ("lie", "strang"):
        raise ValueError(f"splitting must be 'lie' or 'strang', got {splitting!r}")

    t0, t1 = t_span
    n_steps = int(round((t1 - t0) / dt))

    if splitting == "lie":
        solvers, rhs_const = _precompute_implicit_transport(spatial, dt)
    else:
        solvers, rhs_const = _precompute_implicit_transport(spatial, dt / 2.0)

    state = initial_state
    t = t0
    ts, Ns, xs, toxs, qs = [t], [state.N], [state.x], [state.tox], [state.q]

    for step in range(1, n_steps + 1):
        if splitting == "lie":
            state = reaction_substep(state, spatial.reaction, dt)
            state = _transport_substep(state, solvers, rhs_const, spatial.reaction)
        else:
            state = _transport_substep(state, solvers, rhs_const, spatial.reaction)
            state = reaction_substep(state, spatial.reaction, dt)
            state = _transport_substep(state, solvers, rhs_const, spatial.reaction)
        t = t0 + step * dt
        if step % save_every == 0 or step == n_steps:
            ts.append(t)
            Ns.append(state.N)
            xs.append(state.x)
            toxs.append(state.tox)
            qs.append(state.q)

    q_out = np.stack(qs) if spatial.reaction.lag is not None else None
    return SpatialResult(t=np.array(ts), N=np.stack(Ns), x=np.stack(xs), tox=np.stack(toxs), q=q_out)
