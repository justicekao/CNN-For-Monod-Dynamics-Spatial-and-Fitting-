"""
reaction.py -- the spatial solver's reaction sub-step: advance the LOCAL
Monod reaction ODE independently at every mesh cell, in log-space (see
docs/model_equations.md Sec. 4), ignoring transport entirely.

This is deliberately thin: kinetics.reaction_rhs() and
well_mixed.log_space_derivative()/pack_state()/unpack_state() already
batch over an arbitrary leading "..." shape, so plugging in
"..." = (n_cells,) is enough to reuse the exact same validated reaction
math (see shared/tests/test_kinetics.py) per grid cell -- nothing about
the reaction term is reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from shared.monod_core.kinetics import MonodConfig
from shared.monod_core.well_mixed import log_space_derivative, pack_state, unpack_state


@dataclass
class MeshState:
    """One snapshot of every field at every mesh cell."""

    N: np.ndarray               # (n_cells, S)
    x: np.ndarray                # (n_cells, M)
    tox: np.ndarray               # (n_cells, T)
    q: Optional[np.ndarray] = None  # (n_cells, S), or None if config.lag is None


def reaction_substep(state: MeshState, config: MonodConfig, dt: float, **solve_ivp_kwargs) -> MeshState:
    """Advance every cell's local reaction ODE by dt independently (no
    inter-cell coupling here -- that's transport.py's job)."""
    n_cells = state.N.shape[0]
    y0 = pack_state(state.N, state.x, state.tox, state.q)  # (n_cells, dim)
    dim = y0.shape[-1]

    def rhs(t, y_flat):
        y = y_flat.reshape(n_cells, dim)
        N, x, tox, q = unpack_state(y, config)
        return log_space_derivative(N, x, tox, q, config).reshape(-1)

    kwargs = dict(method="LSODA", rtol=1e-6, atol=1e-9)
    kwargs.update(solve_ivp_kwargs)
    sol = solve_ivp(rhs, (0.0, dt), y0.reshape(-1), **kwargs)
    if not sol.success:
        raise RuntimeError(f"reaction sub-step failed: {sol.message}")

    y_end = sol.y[:, -1].reshape(n_cells, dim)
    N, x, tox, q = unpack_state(y_end, config)
    return MeshState(N=N, x=x, tox=tox, q=q)
