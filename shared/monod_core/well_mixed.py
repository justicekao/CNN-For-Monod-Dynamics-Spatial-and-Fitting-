"""
well_mixed.py -- generic well-mixed (single-compartment) ODE integration
on top of kinetics.reaction_rhs.

Generalizes what standalone_two_strain_system.py and
conjugation_unified_model.py each do ad hoc: integrate in log-space
(populations/concentrations can never go negative; see
docs/model_equations.md Sec. 4) with scipy's LSODA. Any MonodConfig --
including the new toxin-denominator / transfer / lag variants -- can be
run through this one function instead of writing a bespoke script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from .kinetics import MonodConfig, reaction_rhs

_EPS = 1e-8


def pack_state(N0, x0, tox0, q0=None) -> np.ndarray:
    """Concatenates along the LAST axis, so this also packs a batch of
    per-cell states (N0.shape == (n_cells, S), etc.) for the spatial
    solver's reaction sub-step -- see spatial_pde/src/reaction.py."""
    parts = [np.log(np.maximum(np.asarray(N0, dtype=float), _EPS)),
              np.log(np.maximum(np.asarray(x0, dtype=float), _EPS))]
    tox0 = np.asarray(tox0, dtype=float) if tox0 is not None else np.zeros(0)
    if tox0.shape[-1]:
        parts.append(np.log(np.maximum(tox0, _EPS)))
    if q0 is not None:
        parts.append(np.asarray(q0, dtype=float))
    return np.concatenate(parts, axis=-1)


def unpack_state(y: np.ndarray, config: MonodConfig):
    """Split a (packed) state array along its LAST axis.

    Works for a single state (y.shape == (dim,)) or a batch/trajectory
    (y.shape == (..., dim)), which is why simulate_well_mixed() can reuse
    this to unpack an entire solve_ivp trajectory at once.
    """
    S, M, T = config.S, config.M, config.T
    i = 0
    logN = y[..., i:i + S]; i += S
    logx = y[..., i:i + M]; i += M
    logtox = y[..., i:i + T]; i += T
    q = y[..., i:i + S] if config.lag is not None else None
    N = np.exp(logN)
    x = np.exp(logx)
    tox = np.exp(logtox) if T else np.zeros(logtox.shape[:-1] + (0,))
    return N, x, tox, q


def log_space_derivative(N, x, tox, q, config: MonodConfig) -> np.ndarray:
    """d(log-state)/dt for the packed [logN, logx, logtox, q] layout,
    batched over any leading "..." shape -- shared by the well-mixed
    integrator below and the spatial solver's reaction sub-step."""
    rs = reaction_rhs(N, x, tox, q, config)
    parts = [rs.dN / (N + _EPS), rs.dx / (x + _EPS)]
    if config.T:
        parts.append(rs.dtox / (tox + _EPS))
    else:
        parts.append(np.zeros(tox.shape))
    if config.lag is not None:
        parts.append(rs.dq)
    return np.concatenate(parts, axis=-1)


def _rhs(t, y, config: MonodConfig) -> np.ndarray:
    N, x, tox, q = unpack_state(y, config)
    return log_space_derivative(N, x, tox, q, config)


@dataclass
class WellMixedResult:
    t: np.ndarray
    N: np.ndarray     # (n_points, S)
    x: np.ndarray     # (n_points, M)
    tox: np.ndarray   # (n_points, T)
    q: Optional[np.ndarray]  # (n_points, S) or None


def simulate_well_mixed(
    config: MonodConfig,
    N0, x0, tox0=None, q0=None,
    t_span=(0.0, 100.0), n_points: int = 200,
    **solve_ivp_kwargs,
) -> WellMixedResult:
    if tox0 is None:
        tox0 = np.zeros(config.T)
    if config.lag is not None and q0 is None:
        q0 = config.lag.q0(N0)
    y0 = pack_state(N0, x0, tox0, q0)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    kwargs = dict(method="LSODA", rtol=1e-6, atol=1e-9)
    kwargs.update(solve_ivp_kwargs)
    sol = solve_ivp(_rhs, t_span, y0, t_eval=t_eval, args=(config,), **kwargs)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")

    N, x, tox, q = unpack_state(sol.y.T, config)
    return WellMixedResult(t=t_eval, N=N, x=x, tox=tox, q=q)
