"""
kinetics.py -- configurable shared-denominator Monod reaction kinetics.

This module is the single, shared implementation of the reaction term
described in docs/model_equations.md. It generalizes the two validated
reference scripts (standalone_two_strain_system.py,
conjugation_unified_model.py) into one configurable engine so that both
curriculum_nn/ and spatial_pde/ evaluate exactly the same biology, and so
that a research group can express the modified-Monod variants they
actually work with (see ToxinMode below) without hand-editing the ODE
right-hand side each time.

Design choices, and why:

- All arrays are LINEAR-space concentrations/populations in this module.
  The "integrate in log-space to keep things positive" trick
  (model_equations.md Sec. 4) is a numerical-integration concern, not a
  kinetics concern, so it lives in the callers (well_mixed.py,
  spatial_pde/src/reaction.py), not here.
- reaction_rhs() is written with numpy broadcasting over an arbitrary
  batch of leading "..." dimensions, so the exact same function computes
  a single well-mixed system's derivative (batch shape ()) and every grid
  cell's local reaction term in the spatial PDE solver (batch shape
  (n_cells,)) with no code duplication.
- A "toxin" is, by convention (matching FitMicrobialSystemApp.m, the
  canonical target form per docs/model_equations.md Sec. 3), always a
  term that is SUBTRACTED from per-capita growth (P is a kill rate). A
  beneficial resource should be modeled as a metabolite instead. This
  differs from standalone_two_strain_system.py's older convention of a
  single signed TOXIN_GROWTH_RATE; see shared/tests/test_kinetics.py for
  the sign mapping used to validate against that script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Sequence, Union

import numpy as np

_EPS = 1e-8

ArrayLike = Union[float, Sequence[float], np.ndarray]


class ToxinMode(IntEnum):
    """How a strain's toxin saturation terms compete for uptake capacity.

    INDEPENDENT: each toxin gets its own denominator (1 + x_t/K_t),
        independent of every other toxin and of the strain's metabolites.
        This is standalone_two_strain_system.py's default behavior.
    SHARED_AMONG_TOXINS: all of a strain's toxins share ONE denominator
        with each other, but still separate from its metabolites.
    COMBINED_WITH_METABOLITES: the toxin(s) join the SAME shared
        denominator as the strain's metabolites (one uptake-capacity pool
        for everything). Requires share_metabolite_uptake=True for that
        strain (see MonodConfig.__post_init__) -- "independent per
        metabolite but toxins share one of them" isn't well-defined.
    """

    INDEPENDENT = 0
    SHARED_AMONG_TOXINS = 1
    COMBINED_WITH_METABOLITES = 2


_TOXIN_MODE_ALIASES = {
    "independent": ToxinMode.INDEPENDENT,
    "separate": ToxinMode.INDEPENDENT,
    "shared_among_toxins": ToxinMode.SHARED_AMONG_TOXINS,
    "shared": ToxinMode.SHARED_AMONG_TOXINS,
    "combined": ToxinMode.COMBINED_WITH_METABOLITES,
    "combined_with_metabolites": ToxinMode.COMBINED_WITH_METABOLITES,
    "shared_with_metabolites": ToxinMode.COMBINED_WITH_METABOLITES,
}


@dataclass
class TransferEvent:
    """A contact-based, non-depleting transfer/transconjugation term.

    Generalizes the ad-hoc `TRANSFER_RATE * Nd * Nr` term in
    conjugation_unified_model.py (and the Z matrix / `conjugationInflow`
    in FitMicrobialSystemApp.m) to an arbitrary strain triple: contact
    between `donor` and `recipient` populations produces new members of
    `product` at rate `rate * N[donor] * N[recipient]`, WITHOUT depleting
    donor or recipient (plasmid conjugation doesn't consume either
    parent). `product` may equal `recipient` (the simple "i converts j"
    reading of a dense S x S transfer matrix) or be a distinct third
    strain (the exact validated conjugation_unified_model.py case:
    donor=Donor, recipient=Recipient, product=Transconjugant).
    """

    donor: int
    recipient: int
    product: int
    rate: float


@dataclass
class LagConfig:
    """Optional per-strain lag/"readiness" state q in [0, 1].

    dq_i/dt = q_rate[i] * (1 - q_i). Realized growth-minus-kill for strain
    i is scaled by q[lag_source[i]], and its mortality is damped toward
    mort_floor[i] the same way -- see reaction_rhs(). `lag_source`
    defaults to i (a strain tracks its own lag), but can point at another
    strain's q, exactly reproducing conjugation_unified_model.py's
    Transconjugant, which inherits Recipient's qr instead of tracking its
    own (a freshly-converted cell isn't a diluted inoculum needing to
    re-adapt). q0 is intentionally NOT part of this config: per the
    validated model it's a deterministic function of each strain's own
    initial density (see q0_from_density()), not a free parameter.
    """

    q_rate: np.ndarray
    q_reference_density: np.ndarray
    mort_floor: np.ndarray
    lag_source: np.ndarray  # int, shape (S,)

    @classmethod
    def build(cls, S: int, q_rate, q_reference_density, mort_floor=0.0, lag_source=None):
        lag_source = np.arange(S) if lag_source is None else np.asarray(lag_source, dtype=int)
        return cls(
            q_rate=_broadcast_1d(q_rate, S),
            q_reference_density=_broadcast_1d(q_reference_density, S),
            mort_floor=_broadcast_1d(mort_floor, S),
            lag_source=lag_source,
        )

    def q0(self, N0: np.ndarray) -> np.ndarray:
        """q0_i = min(1, N0_i / reference_i) -- deterministic, not fit."""
        N0 = np.asarray(N0, dtype=float)
        return np.minimum(1.0, N0 / self.q_reference_density)


def _broadcast_1d(value: ArrayLike, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"expected shape ({n},), got {arr.shape}")
    return arr


def _broadcast_2d(value: ArrayLike, rows: int, cols: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full((rows, cols), float(arr))
    if arr.shape != (rows, cols):
        raise ValueError(f"expected shape ({rows}, {cols}), got {arr.shape}")
    return arr


def _resolve_toxin_mode(value, S: int) -> np.ndarray:
    if isinstance(value, str):
        value = _TOXIN_MODE_ALIASES[value.lower()]
    if isinstance(value, (ToxinMode, int)):
        return np.full(S, int(value), dtype=int)
    arr = np.asarray(
        [_TOXIN_MODE_ALIASES[v.lower()] if isinstance(v, str) else int(v) for v in value],
        dtype=int,
    )
    if arr.shape != (S,):
        raise ValueError(f"expected {S} toxin-mode entries, got {arr.shape}")
    return arr


@dataclass
class MonodConfig:
    """Full configurable parameter set for S strains, M metabolites, T toxins.

    Matrix names/shapes match docs/model_equations.md Sec. 3 (the
    FitMicrobialSystemApp.m canonical target set) wherever an equivalent
    exists. Use MonodConfig.build(...) rather than the constructor
    directly -- it accepts convenient scalars/lists and broadcasts them.
    """

    S: int
    M: int
    T: int
    r: np.ndarray               # (S, M) growth rate on metabolite
    k: np.ndarray                # (S, M) half-saturation on metabolite
    c: np.ndarray                # (S, M) consumption rate of metabolite
    delta: np.ndarray            # (S,) mortality
    m_supply: np.ndarray         # (M,) metabolite external supply
    D_dilution: np.ndarray       # (M,) metabolite dilution/removal
    P: np.ndarray                # (S, T) toxin kill rate (always subtracted)
    K_tox: np.ndarray            # (S, T) toxin half-saturation
    c_tox: np.ndarray            # (S, T) toxin consumption/binding by strain
    s_secretion: np.ndarray      # (S, T) toxin secretion by strain
    m_toxin_supply: np.ndarray   # (T,) toxin external supply
    d_toxin_decay: np.ndarray    # (T,) toxin decay
    share_metabolite_uptake: np.ndarray  # (S,) bool
    toxin_mode: np.ndarray       # (S,) int, see ToxinMode
    transfer_events: list = field(default_factory=list)
    lag: Optional[LagConfig] = None

    def __post_init__(self):
        combined = self.toxin_mode == ToxinMode.COMBINED_WITH_METABOLITES
        if np.any(combined & ~self.share_metabolite_uptake.astype(bool)):
            bad = np.where(combined & ~self.share_metabolite_uptake.astype(bool))[0]
            raise ValueError(
                "toxin_mode=COMBINED_WITH_METABOLITES requires "
                f"share_metabolite_uptake=True for the same strain(s); "
                f"strain index/indices {bad.tolist()} violate this."
            )
        for ev in self.transfer_events:
            for idx, name in ((ev.donor, "donor"), (ev.recipient, "recipient"), (ev.product, "product")):
                if not (0 <= idx < self.S):
                    raise ValueError(f"transfer event {name} index {idx} out of range for S={self.S}")

    @classmethod
    def build(
        cls,
        S: int,
        M: int,
        T: int = 0,
        *,
        r=0.0, k=1.0, c=0.0, delta=0.0,
        m_supply=0.0, D_dilution=0.0,
        P=0.0, K_tox=1.0, c_tox=0.0, s_secretion=0.0,
        m_toxin_supply=0.0, d_toxin_decay=0.0,
        share_metabolite_uptake: Union[bool, ArrayLike] = True,
        toxin_mode: Union[str, ToxinMode, int, Sequence] = ToxinMode.INDEPENDENT,
        transfer_events: Optional[Sequence[TransferEvent]] = None,
        lag: Optional[LagConfig] = None,
    ) -> "MonodConfig":
        sm = share_metabolite_uptake
        sm_arr = np.full(S, bool(sm)) if isinstance(sm, (bool, np.bool_)) else np.asarray(sm, dtype=bool)
        return cls(
            S=S, M=M, T=T,
            r=_broadcast_2d(r, S, M), k=_broadcast_2d(k, S, M), c=_broadcast_2d(c, S, M),
            delta=_broadcast_1d(delta, S),
            m_supply=_broadcast_1d(m_supply, M), D_dilution=_broadcast_1d(D_dilution, M),
            P=_broadcast_2d(P, S, T), K_tox=_broadcast_2d(K_tox, S, T),
            c_tox=_broadcast_2d(c_tox, S, T), s_secretion=_broadcast_2d(s_secretion, S, T),
            m_toxin_supply=_broadcast_1d(m_toxin_supply, T), d_toxin_decay=_broadcast_1d(d_toxin_decay, T),
            share_metabolite_uptake=sm_arr,
            toxin_mode=_resolve_toxin_mode(toxin_mode, S),
            transfer_events=list(transfer_events) if transfer_events else [],
            lag=lag,
        )


@dataclass
class ReactionState:
    """Convenience bundle for reaction_rhs()'s return value."""

    dN: np.ndarray
    dx: np.ndarray
    dtox: np.ndarray
    dq: Optional[np.ndarray]


def reaction_rhs(N: np.ndarray, x: np.ndarray, tox: np.ndarray, q: Optional[np.ndarray],
                  config: MonodConfig) -> ReactionState:
    """Local (well-mixed, single point) reaction term, batched over "...".

    N: (..., S) strain populations (linear space, >= 0)
    x: (..., M) metabolite concentrations (linear space, >= 0)
    tox: (..., T) toxin concentrations (linear space, >= 0); T may be 0
    q: (..., S) lag/readiness states in [0, 1], or None if config.lag is None

    Returns derivatives in the SAME linear space and SAME batch shape.
    """
    S, M, T = config.S, config.M, config.T
    N = np.asarray(N, dtype=float)
    x = np.asarray(x, dtype=float)
    tox = np.asarray(tox, dtype=float)

    sat_x = x[..., np.newaxis, :] / (config.k + _EPS)          # (..., S, M)
    sat_tox = tox[..., np.newaxis, :] / (config.K_tox + _EPS)  # (..., S, T)

    # --- metabolite growth/uptake, two candidate denominators ---
    denom_shared_x = 1.0 + sat_x.sum(axis=-1)                          # (..., S)
    denom_indep_x = 1.0 + sat_x                                        # (..., S, M)
    denom_combined = denom_shared_x + sat_tox.sum(axis=-1)             # (..., S)

    growth_shared_x = np.einsum("...sm,sm->...s", sat_x, config.r) / denom_shared_x
    growth_indep_x = (sat_x * config.r / denom_indep_x).sum(axis=-1)
    growth_combined_x = np.einsum("...sm,sm->...s", sat_x, config.r) / denom_combined

    uptake_shared_x = sat_x * config.c / denom_shared_x[..., np.newaxis]
    uptake_indep_x = sat_x * config.c / denom_indep_x
    uptake_combined_x = sat_x * config.c / denom_combined[..., np.newaxis]

    # --- toxin kill/uptake/secretion, three candidate denominators ---
    denom_tox_indep = 1.0 + sat_tox                                    # (..., S, T)
    denom_tox_shared = 1.0 + sat_tox.sum(axis=-1)                      # (..., S)

    kill_indep = (sat_tox * config.P / denom_tox_indep).sum(axis=-1)
    kill_shared = (sat_tox * config.P).sum(axis=-1) / denom_tox_shared
    kill_combined = (sat_tox * config.P).sum(axis=-1) / denom_combined

    uptake_tox_indep = sat_tox * config.c_tox / denom_tox_indep
    uptake_tox_shared = sat_tox * config.c_tox / denom_tox_shared[..., np.newaxis]
    uptake_tox_combined = sat_tox * config.c_tox / denom_combined[..., np.newaxis]

    secrete_tox_indep = sat_tox * config.s_secretion / denom_tox_indep
    secrete_tox_shared = sat_tox * config.s_secretion / denom_tox_shared[..., np.newaxis]
    secrete_tox_combined = sat_tox * config.s_secretion / denom_combined[..., np.newaxis]

    # --- select per-strain branch ---
    is_combined = config.toxin_mode == ToxinMode.COMBINED_WITH_METABOLITES
    is_shared_tox = config.toxin_mode == ToxinMode.SHARED_AMONG_TOXINS
    sm = config.share_metabolite_uptake

    growth_metab = np.where(is_combined, growth_combined_x, np.where(sm, growth_shared_x, growth_indep_x))
    tox_kill = np.where(is_combined, kill_combined, np.where(is_shared_tox, kill_shared, kill_indep))

    is_combined_col = is_combined[:, np.newaxis]
    is_shared_tox_col = is_shared_tox[:, np.newaxis]
    sm_col = sm[:, np.newaxis]

    uptake_metab = np.where(is_combined_col, uptake_combined_x, np.where(sm_col, uptake_shared_x, uptake_indep_x))
    uptake_tox = np.where(is_combined_col, uptake_tox_combined, np.where(is_shared_tox_col, uptake_tox_shared, uptake_tox_indep))
    secrete_tox = np.where(is_combined_col, secrete_tox_combined, np.where(is_shared_tox_col, secrete_tox_shared, secrete_tox_indep))

    net_percapita = growth_metab - tox_kill  # (..., S)

    if config.lag is not None:
        if q is None:
            raise ValueError("config.lag is set but q was not provided")
        q = np.asarray(q, dtype=float)
        q_eff = np.take(q, config.lag.lag_source, axis=-1)
        net_percapita = net_percapita * q_eff
        mort_floor = config.lag.mort_floor
        mort_eff = config.delta * (mort_floor + (1.0 - mort_floor) * q_eff)
        dq = config.lag.q_rate * (1.0 - q)
    else:
        mort_eff = config.delta
        dq = None

    dN_transfer = np.zeros_like(N)
    for ev in config.transfer_events:
        flux = ev.rate * N[..., ev.donor] * N[..., ev.recipient]
        dN_transfer[..., ev.product] = dN_transfer[..., ev.product] + flux

    dN = N * (net_percapita - mort_eff) + dN_transfer
    dx = config.m_supply - config.D_dilution * x - np.einsum("...sm,...s->...m", uptake_metab, N)
    dtox = (
        config.m_toxin_supply - config.d_toxin_decay * tox
        + np.einsum("...st,...s->...t", secrete_tox, N)
        - np.einsum("...st,...s->...t", uptake_tox, N)
    )

    return ReactionState(dN=dN, dx=dx, dtox=dtox, dq=dq)
