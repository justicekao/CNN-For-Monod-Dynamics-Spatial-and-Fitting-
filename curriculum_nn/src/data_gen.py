"""
data_gen.py -- synthetic (trajectory -> ground-truth parameters) data for
the curriculum-trained parameter-fitting network, per
docs/curriculum_nn_spec.md's "Synthetic training data generation" section.

The key idea, straight from that spec (and from
conjugation_unified_model.py's own `_solve_equilibrium`, which this
generalizes): uniform-random parameters mostly produce either trivial
extinction or numerically unstable trajectories, wasting the sampling
budget. Instead, CHOOSE a target equilibrium first (population and
resource levels that are actually reachable), then SOLVE for the
mortality/supply rates that make it a genuine fixed point of the ODE, and
only THEN perturb away from it and simulate. This is done here for
ARBITRARY (S, M, T) and sharing topology by calling
shared.monod_core.kinetics.reaction_rhs directly with the supply/
dilution/mortality terms temporarily zeroed out, rather than
hand-deriving an equilibrium formula per topology the way
`_solve_equilibrium` did for its one fixed case.

Scope (per docs/curriculum_nn_spec.md's stage list): Stage 0 and Stage 1
only -- pure Monod form (§1-2 of docs/model_equations.md), no
transconjugation/transfer and no lag state. Stage 2 (transfer) and Stage
3 (lag) are explicitly future work; see docs/curriculum_nn_spec.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from shared.monod_core import MonodConfig, ToxinMode, WellMixedResult, reaction_rhs, simulate_well_mixed

Tier = Tuple[int, int, int]  # (S, M, T)


@dataclass
class StageSpec:
    """One curriculum stage: a set of (S, M, T) dimensionality tiers to
    sample from, plus the Monod-variant configuration common to all of
    them. `sparsity` is the fraction of strain-metabolite (and
    strain-toxin) pairs that are active (nonzero rate) -- per
    docs/model_equations.md §3, most real systems are NOT dense, so a
    dense (sparsity=1.0) default is the easy/uninteresting case, and
    lower sparsity is closer to the target complexity level."""

    name: str
    tiers: List[Tier]
    share_metabolite_uptake: bool = True
    toxin_mode: ToxinMode = ToxinMode.INDEPENDENT
    sparsity: float = 1.0


# Stage 0 (docs/curriculum_nn_spec.md): exactly
# standalone_two_strain_system.py's scale, pure Monod, no toxin or one toxin.
STAGE_0 = StageSpec(name="stage0", tiers=[(2, 3, 0), (2, 3, 1)])

# Stage 1: scale S, M, T up within the pure Monod form, still no transfer/lag.
STAGE_1 = StageSpec(name="stage1", tiers=[(2, 3, 1), (3, 4, 1), (4, 6, 2)])

STAGES: Dict[str, StageSpec] = {"stage0": STAGE_0, "stage1": STAGE_1}


@dataclass
class Sample:
    """One (trajectory, ground-truth parameters) training example."""

    S: int
    M: int
    T: int
    config: MonodConfig
    t: np.ndarray
    N: np.ndarray     # (n_points, S)
    x: np.ndarray      # (n_points, M)
    tox: np.ndarray     # (n_points, T)

    def params(self) -> Dict[str, np.ndarray]:
        """The ground-truth parameter set the model is trained to recover
        -- the "core" pure-Monod matrices from docs/model_equations.md §3.
        (c_tox/s_secretion, Z, and lag parameters are out of scope for
        Stage 0-1; see this module's docstring.)"""
        out = {"r": self.config.r, "k": self.config.k, "c": self.config.c,
               "delta": self.config.delta, "m_supply": self.config.m_supply,
               "D_dilution": self.config.D_dilution}
        if self.T:
            out["P"] = self.config.P
            out["K_tox"] = self.config.K_tox
        return out


def _sample_masked_uniform(rng: np.random.Generator, low: float, high: float,
                            shape: Tuple[int, ...], sparsity: float) -> np.ndarray:
    values = rng.uniform(low, high, size=shape)
    if sparsity >= 1.0:
        return values
    mask = rng.random(shape) < sparsity
    return values * mask


def _sample_raw_config(S: int, M: int, T: int, rng: np.random.Generator, stage: StageSpec) -> MonodConfig:
    """Sample growth/uptake/toxin rate matrices freely; delta/m_supply/
    D_dilution/m_toxin_supply/d_toxin_decay are placeholders (zero) here
    -- solve_equilibrium() below fills them in so the sample has a real
    fixed point."""
    r = _sample_masked_uniform(rng, 0.3, 2.0, (S, M), stage.sparsity)
    k = rng.uniform(0.3, 3.0, size=(S, M))
    c = _sample_masked_uniform(rng, 0.02, 0.3, (S, M), stage.sparsity)
    P = _sample_masked_uniform(rng, 0.1, 1.0, (S, T), stage.sparsity) if T else 0.0
    K_tox = rng.uniform(0.3, 3.0, size=(S, T)) if T else 1.0
    return MonodConfig.build(
        S=S, M=M, T=T,
        r=r, k=k, c=c, delta=0.0,
        m_supply=0.0, D_dilution=0.0,
        P=P, K_tox=K_tox, c_tox=0.0, s_secretion=0.0,
        m_toxin_supply=0.0, d_toxin_decay=0.0,
        share_metabolite_uptake=stage.share_metabolite_uptake,
        toxin_mode=stage.toxin_mode,
    )


def solve_equilibrium(config: MonodConfig, N_star: np.ndarray, x_star: np.ndarray,
                       tox_star: np.ndarray, rng: np.random.Generator,
                       D_dilution_range=(0.1, 0.5), d_toxin_decay_range=(0.05, 0.3)) -> MonodConfig:
    """Generalizes conjugation_unified_model.py's `_solve_equilibrium` to
    ARBITRARY (S, M, T) and sharing topology by using reaction_rhs itself
    rather than a hand-derived formula.

    With delta/m_supply/D_dilution/m_toxin_supply/d_toxin_decay all zeroed,
    reaction_rhs's dN at (N*, x*, tox*) is exactly N* * growth(x*, tox*)
    (no mortality, no transfer for Stage 0-1) -- so dividing by N* recovers
    the per-capita growth rate directly, and SETTING delta to that value
    makes (N*, x*, tox*) a fixed point of the population equations. The
    same trick (zero supply/dilution, read off the resulting pure
    consumption/secretion term) gives the supply needed to zero the
    resource/toxin equations too.
    """
    S, M, T = config.S, config.M, config.T
    zeroed = replace(
        config,
        delta=np.zeros(S), m_supply=np.zeros(M), D_dilution=np.zeros(M),
        m_toxin_supply=np.zeros(T), d_toxin_decay=np.zeros(T),
    )
    rs0 = reaction_rhs(N_star, x_star, tox_star, None, zeroed)

    delta = rs0.dN / N_star
    D_dilution = rng.uniform(*D_dilution_range, size=M)
    m_supply = D_dilution * x_star - rs0.dx
    if T:
        d_toxin_decay = rng.uniform(*d_toxin_decay_range, size=T)
        m_toxin_supply = d_toxin_decay * tox_star - rs0.dtox
    else:
        d_toxin_decay = np.zeros(0)
        m_toxin_supply = np.zeros(0)

    return replace(
        config,
        delta=delta, m_supply=m_supply, D_dilution=D_dilution,
        m_toxin_supply=m_toxin_supply, d_toxin_decay=d_toxin_decay,
    )


def _is_physically_valid(config: MonodConfig, delta_range=(0.01, 3.0), m_supply_max=50.0) -> bool:
    if np.any(config.delta <= delta_range[0]) or np.any(config.delta > delta_range[1]):
        return False
    if np.any(config.m_supply < 0) or np.any(config.m_supply > m_supply_max):
        return False
    if config.T and (np.any(config.m_toxin_supply < 0) or np.any(config.m_toxin_supply > m_supply_max)):
        return False
    return True


def sample_config(S: int, M: int, T: int, rng: np.random.Generator, stage: StageSpec,
                   n_pop_log_range=(0.0, 1.5), x_star_range=(0.5, 3.0), tox_star_range=(0.2, 2.0),
                   max_attempts: int = 50) -> Tuple[MonodConfig, np.ndarray, np.ndarray, np.ndarray]:
    """Rejection-samples until the equilibrium-solved config is physically
    sane (positive, bounded mortality/supply rates) -- see
    _is_physically_valid. Returns (config, N_star, x_star, tox_star).

    n_pop_log_range defaults to O(1)-O(30) populations, matching
    standalone_two_strain_system.py's scale (Stage 0's stated scope) --
    NOT realistic ~1e8 cfu/mL populations. docs/model_equations.md §4 is
    explicit that consumption/transfer rates must be scaled down hard
    (~1e-8 to 1e-13, as conjugation_unified_model.py does) to stay
    numerically sane at that realistic scale; since this generator uses
    the same O(0.02-0.3) consumption-rate range as the small reference
    script, it needs the matching small population scale, not the large
    one. A future stage targeting realistic cfu/mL scale would need
    consumption rates sampled on a scale tied to n_pop_log_range, not a
    fixed range independent of it.
    """
    for _ in range(max_attempts):
        raw = _sample_raw_config(S, M, T, rng, stage)
        N_star = 10.0 ** rng.uniform(*n_pop_log_range, size=S)
        x_star = rng.uniform(*x_star_range, size=M)
        tox_star = rng.uniform(*tox_star_range, size=T) if T else np.zeros(0)
        config = solve_equilibrium(raw, N_star, x_star, tox_star, rng)
        if _is_physically_valid(config):
            return config, N_star, x_star, tox_star
    raise RuntimeError(f"could not sample a physically valid config for (S={S}, M={M}, T={T}) "
                        f"in {max_attempts} attempts")


def generate_sample(S: int, M: int, T: int, rng: np.random.Generator, stage: StageSpec,
                     t_span=(0.0, 200.0), n_points: int = 50,
                     perturb_log_std: float = 0.5, max_attempts: int = 20) -> Sample:
    """Sample a config with a guaranteed equilibrium, perturb the initial
    condition away from it (multiplicatively, in log-space, so it stays
    positive), and simulate -- retrying with a fresh config if the
    resulting trajectory isn't finite (extremely rare given the
    equilibrium-solving above, but the ODE is still nonlinear)."""
    for _ in range(max_attempts):
        config, N_star, x_star, tox_star = sample_config(S, M, T, rng, stage)
        N0 = N_star * np.exp(rng.normal(0, perturb_log_std, size=S))
        x0 = x_star * np.exp(rng.normal(0, perturb_log_std, size=M))
        tox0 = tox_star * np.exp(rng.normal(0, perturb_log_std, size=T)) if T else np.zeros(0)

        result: WellMixedResult = simulate_well_mixed(config, N0=N0, x0=x0, tox0=tox0,
                                                        t_span=t_span, n_points=n_points)
        if np.all(np.isfinite(result.N)) and np.all(np.isfinite(result.x)) and np.all(np.isfinite(result.tox)):
            return Sample(S=S, M=M, T=T, config=config, t=result.t, N=result.N, x=result.x, tox=result.tox)
    raise RuntimeError(f"could not generate a finite trajectory for (S={S}, M={M}, T={T}) "
                        f"in {max_attempts} attempts")


def generate_dataset(stage: StageSpec, n_per_tier: int, rng: np.random.Generator,
                      t_span=(0.0, 200.0), n_points: int = 50) -> Dict[Tier, List[Sample]]:
    """One list of Samples per dimensionality tier in the stage."""
    return {
        tier: [generate_sample(*tier, rng=rng, stage=stage, t_span=t_span, n_points=n_points)
               for _ in range(n_per_tier)]
        for tier in stage.tiers
    }


def split_by_tier(dataset: Dict[Tier, List[Sample]], val_frac: float = 0.2, test_frac: float = 0.2,
                   rng: np.random.Generator = None) -> Tuple[Dict[Tier, List[Sample]], Dict[Tier, List[Sample]], Dict[Tier, List[Sample]]]:
    """Train/val/test split done WITHIN each tier (not by holding a whole
    tier out) -- for measuring ordinary generalization at a size the
    model has seen. To measure the curriculum's actual claim (does
    training on small tiers help at a LARGER, unseen-during-training
    tier), hold an entire tier out of `dataset` before calling this and
    evaluate on it separately -- see curriculum.py."""
    rng = rng or np.random.default_rng()
    train, val, test = {}, {}, {}
    for tier, samples in dataset.items():
        idx = rng.permutation(len(samples))
        n_val = int(round(val_frac * len(samples)))
        n_test = int(round(test_frac * len(samples)))
        val_idx, test_idx, train_idx = idx[:n_val], idx[n_val:n_val + n_test], idx[n_val + n_test:]
        train[tier] = [samples[i] for i in train_idx]
        val[tier] = [samples[i] for i in val_idx]
        test[tier] = [samples[i] for i in test_idx]
    return train, val, test
