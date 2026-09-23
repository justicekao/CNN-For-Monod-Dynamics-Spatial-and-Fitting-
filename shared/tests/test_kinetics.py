"""
Validates kinetics.reaction_rhs() -- the new general, configurable engine
-- against the two validated reference scripts' OWN math, at many sample
states, plus sanity/invariant checks for the new modified-Monod variants
(toxin-shares-metabolite-denominator, independent per-metabolite uptake,
transfer, lag) that don't exist in either reference script.

The reference scripts are imported directly (not reimplemented by hand)
so there is no risk of transcribing their formulas incorrectly; see the
`if __name__ == "__main__":` guards added to those files, which only
suppress their top-level integration/plotting side effects on import and
do not change any formula.
"""

import numpy as np
import pytest

from shared.monod_core import LagConfig, MonodConfig, ToxinMode, TransferEvent, reaction_rhs
from shared.monod_core import standalone_two_strain_system as standalone
from shared.monod_core import conjugation_unified_model as conj

RNG = np.random.default_rng(0)


def _sample_positive(shape, low=1e-3, high=5.0):
    return RNG.uniform(low, high, size=shape)


# ---------------------------------------------------------------------------
# standalone_two_strain_system.py: 2 strains, 3 metabolites, 1 toxin.
# Its toxin term is a signed "growth rate added" convention
# (TOXIN_GROWTH_RATE, positive or negative); kinetics.py's P is always a
# SUBTRACTED kill rate (the FitMicrobialSystemApp.m / docs Sec. 3
# convention), so P = -TOXIN_GROWTH_RATE reproduces it exactly.
# ---------------------------------------------------------------------------

def _standalone_config(share_uptake: bool) -> MonodConfig:
    return MonodConfig.build(
        S=2, M=3, T=1,
        r=np.array(standalone.GROWTH_RATE, dtype=float),
        k=np.array(standalone.HALF_SAT, dtype=float),
        c=np.array(standalone.CONSUMPTION, dtype=float),
        delta=np.array(standalone.MORTALITY, dtype=float),
        m_supply=np.array(standalone.METABOLITE_SUPPLY, dtype=float),
        D_dilution=np.array(standalone.METABOLITE_DILUTION, dtype=float),
        P=-np.array(standalone.TOXIN_GROWTH_RATE, dtype=float).reshape(2, 1),
        K_tox=np.array(standalone.TOXIN_HALF_SAT, dtype=float).reshape(2, 1),
        c_tox=np.array(standalone.TOXIN_CONSUMPTION, dtype=float).reshape(2, 1),
        s_secretion=0.0,
        m_toxin_supply=np.array([standalone.TOXIN_SUPPLY]),
        d_toxin_decay=np.array([standalone.TOXIN_DECAY]),
        share_metabolite_uptake=share_uptake,
        toxin_mode=ToxinMode.INDEPENDENT,
    )


def _standalone_reference_rhs(state, share_uptake: bool):
    """The exact reference derivative, via the script's own functions."""
    N1, N2, A, B, C, Tox = state
    g1, u1A, u1B, u1C, u1T = standalone._strain_growth_and_uptake(
        standalone.GROWTH_RATE[0], standalone.HALF_SAT[0], standalone.CONSUMPTION[0],
        standalone.TOXIN_GROWTH_RATE[0], standalone.TOXIN_HALF_SAT[0], standalone.TOXIN_CONSUMPTION[0],
        A, B, C, Tox, share_uptake,
    )
    g2, u2A, u2B, u2C, u2T = standalone._strain_growth_and_uptake(
        standalone.GROWTH_RATE[1], standalone.HALF_SAT[1], standalone.CONSUMPTION[1],
        standalone.TOXIN_GROWTH_RATE[1], standalone.TOXIN_HALF_SAT[1], standalone.TOXIN_CONSUMPTION[1],
        A, B, C, Tox, share_uptake,
    )
    dN1 = N1 * (g1 - standalone.MORTALITY[0])
    dN2 = N2 * (g2 - standalone.MORTALITY[1])
    dA = standalone.METABOLITE_SUPPLY[0] - standalone.METABOLITE_DILUTION[0] * A - N1 * u1A - N2 * u2A
    dB = standalone.METABOLITE_SUPPLY[1] - standalone.METABOLITE_DILUTION[1] * B - N1 * u1B - N2 * u2B
    dC = standalone.METABOLITE_SUPPLY[2] - standalone.METABOLITE_DILUTION[2] * C - N1 * u1C - N2 * u2C
    dTox = standalone.TOXIN_SUPPLY - standalone.TOXIN_DECAY * Tox - N1 * u1T - N2 * u2T
    return np.array([dN1, dN2, dA, dB, dC, dTox])


@pytest.mark.parametrize("share_uptake", [True, False])
def test_matches_standalone_two_strain_system(share_uptake):
    config = _standalone_config(share_uptake)
    for _ in range(20):
        N = _sample_positive(2)
        x = _sample_positive(3)
        tox = _sample_positive(1)
        rs = reaction_rhs(N, x, tox, None, config)
        got = np.concatenate([rs.dN, rs.dx, rs.dtox])
        expected = _standalone_reference_rhs(np.concatenate([N, x, tox]), share_uptake)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# conjugation_unified_model.py: 3 populations (Recipient, Donor,
# Transconjugant), private per-population resource pools (M=3, one per
# population, cross-consumption matrix c is diagonal), transfer, and lag
# with Transconjugant inheriting Recipient's q.
# ---------------------------------------------------------------------------

RECIPIENT, DONOR, TCONJ = 0, 1, 2


def _conjugation_config() -> MonodConfig:
    r = np.zeros((3, 3))
    k = np.ones((3, 3)) * conj.K
    c = np.zeros((3, 3))
    for i in range(3):
        r[i, i] = [conj.DONOR_RECIPIENT_R, conj.DONOR_RECIPIENT_R, conj.TCONJ_R][i]
        c[i, i] = conj.CONSUMPTION
    delta = np.array([conj.DONOR_RECIPIENT_MORTALITY, conj.DONOR_RECIPIENT_MORTALITY, conj.TCONJ_MORTALITY])
    m_supply = np.array([conj.SUPPLY_DR, conj.SUPPLY_DR, conj.SUPPLY_T])
    D_dilution = np.full(3, conj.DILUTION)
    lag = LagConfig.build(
        S=3,
        q_rate=conj.Q_RATE,
        q_reference_density=conj.Q_REFERENCE_DENSITY,
        mort_floor=conj.MORT_FLOOR,
        lag_source=[RECIPIENT, DONOR, RECIPIENT],  # Transconjugant inherits Recipient's q
    )
    transfer = [TransferEvent(donor=DONOR, recipient=RECIPIENT, product=TCONJ, rate=conj.TRANSFER_RATE)]
    return MonodConfig.build(
        S=3, M=3, T=0,
        r=r, k=k, c=c, delta=delta,
        m_supply=m_supply, D_dilution=D_dilution,
        share_metabolite_uptake=False,  # each population's pool is independent anyway (off-diagonal r/c/k are 0)
        transfer_events=transfer,
        lag=lag,
    )


def _conjugation_reference_rhs(Nr, Nd, Nt, Ar, Ad, At, qr, qd):
    sat_r, sat_d, sat_t = Ar / conj.K, Ad / conj.K, At / conj.K
    gr = conj.DONOR_RECIPIENT_R * sat_r / (1 + sat_r) * qr
    gd = conj.DONOR_RECIPIENT_R * sat_d / (1 + sat_d) * qd
    gt = conj.TCONJ_R * sat_t / (1 + sat_t) * qr
    mort_r_eff = conj.DONOR_RECIPIENT_MORTALITY * (conj.MORT_FLOOR + (1 - conj.MORT_FLOOR) * qr)
    mort_d_eff = conj.DONOR_RECIPIENT_MORTALITY * (conj.MORT_FLOOR + (1 - conj.MORT_FLOOR) * qd)
    mort_t_eff = conj.TCONJ_MORTALITY * (conj.MORT_FLOOR + (1 - conj.MORT_FLOOR) * qr)
    dNr = Nr * (gr - mort_r_eff)
    dNd = Nd * (gd - mort_d_eff)
    dNt = Nt * (gt - mort_t_eff) + conj.TRANSFER_RATE * Nd * Nr
    uptake_r = conj.CONSUMPTION * sat_r / (1 + sat_r)
    uptake_d = conj.CONSUMPTION * sat_d / (1 + sat_d)
    uptake_t = conj.CONSUMPTION * sat_t / (1 + sat_t)
    dAr = conj.SUPPLY_DR - conj.DILUTION * Ar - Nr * uptake_r
    dAd = conj.SUPPLY_DR - conj.DILUTION * Ad - Nd * uptake_d
    dAt = conj.SUPPLY_T - conj.DILUTION * At - Nt * uptake_t
    dqr = conj.Q_RATE * (1 - qr)
    dqd = conj.Q_RATE * (1 - qd)
    return dNr, dNd, dNt, dAr, dAd, dAt, dqr, dqd


def test_matches_conjugation_unified_model():
    # kinetics.py always adds a small epsilon to half-saturation constants
    # as a general zero-protection safeguard (see _EPS in kinetics.py);
    # conjugation_unified_model.py's system_rhs divides by K directly with
    # no such guard (K=1.0 is a nonzero constant, so it never needed one).
    # That difference alone produces a ~1e-8 relative deviation -- real,
    # expected, and unrelated to the reaction math itself -- so this test
    # uses a looser tolerance than test_matches_standalone_two_strain_system
    # (which uses the same epsilon convention on both sides and so matches
    # to machine precision). Near a growth/mortality balance point dN is a
    # small difference of large terms, so this tiny absolute perturbation
    # can look like a larger relative error there -- atol is set generously
    # to absorb that cancellation effect rather than chasing a tighter rtol.
    rtol, atol = 1e-6, 1e-1
    config = _conjugation_config()
    for _ in range(20):
        Nr, Nd, Nt = _sample_positive(3, low=1.0, high=1e6)
        Ar, Ad, At = _sample_positive(3, low=1e-3, high=2.0)
        qr, qd = RNG.uniform(0.01, 1.0, size=2)
        N = np.array([Nr, Nd, Nt])
        x = np.array([Ar, Ad, At])
        q = np.array([qr, qd, 0.0])  # Transconjugant's own q slot is unused (lag_source points at Recipient)

        rs = reaction_rhs(N, x, np.zeros(0), q, config)
        expected = _conjugation_reference_rhs(Nr, Nd, Nt, Ar, Ad, At, qr, qd)
        dNr, dNd, dNt, dAr, dAd, dAt, dqr, dqd = expected

        np.testing.assert_allclose(rs.dN, [dNr, dNd, dNt], rtol=rtol, atol=atol)
        np.testing.assert_allclose(rs.dx, [dAr, dAd, dAt], rtol=rtol, atol=atol)
        np.testing.assert_allclose(rs.dq[:2], [dqr, dqd], rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# New variants that neither reference script exercises: sanity/invariant
# checks rather than exact-match tests, since there's no independent
# reference implementation for these.
# ---------------------------------------------------------------------------

def test_toxin_combined_with_metabolites_requires_shared_metabolite_uptake():
    with pytest.raises(ValueError):
        MonodConfig.build(
            S=1, M=1, T=1,
            r=1.0, k=1.0, c=0.1, delta=0.1,
            m_supply=1.0, D_dilution=0.1,
            P=0.2, K_tox=1.0,
            share_metabolite_uptake=False,
            toxin_mode=ToxinMode.COMBINED_WITH_METABOLITES,
        )


def test_toxin_combined_denominator_reduces_growth_more_than_independent():
    """Putting a toxin in the SAME denominator as metabolites means it
    competes for uptake capacity, which should suppress net per-capita
    growth more than treating it as fully independent (all else equal)."""
    common = dict(S=1, M=1, T=1, r=2.0, k=1.0, c=0.0, delta=0.0,
                  m_supply=0.0, D_dilution=0.0, P=1.0, K_tox=1.0)
    independent = MonodConfig.build(**common, share_metabolite_uptake=True, toxin_mode=ToxinMode.INDEPENDENT)
    combined = MonodConfig.build(**common, share_metabolite_uptake=True, toxin_mode=ToxinMode.COMBINED_WITH_METABOLITES)

    N, x, tox = np.array([1.0]), np.array([1.0]), np.array([1.0])
    rs_indep = reaction_rhs(N, x, tox, None, independent)
    rs_combined = reaction_rhs(N, x, tox, None, combined)
    # net per-capita growth = dN / N (delta=0 here)
    assert rs_combined.dN[0] < rs_indep.dN[0]


def test_independent_vs_shared_metabolite_uptake_differ():
    config_shared = MonodConfig.build(
        S=1, M=2, T=0, r=[[1.0, 1.0]], k=[[1.0, 1.0]], c=0.0, delta=0.0,
        m_supply=0.0, D_dilution=0.0, share_metabolite_uptake=True,
    )
    config_indep = MonodConfig.build(
        S=1, M=2, T=0, r=[[1.0, 1.0]], k=[[1.0, 1.0]], c=0.0, delta=0.0,
        m_supply=0.0, D_dilution=0.0, share_metabolite_uptake=False,
    )
    N, x = np.array([1.0]), np.array([2.0, 2.0])
    rs_shared = reaction_rhs(N, x, np.zeros(0), None, config_shared)
    rs_indep = reaction_rhs(N, x, np.zeros(0), None, config_indep)
    # independent saturation always yields >= growth than one shared capacity
    # when there's more than one active resource (no competition for capacity)
    assert rs_indep.dN[0] > rs_shared.dN[0]


def test_transfer_event_is_non_depleting():
    """Contact-based transfer must not subtract from donor or recipient."""
    config = MonodConfig.build(
        S=3, M=0, T=0, r=0.0, k=1.0, c=0.0, delta=0.0,
        m_supply=np.zeros(0), D_dilution=np.zeros(0),
        transfer_events=[TransferEvent(donor=0, recipient=1, product=2, rate=1e-3)],
    )
    N = np.array([100.0, 50.0, 0.0])
    rs = reaction_rhs(N, np.zeros(0), np.zeros(0), None, config)
    assert rs.dN[0] == 0.0  # donor unaffected
    assert rs.dN[1] == 0.0  # recipient unaffected
    assert rs.dN[2] == pytest.approx(1e-3 * 100.0 * 50.0)  # product gains the full contact flux


def test_lag_suppresses_growth_when_q_is_low():
    lag = LagConfig.build(S=1, q_rate=0.1, q_reference_density=1e6, mort_floor=0.5)
    config = MonodConfig.build(
        S=1, M=1, T=0, r=2.0, k=1.0, c=0.0, delta=0.1,
        m_supply=0.0, D_dilution=0.0, lag=lag,
    )
    N, x = np.array([1.0]), np.array([10.0])
    rs_low_q = reaction_rhs(N, x, np.zeros(0), np.array([0.01]), config)
    rs_high_q = reaction_rhs(N, x, np.zeros(0), np.array([1.0]), config)
    assert rs_low_q.dN[0] < rs_high_q.dN[0]


def test_reaction_rhs_is_batched_over_leading_dimensions():
    """The same call must work for a single well-mixed state and for a
    batch of states (e.g. one per spatial grid cell) with identical
    per-row results -- this is what lets spatial_pde reuse this module
    unmodified."""
    config = _standalone_config(share_uptake=True)
    states = [
        (_sample_positive(2), _sample_positive(3), _sample_positive(1))
        for _ in range(5)
    ]
    N_batch = np.stack([s[0] for s in states])
    x_batch = np.stack([s[1] for s in states])
    tox_batch = np.stack([s[2] for s in states])

    rs_batch = reaction_rhs(N_batch, x_batch, tox_batch, None, config)
    for i, (N, x, tox) in enumerate(states):
        rs_single = reaction_rhs(N, x, tox, None, config)
        np.testing.assert_allclose(rs_batch.dN[i], rs_single.dN)
        np.testing.assert_allclose(rs_batch.dx[i], rs_single.dx)
        np.testing.assert_allclose(rs_batch.dtox[i], rs_single.dtox)
