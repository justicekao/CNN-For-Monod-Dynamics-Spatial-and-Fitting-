"""
standalone_two_strain_system.py

A COMPLETELY STANDALONE script for a 2-strain, 3-metabolite, 1-toxin
system. Does NOT import the genmonod package at all -- only numpy,
scipy, and matplotlib (install with `pip install numpy scipy
matplotlib` if you don't already have them). Just run:

    python standalone_two_strain_system.py

Edit the PARAMETERS section below and re-run -- nothing else needs to
change for typical use.

THE MODEL: growth is Monod-type (saturating) on each metabolite and on
the toxin. If SHARE_UPTAKE = True, a strain's three metabolites draw on
one shared, limited uptake capacity (one "denominator" -- more of one
metabolite leaves proportionally less realized benefit from the
others); if False, each metabolite saturates independently. The toxin
always has its own independent effect (added on top, not competing with
the metabolites for capacity) -- see the comment near TOXIN_GROWTH_RATE
if you want to change that.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# PARAMETERS — edit everything in this section; nothing below needs to change
# for typical use.
# =============================================================================

STRAIN_NAMES = ["Strain_1", "Strain_2"]
METABOLITE_NAMES = ["Metabolite_A", "Metabolite_B", "Metabolite_C"]
TOXIN_NAME = "Toxin"

# growth rate (r) and half-saturation (K) on each metabolite.
# Row 0 = Strain_1, row 1 = Strain_2. Column order = A, B, C.
GROWTH_RATE = [
    [1, 1, 0],
    [0, 1, 1],
]
HALF_SAT = [
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
]
# how fast each strain draws down each metabolite (same row/column layout)
CONSUMPTION = [
    [0.2, 0.2, 0],
    [0, 0.2, 0.2],
]
# NEEDS YOUR INPUT: consumption is safe at these small (~O(1)) population
# scales used below. If you push populations up toward real cfu/mL scale
# (~1e8+), nonzero consumption can make the ODE numerically stiff --
# scale it down hard if you go there.

# toxin: effect on each strain's growth (usually negative), and its own
# half-saturation. The toxin is treated as its OWN independent process
# by default (added on top of metabolite growth, not competing with it
# for capacity) -- this matches the common case of a toxin acting
# through a different pathway than nutrient uptake (e.g. membrane
# disruption vs. nutrient transport).
TOXIN_GROWTH_RATE = [-0.5, 0]   # Strain_1, Strain_2
TOXIN_HALF_SAT = [0.5, 0.5]
TOXIN_CONSUMPTION = [0.025, -1]      # uptake/binding of toxin by each strain (usually 0)

MORTALITY = [0.1, 0.1]              # one value per strain

METABOLITE_SUPPLY = [0.5, 0.5, 0.5]     # one value per metabolite
METABOLITE_DILUTION = [0.2, 0.2, 0.2]

TOXIN_SUPPLY = 0.0    # set > 0 for an ambient/external toxin source
TOXIN_DECAY = 0

# do a strain's own 3 metabolites compete for ONE shared uptake capacity
# (True), or saturate independently of each other (False)?
SHARE_UPTAKE = True

# initial conditions, in the order [Strain_1, Strain_2, A, B, C, Toxin]
Y0 = [1.0, 0.00001, 2.0, 2.0, 2.0, 0.1,0]

T_START, T_END, N_POINTS = 0, 1000, 200

# =============================================================================
# THE MODEL — shouldn't need to edit below this line for typical use
# =============================================================================

_EPS = 1e-8


def _strain_growth_and_uptake(r_metab, K_metab, c_metab, r_tox, K_tox, c_tox, A, B, C, Tox, share_uptake):
    """
    Returns (growth_percapita, uptake_A, uptake_B, uptake_C, uptake_tox)
    for ONE strain, given the current metabolite/toxin concentrations.

    Metabolite growth uses a Monod term per metabolite, x/(K+x); when
    share_uptake=True, all three metabolites' saturation terms (x/K)
    share ONE denominator (1 + sum of all three x/K), so drawing more of
    one metabolite leaves proportionally less realized benefit from the
    others -- this is the "metabolic overlap" mechanism. When False,
    each metabolite gets its own denominator (1 + its own x/K),
    independent of the others.
    """
    r_metab = np.asarray(r_metab, dtype=float)
    K_metab = np.asarray(K_metab, dtype=float)
    c_metab = np.asarray(c_metab, dtype=float)
    x = np.array([A, B, C])
    sat = x / (K_metab + _EPS)   # x/K for each metabolite

    if share_uptake:
        denom = 1.0 + sat.sum()
        growth_metab = np.dot(r_metab, sat) / denom
        uptake_metab = c_metab * sat / denom
    else:
        denom_each = 1.0 + sat
        growth_metab = np.sum(r_metab * sat / denom_each)
        uptake_metab = c_metab * sat / denom_each

    # toxin: always its own independent term here (own denominator)
    sat_tox = Tox / (K_tox + _EPS)
    denom_tox = 1.0 + sat_tox
    growth_tox = r_tox * sat_tox / denom_tox
    uptake_tox = c_tox * sat_tox / denom_tox

    growth_percapita = growth_metab + growth_tox
    return growth_percapita, uptake_metab[0], uptake_metab[1], uptake_metab[2], uptake_tox


def system_rhs(t, y_log):
    """
    The full ODE right-hand side, integrated in log-space (log N, log A,
    log B, log C, log Toxin) purely so populations/concentrations can
    never go negative during numerical integration -- this is a solver
    detail, not part of the model itself.
    """
    logN1, logN2, logA, logB, logC, logT = y_log
    N1, N2 = np.exp(logN1), np.exp(logN2)
    A, B, C, Tox = np.exp(logA), np.exp(logB), np.exp(logC), np.exp(logT)

    g1, u1A, u1B, u1C, u1T = _strain_growth_and_uptake(
        GROWTH_RATE[0], HALF_SAT[0], CONSUMPTION[0],
        TOXIN_GROWTH_RATE[0], TOXIN_HALF_SAT[0], TOXIN_CONSUMPTION[0],
        A, B, C, Tox, SHARE_UPTAKE,
    )
    g2, u2A, u2B, u2C, u2T = _strain_growth_and_uptake(
        GROWTH_RATE[1], HALF_SAT[1], CONSUMPTION[1],
        TOXIN_GROWTH_RATE[1], TOXIN_HALF_SAT[1], TOXIN_CONSUMPTION[1],
        A, B, C, Tox, SHARE_UPTAKE,
    )

    dN1 = N1 * (g1 - MORTALITY[0])
    dN2 = N2 * (g2 - MORTALITY[1])

    dA = METABOLITE_SUPPLY[0] - METABOLITE_DILUTION[0] * A - N1 * u1A - N2 * u2A
    dB = METABOLITE_SUPPLY[1] - METABOLITE_DILUTION[1] * B - N1 * u1B - N2 * u2B
    dC = METABOLITE_SUPPLY[2] - METABOLITE_DILUTION[2] * C - N1 * u1C - N2 * u2C
    dT = TOXIN_SUPPLY - TOXIN_DECAY * Tox - N1 * u1T - N2 * u2T

    # convert back to log-space derivatives: d(logX)/dt = dX/dt / X
    return [dN1 / (N1 + _EPS), dN2 / (N2 + _EPS),
            dA / (A + _EPS), dB / (B + _EPS), dC / (C + _EPS), dT / (Tox + _EPS)]


# =============================================================================
# RUN
# =============================================================================

y0 = np.maximum(np.array(Y0, dtype=float), _EPS)
t_eval = np.linspace(T_START, T_END, N_POINTS)

sol = solve_ivp(system_rhs, (T_START, T_END), np.log(y0), t_eval=t_eval, method="LSODA", rtol=1e-6, atol=1e-9)
if not sol.success:
    raise RuntimeError(f"integration failed: {sol.message}")
traj = np.exp(sol.y.T)   # (n_points, 6), back to linear space

# =============================================================================
# PLOT + REPORT
# =============================================================================

state_names = STRAIN_NAMES + METABOLITE_NAMES + [TOXIN_NAME]
colors = plt.cm.tab10(np.linspace(0, 1, len(state_names)))

fig, ax = plt.subplots(figsize=(9, 6))
for i, (name, c) in enumerate(zip(state_names, colors)):
    is_strain = name in STRAIN_NAMES
    ax.plot(t_eval, traj[:, i], "-" if is_strain else "--",
            color=c, linewidth=2.5 if is_strain else 1.5, label=name)
ax.set_yscale("log")
ax.set_xlabel("time")
ax.set_ylabel("population / concentration (log scale)")
ax.set_title(f"2-strain, 3-metabolite, 1-toxin system\n(metabolites {'SHARE' if SHARE_UPTAKE else 'do NOT share'} uptake capacity)")
ax.legend(fontsize=9, loc="best")
fig.tight_layout()
fig.savefig("standalone_two_strain_result.png", dpi=150)
print("saved standalone_two_strain_result.png")

print(f"\nFinal values (t={t_eval[-1]}):")
for name, val in zip(state_names, traj[-1]):
    print(f"  {name}: {val:.4g}")
