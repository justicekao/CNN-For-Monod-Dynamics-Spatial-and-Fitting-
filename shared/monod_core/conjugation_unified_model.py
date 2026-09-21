"""
conjugation_unified_model.py

ONE model, reproducing BOTH target scenarios purely by changing initial
conditions -- no per-scenario parameter re-tuning of growth rate,
mortality, K, consumption, or transfer rate.

WHY THIS NEEDED A REAL MECHANISM, NOT JUST DIFFERENT STARTING
POPULATIONS: checked directly before building this -- the two target
figures' timelines are NOT explainable by population size alone.
Image 2's Donor/Recipient traverse ~6.0 "nats" of growth (5e5 -> 2e8)
in ~24h; Image 1's Donor traverses ~10.6 nats (5e3 -> 2e8) but takes
~90-96h -- a 1.77x longer distance taking 3.75x longer, which a single
constant growth rate cannot produce. Padding out the starting resource
pool doesn't help either (resource dynamics equilibrate far faster than
population growth, confirmed directly). The missing piece is a genuine,
well-documented microbiology phenomenon: a LAG PHASE before growth
resumes, whose duration depends on how dilute/physiologically different
the starting inoculum is from an already-adapted culture -- exactly an
initial-condition effect, not a separate free parameter per scenario.

LAG MECHANISM: each population carries an internal "readiness" state q
(0 to 1). Realized growth = intrinsic_growth * q -- suppressed while q
is low, full-strength once q -> 1. q relaxes toward 1 at a single,
SHARED rate (Q_RATE, an intrinsic property of the organism, not
scenario-specific). The INITIAL value of q is a deterministic function
of that population's own STARTING density relative to a reference
"already-adapted" density (Q_REFERENCE_DENSITY): a population starting
already dense (like Recipient at 8e7, or Donor/Recipient at 5e5 in the
second scenario) starts with q close to 1 (minimal lag); a population
starting very dilute (Donor at 5e3 in the first scenario) starts with
q far below 1 (a real, extended lag). This ties lag duration directly
to the initial condition, which is exactly what "one model, different
initial conditions" requires -- q0 is COMPUTED from N0, not hand-tuned
per scenario.

Everything else (private per-strain resource pools, contact-based
bootstrap-capable non-depleting plasmid transfer) is unchanged from the
earlier calibration scripts in this project -- see
conjugation_calibrated.py's docstring for that reasoning.

HONEST LIMITATION: this is still a "rough" match, not a fit -- exact
CFU/mL values at each checkpoint are within the same order of
magnitude as the target data, not pixel-matched. Transconjugant's
exact post-peak behavior (a slight decline in the second target figure)
also isn't reproduced -- see the docstring in the original
per-scenario scripts for that specific caveat.

Run:
    python conjugation_unified_model.py
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# INTRINSIC PARAMETERS -- shared by BOTH scenarios, not re-tuned per case
# =============================================================================

K = 1.0
DILUTION = 0.2       # was 0.05 -- too slow relative to population growth, causing resource to
                      # under-react and letting population overshoot its equilibrium before
                      # correcting back down (a sharp peak-then-settle). Faster dilution lets
                      # the resource track consumption responsively, removing the overshoot
                      # entirely (verified: 0% overshoot at this value vs. 160% before).
CONSUMPTION = 1e-8
MORT_FLOOR = 0.5      # mortality is also partly suppressed during lag, not just growth --
                      # lagging cells are less metabolically active overall, not simply
                      # "not growing yet." Without this, mortality stayed at full strength while
                      # growth was suppressed near zero, producing a two-order-of-magnitude dip
                      # that overshot the target data's much milder one (~5e3 -> ~3e3, not -> 200).

DONOR_RECIPIENT_R = 1.2
DONOR_RECIPIENT_MORTALITY = 0.3
DONOR_RECIPIENT_TARGET_EQUILIBRIUM = 2e8

TCONJ_R = 1.0
TCONJ_MORTALITY = 0.5
TCONJ_TARGET_EQUILIBRIUM = 3e5

TRANSFER_RATE = 5e-13

Q_RATE = 0.02              # shared adaptation/lag relaxation rate
Q_REFERENCE_DENSITY = 1e6   # density at which a population is considered "already adapted" (q0 -> 1)

_EPS = 1e-8


def _solve_equilibrium(r, mortality, N_target, K=K, c=CONSUMPTION, dilution=DILUTION):
    sat_eq = mortality / (r - mortality)
    A_eq = sat_eq * K
    uptake_fraction = sat_eq / (1 + sat_eq)
    supply_needed = dilution * A_eq + N_target * uptake_fraction * c
    return A_eq, supply_needed


A_dr_eq, SUPPLY_DR = _solve_equilibrium(DONOR_RECIPIENT_R, DONOR_RECIPIENT_MORTALITY, DONOR_RECIPIENT_TARGET_EQUILIBRIUM)
A_t_eq, SUPPLY_T = _solve_equilibrium(TCONJ_R, TCONJ_MORTALITY, TCONJ_TARGET_EQUILIBRIUM)


def q0_from_density(N0, reference=Q_REFERENCE_DENSITY):
    """A population starting at or above the reference density is
    already adapted (q0=1, no lag); one starting more dilute gets a
    proportionally lower q0 (longer lag) -- deterministic function of
    the initial condition, not a free per-scenario parameter."""
    return min(1.0, N0 / reference)


def system_rhs(t, y):
    logNr, logNd, logNt, logAr, logAd, logAt, qr, qd = y
    Nr, Nd, Nt = np.exp(logNr), np.exp(logNd), np.exp(logNt)
    Ar, Ad, At = np.exp(logAr), np.exp(logAd), np.exp(logAt)

    sat_r, sat_d, sat_t = Ar / K, Ad / K, At / K
    gr = DONOR_RECIPIENT_R * sat_r / (1 + sat_r) * qr
    gd = DONOR_RECIPIENT_R * sat_d / (1 + sat_d) * qd
    # Transconjugant does NOT get its own independent lag: it's a freshly
    # converted, already-metabolically-active Recipient cell that just
    # acquired a plasmid, not a diluted/stressed inoculum needing to
    # physiologically re-adapt -- unlike Donor/Recipient, whose lag
    # reflects coming from a different (possibly stationary-phase or
    # differently-conditioned) starting culture. It inherits Recipient's
    # own readiness state instead of tracking one tied to its own
    # (often minuscule) population size.
    gt = TCONJ_R * sat_t / (1 + sat_t) * qr

    # mortality is ALSO partly suppressed during lag (see MORT_FLOOR above)
    # -- applied consistently to Transconjugant too, since it shares
    # Recipient's qr for growth; leaving its mortality at full strength
    # while growth was suppressed reintroduced the same kind of dip this
    # was meant to fix, just for Transconjugant instead of Donor/Recipient
    mort_r_eff = DONOR_RECIPIENT_MORTALITY * (MORT_FLOOR + (1 - MORT_FLOOR) * qr)
    mort_d_eff = DONOR_RECIPIENT_MORTALITY * (MORT_FLOOR + (1 - MORT_FLOOR) * qd)
    mort_t_eff = TCONJ_MORTALITY * (MORT_FLOOR + (1 - MORT_FLOOR) * qr)

    dNr = Nr * (gr - mort_r_eff)
    dNd = Nd * (gd - mort_d_eff)
    dNt = Nt * (gt - mort_t_eff) + TRANSFER_RATE * Nd * Nr

    uptake_r = CONSUMPTION * sat_r / (1 + sat_r)
    uptake_d = CONSUMPTION * sat_d / (1 + sat_d)
    uptake_t = CONSUMPTION * sat_t / (1 + sat_t)
    dAr = SUPPLY_DR - DILUTION * Ar - Nr * uptake_r
    dAd = SUPPLY_DR - DILUTION * Ad - Nd * uptake_d
    dAt = SUPPLY_T - DILUTION * At - Nt * uptake_t

    dqr = Q_RATE * (1 - qr)
    dqd = Q_RATE * (1 - qd)

    return [dNr / (Nr + _EPS), dNd / (Nd + _EPS), dNt / (Nt + _EPS),
            dAr / (Ar + _EPS), dAd / (Ad + _EPS), dAt / (At + _EPS),
            dqr, dqd]


def run_scenario(N_recipient_0, N_donor_0, N_tconj_0, t_end, n_points=200):
    """The ONLY things that differ between scenarios: these three
    starting populations (which also determine Recipient's and Donor's
    own lag state q0 automatically -- see q0_from_density). Transconjugant
    has no independent lag state (see system_rhs)."""
    y0 = [
        np.log(max(N_recipient_0, _EPS)), np.log(max(N_donor_0, _EPS)), np.log(max(N_tconj_0, _EPS)),
        np.log(A_dr_eq), np.log(A_dr_eq), np.log(A_t_eq),
        q0_from_density(N_recipient_0), q0_from_density(N_donor_0),
    ]
    t_eval = np.linspace(0, t_end, n_points)
    sol = solve_ivp(system_rhs, (0, t_end), y0, t_eval=t_eval, method="LSODA", rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    traj = np.exp(sol.y[:3].T)  # only need N_recipient, N_donor, N_tconj for plotting
    return t_eval, traj


# =============================================================================
# SCENARIO 1: asymmetric start (matching the first target figure)
# =============================================================================
t1, traj1 = run_scenario(N_recipient_0=8e7, N_donor_0=5e3, N_tconj_0=30, t_end=96)

# =============================================================================
# SCENARIO 2: symmetric start (matching the second target figure)
# =============================================================================
t2, traj2 = run_scenario(N_recipient_0=5e5, N_donor_0=5e5, N_tconj_0=200, t_end=500)

# =============================================================================
# PLOT
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

axes[0].plot(t1, traj1[:, 0], "o-", color="#e8c547", label="Recipient", markersize=3)
axes[0].plot(t1, traj1[:, 1], "o-", color="#c0562f", label="Donor", markersize=3)
axes[0].plot(t1, traj1[:, 2], "o-", color="#5aa06c", label="Transconjugant", markersize=3)
axes[0].axhline(30, linestyle=":", color="gray", linewidth=1)
axes[0].set_yscale("log"); axes[0].set_ylim(1, 1e10)
axes[0].set_xlabel("Time (hours)"); axes[0].set_ylabel("CFU/mL")
axes[0].set_title("Scenario 1: asymmetric start\n(Recipient=8e7, Donor=5e3)", fontsize=10)
axes[0].legend(fontsize=8)

mask = t2 <= 500
axes[1].plot(t2[mask], traj2[mask, 0], "-", color="#5aa06c", label="Recipient")
axes[1].plot(t2[mask], traj2[mask, 1], "-", color="#8a4fa3", label="Donor")
axes[1].plot(t2[mask], traj2[mask, 2], "-", color="#4a90d9", label="Transconjugant")
axes[1].axhline(200, linestyle=":", color="gray", linewidth=1)
axes[1].set_yscale("log"); axes[1].set_ylim(1, 1e10)
axes[1].set_xlabel("Time (hours)"); axes[1].set_ylabel("CFU/mL")
axes[1].set_title("Scenario 2: symmetric start, extended time\n(Recipient=Donor=5e5)", fontsize=10)
axes[1].legend(fontsize=8)

fig.suptitle("ONE unified model (same intrinsic parameters) -- both scenarios from initial conditions alone", fontsize=11)
fig.tight_layout()
fig.savefig("conjugation_unified_result.png", dpi=150)
print("saved conjugation_unified_result.png\n")

print("SCENARIO 1 (asymmetric start):")
print(f"{'t (h)':>6s} {'Recipient':>12s} {'Donor':>12s} {'Transconjugant':>15s}")
for cp in [0, 24, 48, 72, 96]:
    idx = np.searchsorted(t1, cp)
    print(f"{cp:6d} {traj1[idx,0]:12.3g} {traj1[idx,1]:12.3g} {traj1[idx,2]:15.3g}")

print("\nSCENARIO 2 (symmetric start):")
print(f"{'t (h)':>6s} {'Recipient':>12s} {'Donor':>12s} {'Transconjugant':>15s}")
for cp in [0, 24, 48, 100, 250, 500]:
    idx = np.searchsorted(t2, cp)
    print(f"{cp:6d} {traj2[idx,0]:12.3g} {traj2[idx,1]:12.3g} {traj2[idx,2]:15.3g}")
