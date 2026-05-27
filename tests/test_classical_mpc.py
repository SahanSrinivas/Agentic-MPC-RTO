"""Tests for the classical Wood-Berry MPC (Step 4 deliverable).

Includes the fixed-setpoint closed-loop sanity check the project requires before PRBS:
a small xD setpoint step with xB held, verifying offset-free tracking, no constraint
activity, and innovation statistics consistent with sensor noise. The sanity plot under
tests/outputs/ is saved for visual inspection.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agentic_mpc.controllers import ClassicalMPC
from agentic_mpc.plants import WoodBerryParams, WoodBerryPlant

_OUTDIR = pathlib.Path(__file__).parent / "outputs"


def run_closed_loop(*, t_step: float, xD_sp_after: float, t_end: float, dt: float = 1.0,
                    seed: int = 0, disturbance=None, t_disturb: float | None = None):
    """Closed loop: WoodBerryPlant + ClassicalMPC tracking a (possibly stepped) setpoint.

    Returns a dict of time series (t, R, S, xD, xB, xD_sp, xB_sp, innov_xD, innov_xB,
    active) plus the controller/plant objects.
    """
    plant = WoodBerryPlant(dt=dt, seed=seed)
    mpc = ClassicalMPC(dt=dt)
    n = int(round(t_end / dt))
    rec = {k: np.zeros(n + 1) for k in ("t", "R", "S", "xD", "xB", "xD_sp", "xB_sp",
                                        "innov_xD", "innov_xB")}
    active_any = []

    y = np.array([plant.get_state()["y"]["xD"], plant.get_state()["y"]["xB"]])
    for k in range(n + 1):
        now = k * dt
        xD_sp = xD_sp_after if now >= t_step else 0.96
        y_sp = np.array([xD_sp, 0.005])
        if disturbance is not None and t_disturb is not None and now >= t_disturb:
            plant.set_disturbance(**disturbance)
        u = mpc.compute_control(y, y_sp, t=now)
        h = mpc.get_health()
        rec["t"][k], rec["R"][k], rec["S"][k] = now, u[0], u[1]
        rec["xD"][k], rec["xB"][k] = y[0], y[1]
        rec["xD_sp"][k], rec["xB_sp"][k] = xD_sp, 0.005
        rec["innov_xD"][k] = h["innovation_mean"]["xD"]
        rec["innov_xB"][k] = h["innovation_mean"]["xB"]
        active_any += h["active_constraints"]
        y = plant.step(u, dt)
    rec["active_any"] = sorted(set(active_any))
    return rec, plant, mpc


# --------------------------------------------------------------------------------------
def test_dc_gain_matches_nominal_params():
    """The MPC's internal-model DC gain must equal the plant's nominal gain matrix
    exactly (single source of truth, Step-4 spec 1)."""
    mpc = ClassicalMPC(dt=1.0)
    assert np.allclose(mpc._Kdc, WoodBerryParams().gain, atol=1e-9)


def test_fixed_setpoint_tracking_and_plot():
    """xD: 0.96 -> 0.97 step at t=20 (xB held). Offset-free tracking, no constraint
    activity, innovation ~ sensor noise. Saves the sanity plot."""
    dt = 1.0
    rec, plant, mpc = run_closed_loop(t_step=20.0, xD_sp_after=0.97, t_end=150.0, dt=dt)

    last = slice(-20, None)  # last 20 min: steady state
    err_xD = float(np.mean(np.abs(rec["xD"][last] - 0.97)))
    err_xB = float(np.mean(np.abs(rec["xB"][last] - 0.005)))
    assert err_xD < 2e-3, f"steady-state |xD - sp| = {err_xD:.2e} too large"
    assert err_xB < 1e-3, f"steady-state |xB - sp| = {err_xB:.2e} too large"

    # No hard-constraint activity for a small step.
    assert rec["active_any"] == [], f"unexpected active constraints: {rec['active_any']}"

    # Innovation ~ sensor noise: near-zero mean, modest std (post-warmup window).
    innov = np.c_[rec["innov_xD"], rec["innov_xB"]][5:]
    assert np.all(np.abs(innov.mean(axis=0)) < 5e-4), "innovation mean not ~0"
    assert np.all(innov.std(axis=0) < 1e-3), "innovation std much larger than noise"

    # Nonphysical-output guard: trajectory stays in a physically meaningful range.
    assert 0.90 < rec["xD"].min() and rec["xD"].max() < 1.00
    assert -1e-3 < rec["xB"].min() and rec["xB"].max() < 0.02

    _OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(rec["t"], rec["xD"], label="xD")
    ax[0].plot(rec["t"], rec["xD_sp"], "k--", lw=0.8, label="xD setpoint")
    ax[0].plot(rec["t"], rec["xB"], label="xB")
    ax[0].plot(rec["t"], rec["xB_sp"], "grey", ls=":", lw=0.8, label="xB setpoint")
    ax[0].set_ylabel("composition [mole frac]"); ax[0].legend(loc="best")
    ax[0].set_title("Fixed-setpoint sanity: xD 0.96->0.97 at t=20 min (xB held at 0.005)")
    ax[1].plot(rec["t"], rec["R"], label="R (reflux)")
    ax[1].plot(rec["t"], rec["S"], label="S (steam)")
    ax[1].set_ylabel("inputs [lb/min]"); ax[1].legend(loc="best")
    ax[2].plot(rec["t"], rec["innov_xD"], label="innovation xD")
    ax[2].plot(rec["t"], rec["innov_xB"], label="innovation xB")
    ax[2].set_ylabel("innovation [mole frac]"); ax[2].set_xlabel("time [min]")
    ax[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(_OUTDIR / "classical_mpc_fixed_setpoint.png", dpi=110)
    plt.close(fig)
    assert (_OUTDIR / "classical_mpc_fixed_setpoint.png").exists()


def test_health_contract_keys():
    mpc = ClassicalMPC(dt=1.0)
    mpc.compute_control(np.array([0.96, 0.005]), np.array([0.96, 0.005]), t=0.0)
    h = mpc.get_health()
    assert {"innovation_mean", "innovation_std", "active_constraints", "ise_recent"} <= set(h)
    assert set(h["innovation_mean"]) == {"xD", "xB"}


def test_set_targets_updates_default():
    mpc = ClassicalMPC(dt=1.0)
    mpc.set_targets({"xD": 0.97}, rationale="nudge overhead purity up for the test")
    assert mpc.targets["xD"] == 0.97
    assert np.allclose(mpc.target_vector(), [0.97, 0.005])


def test_move_rate_constraint_enforced_on_large_step():
    """A large setpoint jump must be rate-limited: |Delta u| never exceeds du_max."""
    dt = 1.0
    rec, plant, mpc = run_closed_loop(t_step=5.0, xD_sp_after=1.10, t_end=60.0, dt=dt)
    dR = np.abs(np.diff(rec["R"])); dS = np.abs(np.diff(rec["S"]))
    assert dR.max() <= mpc.config.du_max + 1e-6
    assert dS.max() <= mpc.config.du_max + 1e-6
