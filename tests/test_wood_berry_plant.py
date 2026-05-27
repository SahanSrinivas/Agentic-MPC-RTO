"""Validation tests for the Wood-Berry plant (Step 3 deliverable).

The headline test is an open-loop step on R (S held at nominal): it confirms the
steady-state gain matches the published 12.8 +/- 5% and that the response respects the
1-min transport delay, and it saves a step-response plot for visual inspection. The
plot under tests/outputs/ is part of the paper's "the plant is correct" evidence.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")  # headless: no display needed for CI / saved figures
import matplotlib.pyplot as plt
import numpy as np
import pytest

from agentic_mpc.plants import WoodBerryPlant

_OUTDIR = pathlib.Path(__file__).parent / "outputs"


def _simulate_step(plant: WoodBerryPlant, *, t_step: float, dt: float, t_end: float,
                   u_before: np.ndarray, u_after: np.ndarray):
    """Run an open-loop step and return (t, R, S, xD, xB) arrays of measured outputs."""
    n = int(round(t_end / dt))
    t = np.zeros(n + 1)
    R = np.zeros(n + 1); S = np.zeros(n + 1)
    xD = np.zeros(n + 1); xB = np.zeros(n + 1)
    # record the t=0 state from the seeded history
    st0 = plant.get_state()
    t[0], xD[0], xB[0] = st0["t"], st0["y"]["xD"], st0["y"]["xB"]
    R[0], S[0] = st0["u"]["R"], st0["u"]["S"]
    for k in range(1, n + 1):
        now = k * dt
        u = u_after if now > t_step else u_before
        y = plant.step(u, dt)
        t[k], xD[k], xB[k] = now, y[_idx(plant, "xD")], y[_idx(plant, "xB")]
        R[k], S[k] = u
    return t, R, S, xD, xB


def _idx(plant: WoodBerryPlant, name: str) -> int:
    return plant.metadata["output_names"].index(name)


# --------------------------------------------------------------------------------------
# Headline validation: open-loop step on R, S held; SS gain 12.8 +/- 5% + plot.
# --------------------------------------------------------------------------------------
def test_open_loop_step_response_gain_and_plot():
    dt, t_step, t_end = 1.0, 10.0, 100.0
    dR = 0.10
    plant = WoodBerryPlant(dt=dt, seed=0)
    u_before = np.array([1.95, 1.71])
    u_after = np.array([1.95 + dR, 1.71])  # S held at nominal

    t, R, S, xD, xB = _simulate_step(plant, t_step=t_step, dt=dt, t_end=t_end,
                                     u_before=u_before, u_after=u_after)

    # Steady-state deviation after 90 min of settling (tau_xD,R = 16.7 -> ~e^-6 settled).
    dxD_ss = xD[-1] - 0.96
    gain_xD = dxD_ss / dR
    assert 12.8 * 0.95 <= gain_xD <= 12.8 * 1.05, f"xD/R SS gain {gain_xD:.3f} not 12.8 +/-5%"

    # Bonus: bottoms also responds to R with gain 6.6 (7-min delay) -- free extra check.
    gain_xB = (xB[-1] - 0.005) / dR
    assert 6.6 * 0.95 <= gain_xB <= 6.6 * 1.05, f"xB/R SS gain {gain_xB:.3f} not 6.6 +/-5%"

    # --- save the step-response plot ---
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, R, label="R (reflux)"); axes[0].plot(t, S, label="S (steam)")
    axes[0].axvline(t_step, color="k", ls=":", lw=0.8)
    axes[0].set_ylabel("inputs [lb/min]"); axes[0].legend(loc="best")
    axes[0].set_title("Wood-Berry open-loop step: R 1.95 -> 2.05 at t=10 min (S held)")
    axes[1].plot(t, xD, color="C2")
    axes[1].axhline(0.96 + 12.8 * dR, color="grey", ls="--", lw=0.8,
                    label=f"predicted SS = 0.96 + 12.8x{dR} = {0.96 + 12.8 * dR:.3f}")
    axes[1].set_ylabel("xD [mole frac]"); axes[1].legend(loc="best")
    axes[2].plot(t, xB, color="C3")
    axes[2].axhline(0.005 + 6.6 * dR, color="grey", ls="--", lw=0.8,
                    label=f"predicted SS = 0.005 + 6.6x{dR} = {0.005 + 6.6 * dR:.3f}")
    axes[2].set_ylabel("xB [mole frac]"); axes[2].set_xlabel("time [min]")
    axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(_OUTDIR / "wood_berry_step_response.png", dpi=110)
    plt.close(fig)
    assert (_OUTDIR / "wood_berry_step_response.png").exists()


def test_transport_delay_respected():
    """xD must not move within its 1-min deadtime: at the sample at t = t_step + 1 it
    is still ~baseline (the continuous response is exactly 0 at t = theta)."""
    dt, t_step = 1.0, 10.0
    plant = WoodBerryPlant(dt=dt, meas_noise_std=0.0, seed=0)  # noise off for a crisp check
    u_before, u_after = np.array([1.95, 1.71]), np.array([2.05, 1.71])
    t, R, S, xD, xB = _simulate_step(plant, t_step=t_step, dt=dt, t_end=20.0,
                                     u_before=u_before, u_after=u_after)
    # index of t == t_step + 1 (== theta after the step): still at baseline
    k_delay = int(round((t_step + 1.0) / dt))
    assert abs(xD[k_delay] - 0.96) < 1e-9, "xD moved before its 1-min transport delay"
    # one sample later it has started to rise
    assert xD[k_delay + 1] - 0.96 > 1e-4, "xD failed to respond after the delay"


# --------------------------------------------------------------------------------------
# Interface / housekeeping
# --------------------------------------------------------------------------------------
def test_metadata_contract():
    plant = WoodBerryPlant(history_window=25, dt=1.0)
    md = plant.metadata
    assert md["input_names"] == ["R", "S"]
    assert md["output_names"] == ["xD", "xB"]
    assert len(md["input_units"]) == 2 and len(md["output_units"]) == 2
    assert md["history_window_samples"] == 25
    assert md["history_window_duration"] == 25 * md["dt"]
    assert md["time_units"] == "min"


def test_history_window_is_configurable_and_self_describing():
    plant = WoodBerryPlant(history_window=10, dt=1.0, seed=1)
    for _ in range(50):
        plant.step(np.array([1.95, 1.71]), 1.0)
    st = plant.get_state()
    # window holds exactly the configured number of samples (after warm-up) ...
    assert len(st["history"]["t"]) == 10
    assert len(st["history"]["y"]["xD"]) == 10
    # ... and is self-describing: timestamps are present and monotonically increasing
    ts = st["history"]["t"]
    assert all(ts[i] < ts[i + 1] for i in range(len(ts) - 1))


def test_step_rejects_mismatched_dt():
    plant = WoodBerryPlant(dt=1.0)
    with pytest.raises(ValueError):
        plant.step(np.array([1.95, 1.71]), dt=0.5)


def test_disturbance_hook_changes_only_the_plant_gain():
    """A +15% gain bump on R->xD (the Step-7 disturbance) scales the SS gain by 1.15;
    other channels are untouched. The controller's model is not involved here."""
    dt, dR = 1.0, 0.10
    plant = WoodBerryPlant(dt=dt, seed=0)
    plant.set_disturbance(gain_multiplier={("xD", "R"): 1.15})
    u_before, u_after = np.array([1.95, 1.71]), np.array([1.95 + dR, 1.71])
    t, R, S, xD, xB = _simulate_step(plant, t_step=10.0, dt=dt, t_end=120.0,
                                     u_before=u_before, u_after=u_after)
    gain_xD = (xD[-1] - 0.96) / dR
    assert 12.8 * 1.15 * 0.95 <= gain_xD <= 12.8 * 1.15 * 1.05, \
        f"disturbed xD/R gain {gain_xD:.3f} != ~{12.8 * 1.15:.3f}"
    # xB/R channel was NOT disturbed -> still ~6.6
    gain_xB = (xB[-1] - 0.005) / dR
    assert 6.6 * 0.95 <= gain_xB <= 6.6 * 1.05
