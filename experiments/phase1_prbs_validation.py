"""Phase 1, Step 5 -- classical-MPC validation on the Wood-Berry plant (two-part).

This is the "MPC is in place and working" deliverable. It deliberately answers two
DIFFERENT questions with two experiments, because conflating them at a dwell time near
the bottom loop's transport delay is self-contradictory:

  PART A -- PRBS excitation tracking (does the MPC track fast multi-level setpoint
            variation safely?). A 600-min, dwell-10-min, 5-level pseudo-random setpoint
            perturbation. At dwell ~ xB's 7-min R->xB transport delay, this run is
            deliberately DEADTIME-STRESSED: the bottom loop cannot settle between steps,
            so residual error is physical, not a controller deficiency. The meaningful
            criteria here are ISE reduction vs a do-nothing baseline and ZERO
            hard-constraint violations. Settling time is NOT assessed from this run --
            it is the wrong metric at this dwell.

  PART B -- clean held-step settling (how fast does the MPC settle a sustained setpoint
            change, vs the 30-min design bar?). Four single-step scenarios held 80 min
            each. Settling time is a single-step concept; this is the right place to
            measure it.

PRBS amplitudes are scaled to each output's nominal operating envelope to maintain
physical realism while exercising both loops and their interaction: xD setpoint +/-0.02
(0.94-0.98 mole frac) and xB setpoint +/-0.002 (0.003-0.007 mole frac).

Overall Step-5 verdict = PART A PASS and PART B PASS (all four scenarios).

Run:  python experiments/phase1_prbs_validation.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agentic_mpc.controllers import ClassicalMPC
from agentic_mpc.metrics import iae, ise, settling_time
from agentic_mpc.plants import WoodBerryPlant

OUTDIR = pathlib.Path(__file__).parent / "outputs" / "phase1_prbs"

# --- shared configuration ---
DT = 1.0                 # min
XD_SP0, XB_SP0 = 0.96, 0.005
# PART A (PRBS)
T_END = 600.0
DWELL = 10.0             # min  (per spec; deliberately deadtime-stressed for xB)
XD_AMP, XB_AMP = 0.02, 0.002
XD_LEVELS = np.linspace(-XD_AMP, XD_AMP, 5)   # multi-level: {-A,-A/2,0,A/2,A}
XB_LEVELS = np.linspace(-XB_AMP, XB_AMP, 5)
SEED_PLANT, SEED_XD, SEED_XB = 0, 11, 22
ISE_REDUCTION_BAR = 0.50  # PART A: >=50% ISE reduction vs do-nothing (deadtime regime)
# PART B (clean steps)
STEP_T = 10.0            # step applied at t=10 min
STEP_T_END = 80.0        # 80-min window
SETTLE_BAR_MIN = 30.0    # design settling bar


# ======================================================================================
# PART A -- PRBS excitation tracking
# ======================================================================================
def multilevel_prbs(n: int, dwell_steps: int, levels: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """Dwell-time-held multi-level pseudo-random sequence (seeded, reproducible)."""
    n_blocks = int(np.ceil(n / dwell_steps))
    block_vals = rng.choice(levels, size=n_blocks)
    return np.repeat(block_vals, dwell_steps)[:n]


def run_prbs() -> dict:
    n = int(round(T_END / DT))
    dwell_steps = int(round(DWELL / DT))
    prbs_xD = multilevel_prbs(n + 1, dwell_steps, XD_LEVELS, np.random.default_rng(SEED_XD))
    prbs_xB = multilevel_prbs(n + 1, dwell_steps, XB_LEVELS, np.random.default_rng(SEED_XB))

    plant = WoodBerryPlant(dt=DT, seed=SEED_PLANT)
    mpc = ClassicalMPC(dt=DT)
    u_min, u_max, du_max = mpc._u_min, mpc._u_max, mpc._du_max

    t = np.zeros(n + 1); R = np.zeros(n + 1); S = np.zeros(n + 1)
    xD = np.zeros(n + 1); xB = np.zeros(n + 1)
    xD_sp = XD_SP0 + prbs_xD; xB_sp = XB_SP0 + prbs_xB
    dU = np.zeros(n + 1); violation = np.zeros(n + 1)

    y = np.array([plant.get_state()["y"]["xD"], plant.get_state()["y"]["xB"]])
    u_prev = plant.params.u_nominal.copy()
    for k in range(n + 1):
        u = mpc.compute_control(y, np.array([xD_sp[k], xB_sp[k]]), t=k * DT)
        du = np.abs(u - u_prev)
        viol = np.concatenate([u - u_max, u_min - u, du - du_max])
        t[k], R[k], S[k], xD[k], xB[k] = k * DT, u[0], u[1], y[0], y[1]
        dU[k] = du.sum(); violation[k] = max(0.0, float(viol.max()))
        u_prev = u.copy()
        y = plant.step(u, DT)

    eD, eB = xD - xD_sp, xB - xB_sp
    ise_donothing = ise(prbs_xD[:n + 1], DT) + ise(prbs_xB[:n + 1], DT)
    ise_closed = ise(np.c_[eD, eB], DT)
    metrics = {
        "ISE_total": ise_closed, "ISE_xD": ise(eD, DT), "ISE_xB": ise(eB, DT),
        "IAE_total": iae(np.c_[eD, eB], DT),
        "control_effort_total_abs_du": float(dU.sum()),
        "control_effort_R": float(np.abs(np.diff(R)).sum()),
        "control_effort_S": float(np.abs(np.diff(S)).sum()),
        "max_constraint_violation": float(violation.max()),
        "ise_donothing_reference": ise_donothing,
        "ise_reduction_vs_donothing": 1.0 - ise_closed / ise_donothing,
    }
    series = dict(t=t, R=R, S=S, xD=xD, xB=xB, xD_sp=xD_sp, xB_sp=xB_sp, dU=dU)
    return {"metrics": metrics, "series": series, "bounds": (u_min, u_max, du_max)}


def prbs_verdict(m: dict) -> tuple[bool, dict, str]:
    """ISE reduction >= 50% (deadtime-dominated regime) + zero constraint violations.

    Justification for the 50% bar: at dwell (10 min) ~ xB's transport delay (7 min), the
    bottom loop cannot settle between steps, so a floor of the ISE is the unavoidable
    deadtime response -- physical, not controller deficiency. A >=50% reduction vs the
    do-nothing baseline (output frozen at nominal) certifies that the controller is
    delivering genuine tracking value above that floor under fast excitation.
    """
    ise_thresh = (1.0 - ISE_REDUCTION_BAR) * m["ise_donothing_reference"]
    checks = {
        "ISE_reduction_ge_50pct": (m["ISE_total"] < ise_thresh, m["ISE_total"], ise_thresh),
        "zero_constraint_violations": (m["max_constraint_violation"] <= 1e-9,
                                       m["max_constraint_violation"], 0.0),
    }
    return all(c[0] for c in checks.values()), checks, \
        f"ISE threshold = {1-ISE_REDUCTION_BAR:.2f} x do-nothing ISE = {ise_thresh:.4g}"


# ======================================================================================
# PART B -- clean held-step settling
# ======================================================================================
SCENARIOS = [
    {"name": "xD_up", "changed": "xD", "xD": 0.98, "xB": 0.005},
    {"name": "xD_down", "changed": "xD", "xD": 0.94, "xB": 0.005},
    {"name": "xB_up", "changed": "xB", "xD": 0.96, "xB": 0.007},
    {"name": "xB_down", "changed": "xB", "xD": 0.96, "xB": 0.003},
]


def run_clean_step(sc: dict) -> dict:
    dt = DT
    plant = WoodBerryPlant(dt=dt, seed=SEED_PLANT)
    mpc = ClassicalMPC(dt=dt)
    u_min, u_max, du_max = mpc._u_min, mpc._u_max, mpc._du_max
    n = int(round(STEP_T_END / dt))
    rec = {k: np.zeros(n + 1) for k in ("t", "R", "S", "xD", "xB", "xD_sp", "xB_sp")}
    viol = 0.0
    y = np.array([plant.get_state()["y"]["xD"], plant.get_state()["y"]["xB"]])
    u_prev = plant.params.u_nominal.copy()
    for k in range(n + 1):
        now = k * dt
        xD_sp = sc["xD"] if now >= STEP_T else XD_SP0
        xB_sp = sc["xB"] if now >= STEP_T else XB_SP0
        u = mpc.compute_control(y, np.array([xD_sp, xB_sp]), t=now)
        viol = max(viol, float(np.concatenate(
            [u - u_max, u_min - u, np.abs(u - u_prev) - du_max]).max()))
        rec["t"][k], rec["R"][k], rec["S"][k] = now, u[0], u[1]
        rec["xD"][k], rec["xB"][k] = y[0], y[1]
        rec["xD_sp"][k], rec["xB_sp"][k] = xD_sp, xB_sp
        u_prev = u.copy()
        y = plant.step(u, dt)

    changed = sc["changed"]
    base = XD_SP0 if changed == "xD" else XB_SP0
    target = sc[changed]
    step_mag = abs(target - base)
    band = max(0.05 * step_mag, 3.0 * plant.meas_noise_std)
    ts = settling_time(rec["t"], rec[changed], target, band, t_start=STEP_T)
    return {
        "name": sc["name"], "changed": changed, "target": target, "step_mag": step_mag,
        "band": band, "settling_min": ts, "max_constraint_violation": max(0.0, viol),
        "series": rec,
    }


def cleanstep_verdict(results: list[dict]) -> tuple[bool, dict]:
    checks = {}
    for r in results:
        ok = (r["settling_min"] is not None and r["settling_min"] < SETTLE_BAR_MIN
              and r["max_constraint_violation"] <= 1e-9)
        checks[r["name"]] = (ok, r["settling_min"], SETTLE_BAR_MIN)
    return all(c[0] for c in checks.values()), checks


# ======================================================================================
# Plots
# ======================================================================================
def make_prbs_plots(res: dict) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = res["series"]; u_min, u_max, du_max = res["bounds"]

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(s["t"], s["xD_sp"], "k--", lw=0.8, label="xD setpoint")
    ax[0].plot(s["t"], s["xD"], color="C2", lw=0.9, label="xD")
    ax[0].set_ylabel("xD [mole frac]"); ax[0].legend(loc="upper right")
    ax[0].set_title("PART A -- PRBS excitation tracking (600 min, dwell 10 min)")
    ax[1].plot(s["t"], s["xB_sp"], "k--", lw=0.8, label="xB setpoint")
    ax[1].plot(s["t"], s["xB"], color="C3", lw=0.9, label="xB")
    ax[1].set_ylabel("xB [mole frac]"); ax[1].set_xlabel("time [min]")
    ax[1].legend(loc="upper right")
    fig.tight_layout(); fig.savefig(OUTDIR / "prbs_outputs.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for a, sig, name, c in [(ax[0], "R", "R (reflux)", "C0"), (ax[1], "S", "S (steam)", "C1")]:
        a.plot(s["t"], s[sig], color=c, lw=0.9, label=name)
        a.axhline(u_min[0], color="grey", ls=":", lw=0.8)
        a.axhline(u_max[0], color="grey", ls=":", lw=0.8, label="hard bounds [0.5, 3.0]")
        a.set_ylabel(f"{sig} [lb/min]"); a.legend(loc="upper right")
    ax[0].set_title("PART A -- manipulated inputs vs hard bounds")
    ax[1].set_xlabel("time [min]")
    fig.tight_layout(); fig.savefig(OUTDIR / "prbs_inputs.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(s["t"], s["dU"], color="C4", lw=0.8)
    ax[0].axhline(du_max, color="grey", ls=":", lw=0.8, label=f"|Du| limit/input = {du_max}")
    ax[0].set_ylabel("|DR|+|DS| per step"); ax[0].legend(loc="upper right")
    ax[0].set_title("PART A -- control effort")
    ax[1].plot(s["t"], np.cumsum(s["dU"]), color="C5", lw=1.0)
    ax[1].set_ylabel("cumulative integral|Du|"); ax[1].set_xlabel("time [min]")
    fig.tight_layout(); fig.savefig(OUTDIR / "prbs_control_effort.png", dpi=110); plt.close(fig)


def make_cleanstep_plot(results: list[dict]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for r, ax in zip(results, axes.ravel()):
        ch = r["changed"]; s = r["series"]; sp = s[f"{ch}_sp"]
        ax.plot(s["t"], s[ch], color="C2" if ch == "xD" else "C3", lw=1.1, label=ch)
        ax.plot(s["t"], sp, "k--", lw=0.8, label="setpoint")
        ax.fill_between(s["t"], r["target"] - r["band"], r["target"] + r["band"],
                        color="grey", alpha=0.25, label=f"+/-band ({r['band']:.1e})")
        ts = r["settling_min"]
        if ts is not None:
            ax.axvline(STEP_T + ts, color="C1", ls="-.", lw=0.9,
                       label=f"settled @ {ts:.0f} min")
        ax.set_title(f"{r['name']}: {ch} step {r['step_mag']:.3f}")
        ax.set_ylabel(f"{ch} [mole frac]"); ax.legend(loc="best", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("time [min]")
    fig.suptitle("PART B -- clean held-step settling (80 min each; bar = 30 min)")
    fig.tight_layout(); fig.savefig(OUTDIR / "clean_step_settling.png", dpi=110); plt.close(fig)


# ======================================================================================
def main() -> None:
    # PART A
    res = run_prbs(); mA = res["metrics"]
    make_prbs_plots(res)
    passA, checksA, ise_note = prbs_verdict(mA)
    # PART B
    resultsB = [run_clean_step(sc) for sc in SCENARIOS]
    make_cleanstep_plot(resultsB)
    passB, checksB = cleanstep_verdict(resultsB)

    overall = passA and passB
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "prbs_metrics.json").write_text(json.dumps({
        "part_a_prbs": {"metrics": mA, "pass": passA,
                        "checks": {k: {"pass": v[0], "value": v[1], "threshold": v[2]}
                                   for k, v in checksA.items()}},
        "part_b_clean_steps": {"pass": passB,
            "scenarios": [{"name": r["name"], "changed": r["changed"],
                           "step_mag": r["step_mag"], "band": r["band"],
                           "settling_min": r["settling_min"],
                           "max_constraint_violation": r["max_constraint_violation"]}
                          for r in resultsB]},
        "overall_pass": overall,
    }, indent=2))

    print("=" * 74)
    print("PHASE 1 / STEP 5 -- CLASSICAL-MPC VALIDATION (two-part)")
    print("=" * 74)
    print("  NOTE: PART A (PRBS, dwell 10 min) is deliberately deadtime-stressed -- it")
    print("  validates EXCITATION TRACKING + constraint safety, where at dwell ~ xB's")
    print("  7-min transport delay the residual ISE is physical, not a controller flaw.")
    print("  PART B (clean held steps) validates SETTLING vs the 30-min bar. The two are")
    print("  kept separate on purpose so the metrics are not conflated.")
    print("-" * 74)
    print("PART A -- PRBS excitation tracking (600 min, dwell 10 min, 5-level):")
    print(f"  amplitudes: xD +/-{XD_AMP} (0.94-0.98), xB +/-{XB_AMP} (0.003-0.007)")
    print(f"  ISE total = {mA['ISE_total']:.5g}  (xD {mA['ISE_xD']:.4g}, xB {mA['ISE_xB']:.4g})")
    print(f"  IAE total = {mA['IAE_total']:.5g}")
    print(f"  ISE reduction vs do-nothing = {100*mA['ise_reduction_vs_donothing']:.1f}%")
    print(f"  control effort int|Du| = {mA['control_effort_total_abs_du']:.4g}  "
          f"(R {mA['control_effort_R']:.3g}, S {mA['control_effort_S']:.3g})")
    print(f"  max constraint violation = {mA['max_constraint_violation']:.2e}")
    print(f"  {ise_note}")
    for name, (ok, val, thr) in checksA.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: value={_fmt(val)} threshold={_fmt(thr)}")
    print(f"  PART A: {'PASS' if passA else 'FAIL'}")
    print("-" * 74)
    print("PART B -- clean held-step settling (80 min each; band = max(5% step, 3*noise)):")
    for r in resultsB:
        print(f"    [{'PASS' if checksB[r['name']][0] else 'FAIL'}] {r['name']:8s} "
              f"({r['changed']} step {r['step_mag']:.3f}, band {r['band']:.1e}): "
              f"settling = {_fmt(r['settling_min'])} min  (bar {SETTLE_BAR_MIN:.0f}), "
              f"max viol {r['max_constraint_violation']:.1e}")
    print(f"  PART B: {'PASS' if passB else 'FAIL'}")
    print("-" * 74)
    print(f"  OVERALL STEP-5 VERDICT: {'PASS' if overall else 'FAIL'}")
    print(f"  plots + metrics -> {OUTDIR}")
    print("=" * 74)


def _fmt(x) -> str:
    return "n/a" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))


if __name__ == "__main__":
    main()
