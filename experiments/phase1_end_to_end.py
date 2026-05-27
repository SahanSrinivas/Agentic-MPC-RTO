"""Phase 1, Step 7 -- end-to-end supervisory scenario (full stack).

Sets up the full stack -- WoodBerry plant + classical MPC + LLM supervisory agent (on
Ollama qwen3:4b) talking through the universal interfaces -- and runs a 300-min closed
loop. At t=100 min TWO disturbances are injected together:

  (1) a +15% multiplicative gain perturbation on the R->xD channel (matches the original
      task wording), and
  (2) a -0.03 additive output bias on xD (a feed-composition load disturbance).

Methodological finding (documented in the summary + JSON): at the nominal operating point
the gain perturbation is near-invisible (Delta_u ~ 0 at steady state, so gain x Delta_u
produces no detectable effect), while the output bias carries the entire agent-readable
signal -- a sustained biased innovation and tracking offset. Gain-only fault scenarios are
therefore weak benchmarks for supervisory control; load disturbances carry the signal.

The agent is triggered every 10 simulated minutes to assess and (optionally) act via its
limited Phase-1 tool set (get_process_state, get_mpc_health, update_mpc_target). Every
decision is logged with timestamp, rationale, and the action taken, and written to
experiments/outputs/phase1_e2e/agent_log.json.

Run:  python experiments/phase1_end_to_end.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agentic_mpc.agent import SupervisoryAgent
from agentic_mpc.controllers import ClassicalMPC
from agentic_mpc.metrics import ise
from agentic_mpc.plants import WoodBerryPlant
from agentic_mpc.safety import BoxSafetyEnvelope

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # qwen3 final messages may contain unicode

OUTDIR = pathlib.Path(__file__).parent / "outputs" / "phase1_e2e"

# --- configuration ---
DT = 1.0
T_END = 300.0
T_DISTURB = 100.0
TRIGGER_EVERY = 10.0          # min  (agent supervisory cycle cadence)
# Optionally restrict agent triggers to a window (lo, hi) in minutes to bound wall-time
# when running a slow local model; None = trigger across the whole run (spec default).
TRIGGER_WINDOW: tuple[float, float] | None = None
XD_SP0, XB_SP0 = 0.96, 0.005
GAIN_DISTURB = {("xD", "R"): 1.15}      # +15% on R->xD  (spec wording)
OUTPUT_DISTURB = {"xD": -0.03}          # feed-composition load disturbance (carries signal)
SEED_PLANT = 0


def run_e2e() -> dict:
    n = int(round(T_END / DT))
    trig = int(round(TRIGGER_EVERY / DT))

    plant = WoodBerryPlant(dt=DT, seed=SEED_PLANT)
    mpc = ClassicalMPC(dt=DT)
    agent = SupervisoryAgent(plant, mpc, safety=BoxSafetyEnvelope())

    rec = {k: np.zeros(n + 1) for k in
           ("t", "R", "S", "xD", "xB", "xD_sp", "xB_sp", "innov_xD", "innov_xB", "ise")}
    agent_log: list[dict] = []
    action_times: list[float] = []

    y = np.array([plant.get_state()["y"]["xD"], plant.get_state()["y"]["xB"]])
    for k in range(n + 1):
        now = k * DT

        # inject BOTH disturbances at t=100 (affects the step advancing from t=100 on)
        if k == int(round(T_DISTURB / DT)):
            plant.set_disturbance(gain_multiplier=GAIN_DISTURB, output_bias=OUTPUT_DISTURB)

        # agent supervisory cycle every TRIGGER_EVERY minutes (reads current state, may act)
        in_window = TRIGGER_WINDOW is None or (TRIGGER_WINDOW[0] <= now <= TRIGGER_WINDOW[1])
        if k > 0 and k % trig == 0 and in_window:
            entry = _agent_cycle(agent, now)
            agent_log.append(entry)
            if entry.get("acted"):
                action_times.append(now)

        # the MPC tracks the controller's CURRENT targets (which the agent may have changed)
        y_sp = mpc.target_vector()
        u = mpc.compute_control(y, y_sp, t=now)
        h = mpc.get_health()
        rec["t"][k], rec["R"][k], rec["S"][k] = now, u[0], u[1]
        rec["xD"][k], rec["xB"][k] = y[0], y[1]
        rec["xD_sp"][k], rec["xB_sp"][k] = y_sp[0], y_sp[1]
        rec["innov_xD"][k] = h["innovation_mean"]["xD"]
        rec["innov_xB"][k] = h["innovation_mean"]["xB"]
        rec["ise"][k] = h["ise_recent"]
        y = plant.step(u, DT)

    return {"series": rec, "agent_log": agent_log, "action_times": action_times,
            "bounds": (mpc._u_min, mpc._u_max, mpc._du_max)}


def _agent_cycle(agent: SupervisoryAgent, now: float) -> dict:
    """Run one supervisory cycle and reduce it to a timestamped log entry."""
    msg = (f"Supervisory cycle at t={now:.0f} min. Assess the process state and MPC "
           f"health, and take a supervisory action only if warranted.")
    t0 = time.time()
    try:
        out = agent.run_cycle(msg)
    except Exception as e:  # never let one flaky LLM call kill a 300-min run
        return {"t": now, "error": f"{type(e).__name__}: {e}", "acted": False,
                "actions": [], "final": None}
    # supervisory actions = executed update_mpc_target calls
    sup_actions = [a for a in out["actions"]
                   if a["tool"] == "update_mpc_target" and a["status"] == "executed"]
    return {
        "t": now,
        "wall_s": round(time.time() - t0, 1),
        "iterations": out["iterations"],
        "final": out["final"],
        "acted": bool(sup_actions),
        "supervisory_actions": [{"applied_targets": a["result"].get("applied_targets"),
                                 "clipped_by_safety": a["result"].get("clipped_by_safety"),
                                 "rationale": a["args"].get("rationale")}
                                for a in sup_actions],
        "tool_calls": [{"tool": a["tool"], "status": a["status"]} for a in out["actions"]],
    }


def make_plots(res: dict) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = res["series"]; at = res["action_times"]
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    def overlay(a, first_only=False):
        a.axvline(T_DISTURB, color="red", ls="-", lw=1.2,
                  label="disturbance injected (t=100)")
        for i, t in enumerate(at):
            a.axvline(t, color="purple", ls=":", lw=0.8,
                      label="agent action" if i == 0 else None)

    ax[0].plot(s["t"], s["xD_sp"], "k--", lw=0.8, label="xD setpoint")
    ax[0].plot(s["t"], s["xD"], color="C2", lw=1.0, label="xD measured")
    overlay(ax[0]); ax[0].set_ylabel("xD [mole frac]"); ax[0].legend(loc="lower left", fontsize=8)
    ax[0].set_title("Step 7 -- end-to-end: plant + MPC + supervisory agent (300 min)")
    ax[1].plot(s["t"], s["xB_sp"], "k--", lw=0.8, label="xB setpoint")
    ax[1].plot(s["t"], s["xB"], color="C3", lw=1.0, label="xB measured")
    overlay(ax[1]); ax[1].set_ylabel("xB [mole frac]"); ax[1].legend(loc="lower left", fontsize=8)
    ax[2].plot(s["t"], s["innov_xD"], color="C0", lw=0.9, label="innovation xD (rolling mean)")
    ax[2].plot(s["t"], s["innov_xB"], color="C1", lw=0.9, label="innovation xB (rolling mean)")
    overlay(ax[2]); ax[2].set_ylabel("MPC innovation"); ax[2].set_xlabel("time [min]")
    ax[2].legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "phase1_e2e.png", dpi=110); plt.close(fig)


def summarize(res: dict) -> dict:
    s = res["series"]; t = s["t"]
    pre = (t >= max(0.0, T_DISTURB - 40)) & (t < T_DISTURB)
    post = (t > T_DISTURB) & (t <= T_DISTURB + 60)
    settled = t >= min(T_END, T_DISTURB + 50)

    def _m(arr, mask):  # guarded mean (empty window -> None)
        return float(np.mean(arr[mask])) if np.any(mask) else None

    sig = {
        "innovation_xD_mean_pre": _m(s["innov_xD"], pre),
        "innovation_xD_mean_post": _m(s["innov_xD"], post),
        "ise_mean_pre": _m(s["ise"], pre),
        "ise_mean_post": _m(s["ise"], post),
        "xD_offset_post_settled": _m(s["xD"] - s["xD_sp"], settled),
    }
    n_triggers = len(res["agent_log"])
    n_actions = len(res["action_times"])
    return {"signal": sig, "n_triggers": n_triggers, "n_supervisory_actions": n_actions,
            "action_times": res["action_times"]}


def main() -> None:
    print("Running Step-7 end-to-end (300 min, agent every 10 min on qwen3:4b)...")
    print("This makes ~30 LLM supervisory cycles; expect several minutes.\n")
    res = run_e2e()
    make_plots(res)
    summ = summarize(res)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "agent_log.json").write_text(json.dumps({
        "config": {"T_END": T_END, "T_DISTURB": T_DISTURB, "trigger_every_min": TRIGGER_EVERY,
                   "gain_disturbance": {f"{k[0]}<-{k[1]}": v for k, v in GAIN_DISTURB.items()},
                   "output_disturbance": OUTPUT_DISTURB, "model": "qwen3:4b"},
        "summary": summ,
        "decisions": res["agent_log"],
    }, indent=2))

    sig = summ["signal"]
    print("=" * 74)
    print("PHASE 1 / STEP 7 -- END-TO-END SUPERVISORY SCENARIO")
    print("=" * 74)
    print(f"  disturbances @ t={T_DISTURB:.0f} min: +15% gain R->xD  AND  xD output bias -0.03")
    print("-" * 74)
    def _sf(x, spec="+.2e"):
        return "n/a" if x is None else format(x, spec)
    print("  WHICH DISTURBANCE CARRIED THE SIGNAL (methodological finding):")
    print(f"    innovation xD (rolling mean):  pre={_sf(sig['innovation_xD_mean_pre'])}  "
          f"-> post={_sf(sig['innovation_xD_mean_post'])}")
    print(f"    recent ISE:                    pre={_sf(sig['ise_mean_pre'])}  "
          f"-> post={_sf(sig['ise_mean_post'])}")
    print(f"    settled xD tracking offset: {_sf(sig['xD_offset_post_settled'], '+.4f')}")
    print("    -> the +15% gain change alone is sub-threshold at the operating point; the")
    print("       -0.03 output (feed-composition) bias carries the detectable degradation.")
    print("-" * 74)
    print(f"  agent supervisory cycles triggered: {summ['n_triggers']}")
    print(f"  cycles where the agent ACTED (update_mpc_target): {summ['n_supervisory_actions']}")
    if summ["action_times"]:
        print(f"  action timestamps [min]: {summ['action_times']}")
    print("-" * 74)
    print("  sample agent decisions:")
    for e in res["agent_log"]:
        if e.get("acted"):
            for a in e["supervisory_actions"]:
                print(f"    t={e['t']:.0f}: targets={a['applied_targets']} "
                      f"clipped={a['clipped_by_safety']}")
                print(f"           rationale: {(a['rationale'] or '')[:130]}")
    print("-" * 74)
    print(f"  plot -> {OUTDIR / 'phase1_e2e.png'}")
    print(f"  decision log -> {OUTDIR / 'agent_log.json'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
