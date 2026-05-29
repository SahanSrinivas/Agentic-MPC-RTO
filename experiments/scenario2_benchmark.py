"""Scenario 2 benchmark: drive the MPC sim sandbox through reproducible mismatch episodes and show
the DiagnosticSupervisor's decision on each, using only the MCP tool surface
(get_mpc_diagnostics / get_plant_snapshot / set_target).

Cases (event injected at t=100, deterministic, seed-fixed):
  nominal              -> HOLD              (no disturbance)
  sensor_fault         -> VETO_HOLD          (xD analyzer +0.05 -> reading > 1, physically impossible)
  coupled_load         -> PROPOSE_SETPOINT   (output bias on BOTH xD & xB -> coupled real disturbance)
  ambiguous_load       -> ESCALATE          (single-channel xD load, in-range -> not separable)
  out_of_range_request -> CLIP              (a setpoint of xD=1.2 is clipped to 0.99, logged)

Outputs: a thesis figure (disturbance -> offset + innovation -> veto / bounded action) and a results
table. The supervisor is deterministic (no LLM), so this is fully reproducible.

Run:  python experiments/scenario2_benchmark.py
"""
from __future__ import annotations

import json
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agentic_mpc.mcp_sandbox import MPCSandbox
from agentic_mpc.scenario2_agent import DiagnosticSupervisor

warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).parent / "outputs" / "scenario2"
OUT.mkdir(parents=True, exist_ok=True)

T_EVENT, T_END, STEP, SEED = 100, 200, 5, 1
INJECT = {
    "nominal":        lambda sb: None,
    "sensor_fault":   lambda sb: sb.plant.set_sensor_bias({"xD": 0.05}),
    "coupled_load":   lambda sb: sb.plant.set_disturbance(output_bias={"xD": -0.02, "xB": 0.012}),
    "ambiguous_load": lambda sb: sb.plant.set_disturbance(output_bias={"xD": -0.03}),
}
EXPECTED = {"nominal": ("NOMINAL", "HOLD"), "sensor_fault": ("SENSOR_FAULT", "VETO_HOLD"),
            "coupled_load": ("REAL_DISTURBANCE", "PROPOSE_SETPOINT"),
            "ambiguous_load": ("AMBIGUOUS", "ESCALATE")}


def run_episode(name) -> dict:
    sb = MPCSandbox(seed=SEED)
    sup = DiagnosticSupervisor()
    series, injected, acted = [], False, False
    t = 0
    while t < T_END:
        sb.advance(STEP); t += STEP
        if not injected and t >= T_EVENT:
            INJECT[name](sb); injected = True
        diag, snap = sb.get_mpc_diagnostics(), sb.get_plant_snapshot()
        dec = sup.assess(diag, snap)
        clip = None
        if dec.action == "PROPOSE_SETPOINT" and not acted:    # apply ONE bounded, clipped move
            clip = sb.set_target(**dec.proposed_targets, rationale=dec.rationale)
            acted = True
        series.append({"t": t, "innov_xD": diag["innovation_mean"]["xD"],
                       "innov_xB": diag["innovation_mean"]["xB"],
                       "off_xD": diag["steady_state_offset"]["xD"],
                       "off_xB": diag["steady_state_offset"]["xB"],
                       "state": dec.state, "action": dec.action, "applied": clip})
    final = next(s for s in reversed(series) if s["t"] >= T_EVENT + 20)   # settled post-event decision
    return {"name": name, "series": series, "final_state": final["state"],
            "final_action": final["action"], "final_rationale": dec.rationale,
            "proposed": dec.proposed_targets}


def out_of_range_case() -> dict:
    """Safety-boundary check, independent of the classifier: an out-of-box setpoint is clipped."""
    sb = MPCSandbox(seed=SEED); sb.advance(30)
    r = sb.set_target(xD=1.2, xB=0.004, rationale="benchmark: deliberately out-of-range request")
    return {"name": "out_of_range_request", "clipped_by_safety": r["clipped_by_safety"],
            "applied_targets": r["applied_targets"]}


def thesis_figure(eps: dict) -> None:
    """disturbance -> offset + innovation -> veto / bounded action, for the two key episodes."""
    show = ["sensor_fault", "coupled_load"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for j, name in enumerate(show):
        s = eps[name]["series"]; t = [r["t"] for r in s]
        # row 0: the exposed signals
        a0 = ax[0, j]
        a0.plot(t, [r["innov_xD"] for r in s], "C3", label="innovation xD")
        a0.plot(t, [r["off_xD"] for r in s], "C0", label="offset xD (y-y_sp)")
        a0.axhline(5e-4, color="grey", ls=":", lw=0.7); a0.axhline(-5e-4, color="grey", ls=":", lw=0.7)
        a0.axvline(T_EVENT, color="purple", ls="--", lw=0.8, label="disturbance")
        a0.set_title(f"{name}: MPC exposes mismatch"); a0.legend(fontsize=7, loc="best")
        a0.set_ylabel("xD signal")
        # row 1: the agent decision over time (categorical)
        a1 = ax[1, j]
        order = ["HOLD", "ESCALATE", "PROPOSE_SETPOINT", "VETO_HOLD"]
        yv = [order.index(r["action"]) for r in s]
        a1.step(t, yv, where="post", color="C2")
        applied_t = [r["t"] for r in s if r["applied"] is not None]
        for at in applied_t:
            a1.axvline(at, color="C1", ls="-", lw=0.8)
        a1.axvline(T_EVENT, color="purple", ls="--", lw=0.8)
        a1.set_yticks(range(len(order))); a1.set_yticklabels(order, fontsize=7)
        a1.set_xlabel("time [min]"); a1.set_title(f"decision -> {eps[name]['final_action']}")
    fig.suptitle("Scenario 2: disturbance -> innovation + offset (no-integral MPC) -> veto / bounded action")
    fig.tight_layout(); fig.savefig(OUT / "thesis_figure.png", dpi=130); plt.close(fig)


def main() -> None:
    eps = {name: run_episode(name) for name in INJECT}
    oor = out_of_range_case()
    thesis_figure(eps)

    print("=" * 78)
    print(f"{'case':<22}{'final_state':<18}{'action':<18}{'expected?'}")
    print("-" * 78)
    all_ok = True
    rows = []
    for name, e in eps.items():
        exp_s, exp_a = EXPECTED[name]
        ok = (e["final_state"] == exp_s and e["final_action"] == exp_a)
        all_ok &= ok
        print(f"{name:<22}{e['final_state']:<18}{e['final_action']:<18}{'OK' if ok else 'MISS '+exp_s+'/'+exp_a}")
        rows.append({"case": name, "final_state": e["final_state"], "final_action": e["final_action"],
                     "expected": ok, "proposed_targets": e["proposed"]})
    oor_ok = oor["clipped_by_safety"] and oor["applied_targets"]["xD"] == 0.99
    all_ok &= oor_ok
    print(f"{'out_of_range_request':<22}{'(safety clip)':<18}"
          f"{'xD->'+str(oor['applied_targets']['xD']):<18}{'OK' if oor_ok else 'MISS'}")
    rows.append({"case": "out_of_range_request", "clipped_by_safety": oor["clipped_by_safety"],
                 "applied_targets": oor["applied_targets"], "expected": oor_ok})
    print("=" * 78)
    print(f"ALL CASES AS EXPECTED: {all_ok}")
    print(f"thesis figure -> {OUT / 'thesis_figure.png'}")
    (OUT / "benchmark_results.json").write_text(json.dumps(
        {"all_ok": bool(all_ok), "cases": rows,
         "rationales": {n: e["final_rationale"] for n, e in eps.items()}}, indent=2, default=str))


if __name__ == "__main__":
    main()
