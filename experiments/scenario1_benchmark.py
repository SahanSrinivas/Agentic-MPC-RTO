"""Scenario 1 (lite) benchmark: diagnostics-GATED economics over the MPC sim sandbox.

Three reproducible cases (event at t=100):
  green          -> economics runs every cycle (diagnostics NOMINAL); setpoint moved to the economic
                    optimum (clipped to the envelope); realized margin improves.
  coupled_load   -> economics runs while green, then Scenario 2 flags REAL_DISTURBANCE at the event
                    and the economic move is BLOCKED (Scenario 2 governs; S1 does not chase).
  ambiguous_load -> single-channel mismatch -> Scenario 2 ESCALATES -> economics BLOCKED.

Shows the "Scenario 2 vetoes Scenario 1" interlock without any RTO. Deterministic (no LLM).

Run:  python experiments/scenario1_benchmark.py
"""
from __future__ import annotations

import json
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic_mpc.mcp_sandbox import MPCSandbox
from agentic_mpc.scenario1_agent import EconWeights, Scenario1Agent, margin

warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).parent / "outputs" / "scenario1"
OUT.mkdir(parents=True, exist_ok=True)

T_EVENT, T_END, STEP, SEED = 100, 200, 5, 1
W = EconWeights()
INJECT = {
    "green":          lambda sb: None,
    "coupled_load":   lambda sb: sb.plant.set_disturbance(output_bias={"xD": -0.02, "xB": 0.012}),
    "ambiguous_load": lambda sb: sb.plant.set_disturbance(output_bias={"xD": -0.03}),
}


def run_episode(name) -> dict:
    sb = MPCSandbox(seed=SEED)
    agent = Scenario1Agent(weights=W)
    series, injected = [], False
    t = 0
    while t < T_END:
        sb.advance(STEP); t += STEP
        if not injected and t >= T_EVENT:
            INJECT[name](sb); injected = True
        diag, snap = sb.get_mpc_diagnostics(), sb.get_plant_snapshot()
        res = agent.step(diag, snap, sandbox=sb)
        y, u = snap["y"], snap["u"]
        series.append({"t": t, "J": margin(y["xD"], y["xB"], u["S"], W),
                       "gated": res["gated"], "s2_state": res["s2_state"],
                       "applied": res["applied"] is not None})
    return {"name": name, "series": series}


def _bands(ax, s):
    """Shade the time axis green where economics ran, red where it was gated by Scenario 2."""
    for i in range(len(s)):
        t0 = s[i]["t"] - STEP
        ax.axvspan(t0, s[i]["t"], color=("#fde0e0" if s[i]["gated"] else "#e2f3e2"), lw=0)


def figure(eps: dict) -> None:
    names = ["green", "coupled_load", "ambiguous_load"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for k, name in enumerate(names):
        s = eps[name]["series"]; t = [r["t"] for r in s]
        a = ax[k]
        _bands(a, s)
        a.plot(t, [r["J"] for r in s], "C0", lw=1.4, label="realized margin J")
        applied_t = [r["t"] for r in s if r["applied"]]
        if applied_t:
            a.scatter(applied_t, [next(r["J"] for r in s if r["t"] == at) for at in applied_t],
                      s=12, color="C2", zorder=5, label="economic set_target applied")
        a.axvline(T_EVENT, color="purple", ls="--", lw=0.8, label="disturbance")
        a.set_ylabel("margin J"); a.set_title(f"{name}", loc="left", fontsize=10)
        a.legend(fontsize=7, loc="best")
    ax[-1].set_xlabel("time [min]")
    fig.suptitle("Scenario 1 (lite): diagnostics-gated economics  "
                 "(green = economics active, red = gated by Scenario 2)")
    fig.tight_layout(); fig.savefig(OUT / "scenario1_figure.png", dpi=130); plt.close(fig)


def main() -> None:
    eps = {name: run_episode(name) for name in INJECT}
    figure(eps)

    def post(s):    # decision state after the event has settled
        return [r for r in s if r["t"] >= T_EVENT + 20]

    print("=" * 76)
    print(f"{'case':<16}{'pre-event economics':<22}{'post-event (>=120)':<26}{'ok?'}")
    print("-" * 76)
    rows, all_ok = [], True
    for name, e in eps.items():
        s = e["series"]
        pre_applied = any(r["applied"] for r in s if r["t"] < T_EVENT)
        post_gated = all(r["gated"] for r in post(s))
        post_state = post(s)[-1]["s2_state"]
        if name == "green":
            ok = (not post_gated) and any(r["applied"] for r in s)     # economics keeps running
            desc_pre, desc_post = "applied", f"active ({post_state})"
        else:
            ok = pre_applied and post_gated                            # ran while green, blocked after
            desc_pre, desc_post = "applied" if pre_applied else "none", f"GATED ({post_state})"
        all_ok &= ok
        print(f"{name:<16}{desc_pre:<22}{desc_post:<26}{'OK' if ok else 'MISS'}")
        rows.append({"case": name, "pre_event_applied": pre_applied,
                     "post_event_gated": post_gated, "post_state": post_state, "ok": ok})
    # margins: green should improve vs its own start; gated cases should not be chased economically
    jg = eps["green"]["series"]
    dJ = jg[-1]["J"] - jg[0]["J"]
    print("-" * 76)
    print(f"green margin gain: J0={jg[0]['J']:.4f} -> Jend={jg[-1]['J']:.4f}  (dJ={dJ:+.4f})")
    print(f"ALL CASES AS EXPECTED: {all_ok and dJ > 0}")
    print(f"figure -> {OUT / 'scenario1_figure.png'}")
    (OUT / "benchmark_results.json").write_text(json.dumps(
        {"all_ok": bool(all_ok and dJ > 0), "green_margin_gain": dJ, "cases": rows,
         "weights": vars(W)}, indent=2, default=str))


if __name__ == "__main__":
    main()
