"""Control-win attribution test for R2 (abrupt fault) & R7 (load disturbance):
integrated economic regret across four supervisors on the SAME seed-matched environment --
  no-agent baseline | rule-based-naive | rule-based-smart | Sonnet-v2 LLM agent.

METRIC (fixed before any results were viewed; identical to T6):
  regret = sum_t  max(P_opt - profit(xD_t, xB_t), 0) * dt        (dt = 1 min)
  profit() = WoodBerryEconomics().profit; P_opt = nominal RTO-optimal objective (constant:
  R2/R7 change the PLANT, not prices/constraints). Lower is better; instantaneous term clipped
  at 0. Applied IDENTICALLY to every arm (same P_opt, same trajectory source, same window).

The point of adding the rule-based arms: if the LLM does not beat the rules on regret, the
R2/R7 control win is attributable to TIMELY RE-OPTIMIZATION, not to the LLM. Verdict per scenario
compares the LLM agent against each rule arm: BEAT / TIE / LOSE, across 3 seeds, beyond seed noise.

Run from repo root:  python analysis/control_regret.py
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

from agentic_mpc.plants import WoodBerryPlant
from agentic_mpc.rto import WoodBerryEconomics, WoodBerryRTO

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "experiments" / "outputs" / "phase1_5"
TBL = ROOT / "analysis" / "tables"
TBL.mkdir(parents=True, exist_ok=True)

SCEN = ["R2", "R7"]
SEEDS = [1, 2, 3]
DT = 1.0
EVENT_T = 100.0

# arm -> output sub-path (under experiments/outputs/phase1_5). All seed-separated (.../seed<N>).
ARMS = {
    "baseline":   "claude_sonnet_4_6/baseline_ma",
    "rule_naive": "rule_based_naive_ma",
    "rule_smart": "rule_based_smart_ma",
    "agentic":    "claude_sonnet_4_6_promptv2/agentic_ma",
}


def nominal_p_opt() -> tuple[float, dict]:
    econ = WoodBerryEconomics()
    plant = WoodBerryPlant(dt=DT, seed=0)
    rto = WoodBerryRTO(economics=econ, plant_params=plant.params, seed=0)
    res = rto.solve()
    sp = res["setpoints"]
    return float(res["objective"]), {"setpoints": sp, "profit_at_opt": econ.profit(sp["xD"], sp["xB"])}


def regret(traj_path: pathlib.Path, econ: WoodBerryEconomics, p_opt: float) -> float:
    d = json.loads(traj_path.read_text())
    xD = np.asarray(d["xD"], float); xB = np.asarray(d["xB"], float)
    inst = np.array([max(p_opt - econ.profit(float(a), float(b)), 0.0) for a, b in zip(xD, xB)])
    return float(inst.sum() * DT)


def n_actions(log_path: pathlib.Path) -> int:
    if not log_path.exists():
        return -1
    return int(json.loads(log_path.read_text()).get("n_agent_actions", -1))


def main() -> None:
    econ = WoodBerryEconomics()
    p_opt, info = nominal_p_opt()
    print(f"P_opt (nominal RTO optimum) = {p_opt:.6f}  at xD={info['setpoints']['xD']:.4f} "
          f"xB={info['setpoints']['xB']:.5f}")
    print(f"PRIMARY metric = full-scenario integrated economic regret; lower is better. dt={DT} min.\n")

    # collect regret[arm][scenario] = list over seeds; also per (scen,seed) row
    data: dict[str, dict[str, list[float]]] = {a: {s: [] for s in SCEN} for a in ARMS}
    acts: dict[str, list[int]] = {s: [] for s in SCEN}
    rows = []
    for scen in SCEN:
        for s in SEEDS:
            row = {"scenario": scen, "seed": s}
            ok = True
            for arm, sub in ARMS.items():
                tp = P / sub / scen / f"seed{s}" / "trajectory.json"
                if not tp.exists():
                    print(f"  !! MISSING {arm} {scen} seed{s} ({tp})"); ok = False; continue
                r = regret(tp, econ, p_opt)
                row[arm] = r
                data[arm][scen].append(r)
            if not ok:
                continue
            row["d_naive"] = row["rule_naive"] - row["baseline"]
            row["d_smart"] = row["rule_smart"] - row["baseline"]
            row["d_agent"] = row["agentic"] - row["baseline"]
            row["agent_vs_smart"] = row["agentic"] - row["rule_smart"]
            row["actions"] = n_actions(P / ARMS["agentic"] / scen / f"seed{s}" / "log.json")
            acts[scen].append(row["actions"])
            rows.append(row)

    # ---- per (scenario, seed) 4-arm table ----
    cols = ["baseline", "rule_naive", "rule_smart", "agentic"]
    hdr = (f"{'scn':<4}{'sd':<3}" + "".join(f"{c:>12}" for c in cols)
           + f"{'d_agent':>10}{'ag-smart':>10}{'act':>5}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['scenario']:<4}{r['seed']:<3}" + "".join(f"{r[c]:>12.4f}" for c in cols)
              + f"{r['d_agent']:>+10.4f}{r['agent_vs_smart']:>+10.4f}{r['actions']:>5}")

    # ---- per-scenario means + LLM-vs-rules verdict ----
    print("\nPer-scenario means (regret, lower=better) and LLM-vs-rule verdict:")
    for scen in SCEN:
        means = {a: float(np.mean(data[a][scen])) for a in ARMS}
        stds = {a: float(np.std(data[a][scen], ddof=1)) for a in ARMS}
        print(f"\n  {scen}:  " + " | ".join(f"{a}={means[a]:.4f}+/-{stds[a]:.4f}" for a in ARMS))
        for rule in ("rule_naive", "rule_smart"):
            d = np.array(data["agentic"][scen]) - np.array(data[rule][scen])   # agent - rule
            noise = max(stds["agentic"], stds[rule])                            # seed-to-seed noise floor
            consistent_better = bool((d < 0).all())
            consistent_worse = bool((d > 0).all())
            beyond = abs(float(d.mean())) > noise
            if consistent_better and beyond:
                v = "LLM BEATS"
            elif consistent_worse and beyond:
                v = "LLM LOSES"
            else:
                v = "TIE (within seed noise / inconsistent sign)"
            print(f"    agent vs {rule}: per-seed delta {np.round(d, 4).tolist()} "
                  f"mean {d.mean():+.4f} (noise~{noise:.4f}) -> {v}")

    # ---- CSV ----
    out = TBL / "T7_control_regret_4arm_r2_r7.csv"
    fields = ["scenario", "seed", "baseline", "rule_naive", "rule_smart", "agentic",
              "d_naive", "d_smart", "d_agent", "agent_vs_smart", "actions"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 6) if isinstance(r.get(k), float) else r.get(k)) for k in fields})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
