"""Control-win test for R2 (abrupt fault) & R7 (load disturbance): integrated economic regret,
agentic-v2 (Sonnet) vs seed-matched no-agent baseline.

METRIC (fixed before results were viewed):
  PRIMARY = integrated economic regret over the full scenario:
      regret = sum_t  max(P_opt - profit(xD_t, xB_t), 0) * dt        (dt = 1 min)
  where profit() is WoodBerryEconomics().profit and P_opt is the nominal RTO-optimal objective
  (constant: R2/R7 change the PLANT, not prices/constraints, so the economically-optimal
  composition value is unchanged). Lower is better. The instantaneous term is clipped at 0 so a
  lucky instant cannot mask regret elsewhere. Applied identically to agentic and baseline.

A control win (per scenario) requires BOTH: (a) agentic regret below baseline on all 3 seeds,
and (b) by a margin larger than the baseline seed-to-seed spread.

Run from repo root:  python analysis/control_regret.py
"""
from __future__ import annotations

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
EVENT_T = 100.0                                   # disturbance onset (for the auxiliary post-event view)

AGENTIC_DIR = "claude_sonnet_4_6_promptv2/agentic_ma"   # prompt v2, Sonnet
BASELINE_DIR = "claude_sonnet_4_6/baseline_ma"          # seed-matched, no LLM


def nominal_p_opt() -> tuple[float, dict]:
    """P_opt = nominal RTO-optimal objective (best-achievable instantaneous profit at nominal prices)."""
    econ = WoodBerryEconomics()
    plant = WoodBerryPlant(dt=DT, seed=0)
    rto = WoodBerryRTO(economics=econ, plant_params=plant.params, seed=0)
    res = rto.solve()
    sp = res["setpoints"]
    # cross-check the RTO objective against profit() evaluated at its own optimum (same scale).
    p_from_profit = econ.profit(sp["xD"], sp["xB"])
    return float(res["objective"]), {"setpoints": sp, "profit_at_opt": p_from_profit}


def regret(traj_path: pathlib.Path, econ: WoodBerryEconomics, p_opt: float):
    """Integrated economic regret (full scenario) + auxiliary post-event regret (t >= EVENT_T)."""
    d = json.loads(traj_path.read_text())
    t = np.asarray(d["t"], float)
    xD = np.asarray(d["xD"], float)
    xB = np.asarray(d["xB"], float)
    inst = np.array([max(p_opt - econ.profit(float(a), float(b)), 0.0) for a, b in zip(xD, xB)])
    full = float(inst.sum() * DT)
    post = float(inst[t >= EVENT_T].sum() * DT)
    return full, post


def n_actions(log_path: pathlib.Path) -> int:
    if not log_path.exists():
        return -1
    d = json.loads(log_path.read_text())
    return int(d.get("n_agent_actions", -1))


def main() -> None:
    econ = WoodBerryEconomics()
    p_opt, info = nominal_p_opt()
    print(f"P_opt (nominal RTO optimum) = {p_opt:.6f}  at setpoints "
          f"xD={info['setpoints']['xD']:.4f} xB={info['setpoints']['xB']:.5f}  "
          f"(profit() at opt = {info['profit_at_opt']:.6f})")
    print(f"PRIMARY metric = full-scenario integrated economic regret; lower is better. dt={DT} min.\n")

    rows = []
    for scen in SCEN:
        for s in SEEDS:
            b_traj = P / BASELINE_DIR / scen / f"seed{s}" / "trajectory.json"
            a_traj = P / AGENTIC_DIR / scen / f"seed{s}" / "trajectory.json"
            a_log = P / AGENTIC_DIR / scen / f"seed{s}" / "log.json"
            if not (b_traj.exists() and a_traj.exists()):
                print(f"  !! MISSING trajectory for {scen} seed{s} "
                      f"(baseline={b_traj.exists()} agentic={a_traj.exists()})")
                continue
            b_full, b_post = regret(b_traj, econ, p_opt)
            a_full, a_post = regret(a_traj, econ, p_opt)
            acts = n_actions(a_log)
            rows.append(dict(scenario=scen, seed=s, baseline=b_full, agentic=a_full,
                             delta=a_full - b_full, base_post=b_post, agent_post=a_post,
                             actions=acts))

    # ---- per (scenario, seed) table ----
    hdr = f"{'scen':<5}{'seed':<5}{'baseline':>12}{'agentic':>12}{'delta':>12}{'actions':>9}  win?"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        win = "YES" if r["delta"] < 0 else "no"
        print(f"{r['scenario']:<5}{r['seed']:<5}{r['baseline']:>12.4f}{r['agentic']:>12.4f}"
              f"{r['delta']:>+12.4f}{r['actions']:>9}  {win}")

    # ---- per-scenario mean +/- spread ----
    print("\nPer-scenario summary (full-scenario regret; mean +/- sample std across 3 seeds):")
    for scen in SCEN:
        sr = [r for r in rows if r["scenario"] == scen]
        if not sr:
            continue
        b = np.array([r["baseline"] for r in sr]); a = np.array([r["agentic"] for r in sr])
        dl = np.array([r["delta"] for r in sr]); ac = [r["actions"] for r in sr]
        bsd = float(b.std(ddof=1)) if len(b) > 1 else 0.0
        improved_all = bool((dl < 0).all())
        margin_ok = bool((-dl.mean()) > bsd)              # mean improvement exceeds baseline spread
        verdict = ("CONTROL WIN" if (improved_all and margin_ok)
                   else "NO WIN -- improvement within seed noise" if improved_all
                   else "NO WIN -- not consistent across seeds")
        print(f"  {scen}: baseline {b.mean():.4f}+/-{bsd:.4f} | agentic {a.mean():.4f}+/-"
              f"{a.std(ddof=1) if len(a)>1 else 0:.4f} | mean delta {dl.mean():+.4f} "
              f"(all-seeds-improved={improved_all}, margin>spread={margin_ok}) | actions={ac}")
        print(f"      -> {verdict}")

    # ---- CSV ----
    import csv
    out = TBL / "T6_control_regret_r2_r7.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario", "seed", "baseline", "agentic", "delta",
                                           "base_post", "agent_post", "actions"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
