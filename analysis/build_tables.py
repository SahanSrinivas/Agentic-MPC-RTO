"""Phase 1.5 results analysis: extract per-run metrics from experiments/outputs/phase1_5/,
build the master table, and emit the comparison tables (CSV) + markdown + a printed digest.

Run:  python analysis/build_tables.py
Outputs: analysis/tables/T1..T5*.csv  and prints markdown tables + a comparison digest.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "experiments" / "outputs" / "phase1_5"
TBL = ROOT / "analysis" / "tables"
TBL.mkdir(parents=True, exist_ok=True)

# (output dir, supervisor label, rto variant)
CONFIGS = [
    ("qwen3_30b/agentic_ma", "agentic", "ma"),
    ("qwen3_30b/agentic_ma-gp", "agentic", "ma-gp"),
    ("qwen3_30b/baseline_ma", "baseline", "ma"),
    ("qwen3_30b/baseline_ma-gp", "baseline", "ma-gp"),
    ("rule_based_naive_ma", "rule_naive", "ma"),
    ("rule_based_naive_ma-gp", "rule_naive", "ma-gp"),
    ("rule_based_smart_ma", "rule_smart", "ma"),
    ("rule_based_smart_ma-gp", "rule_smart", "ma-gp"),
]
SCEN = [f"R{i}" for i in range(1, 8)]


def extract() -> pd.DataFrame:
    rows = []
    for cdir, sup, rto in CONFIGS:
        for s in SCEN:
            lj = P / cdir / s / "log.json"
            rec = {"supervisor": sup, "rto": rto, "scenario": s, "present": False}
            if lj.exists() and lj.stat().st_size > 0:
                try:
                    d = json.loads(lj.read_text())
                except Exception:
                    rows.append(rec); continue
                h = d.get("rto_handoffs", [])
                last_conv = (h[-1].get("rto_status", {}).get("converged") if h else None)
                any_infeas = any(hd.get("rto_status", {}).get("converged") is False for hd in h)
                fr, fs = d.get("final_realized", {}), d.get("final_setpoint", {})
                trig = upd = 0
                for dec in d.get("agent_decisions", []):
                    for a in dec.get("actions", []):
                        if a.get("status") == "executed" and a.get("tool") == "trigger_rto_run":
                            trig += 1
                        elif a.get("status") == "executed" and a.get("tool") == "update_mpc_target":
                            upd += 1
                rec.update(present=True, n_decisions=d.get("n_agent_decisions"),
                           n_actions=d.get("n_agent_actions"), n_rto=d.get("n_rto_commands"),
                           rxD=fr.get("xD"), rxB=fr.get("xB"), sxD=fs.get("xD"), sxB=fs.get("xB"),
                           final_converged=last_conv, any_infeasible=any_infeas,
                           n_trigger=trig, n_update=upd)
            rows.append(rec)
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main() -> None:
    df = extract()
    present = df[df["present"]].copy()

    def cfg(sup, rto):
        return present[(present.supervisor == sup) & (present.rto == rto)].set_index("scenario")

    # ---- T1 / T1b: agent action count by scenario (agentic only) ----
    tests = {"R1": "slow drift - mostly hold", "R2": "abrupt fault - intervene",
             "R3": "steam-cost spike - respond", "R4": "demand cap - RTO drives",
             "R5": "infeasibility - recognize/hold", "R6": "sensor anomaly - diagnose",
             "R7": "load disturbance - detect MPC stress"}
    for rto, fn in (("ma", "T1_agent_actions_by_scenario"), ("ma-gp", "T1b_agent_actions_ma-gp")):
        a = cfg("agentic", rto)
        t = pd.DataFrame({"scenario": SCEN,
                          f"agentic_{rto}_actions": [int(a.loc[s, "n_actions"]) if s in a.index else None for s in SCEN],
                          "n_decisions": [int(a.loc[s, "n_decisions"]) if s in a.index else None for s in SCEN],
                          "what_this_tests": [tests[s] for s in SCEN]})
        t.to_csv(TBL / f"{fn}.csv", index=False)
        print(f"\n### {fn}\n"); print(_md(t))

    # ---- T2: final operating points, all 8 configs x 7 scenarios ----
    t2 = pd.DataFrame({"scenario": SCEN})
    for cdir, sup, rto in CONFIGS:
        c = cfg(sup, rto)
        tag = f"{sup}_{rto}"
        for col, key in (("rxD", "rxD"), ("rxB", "rxB"), ("sxD", "sxD"), ("sxB", "sxB")):
            t2[f"{tag}_{ {'rxD':'realxD','rxB':'realxB','sxD':'spxD','sxB':'spxB'}[col] }"] = \
                [round(c.loc[s, key], 4) if s in c.index else None for s in SCEN]
    t2.to_csv(TBL / "T2_final_operating_points.csv", index=False)
    print("\n### T2_final_operating_points (realized xD/xB per config)\n")
    # compact view: realized xD only, all configs
    comp = pd.DataFrame({"scenario": SCEN})
    for cdir, sup, rto in CONFIGS:
        c = cfg(sup, rto)
        comp[f"{sup}_{rto}"] = [round(c.loc[s, "rxD"], 4) if s in c.index else None for s in SCEN]
    print("realized xD by config:"); print(_md(comp))

    # ---- T3: supervisor comparison (MA only) ----
    t3 = pd.DataFrame({"scenario": SCEN})
    for sup in ("agentic", "baseline", "rule_naive", "rule_smart"):
        c = cfg(sup, "ma")
        t3[sup] = [f"act={int(c.loc[s,'n_actions'])}, xD={round(c.loc[s,'rxD'],4)}"
                   if s in c.index else "MISSING" for s in SCEN]
    t3["notes"] = [tests[s] for s in SCEN]
    t3.to_csv(TBL / "T3_supervisor_comparison_ma.csv", index=False)
    print("\n### T3_supervisor_comparison_ma\n"); print(_md(t3))

    # ---- T4: RTO command count (overhead) ----
    t4 = pd.DataFrame({"scenario": SCEN})
    for cdir, sup, rto in CONFIGS:
        c = cfg(sup, rto)
        t4[f"{sup}_{rto}"] = [int(c.loc[s, "n_rto"]) if s in c.index else None for s in SCEN]
    t4.to_csv(TBL / "T4_rto_commands_overhead.csv", index=False)
    print("\n### T4_rto_commands_overhead\n"); print(_md(t4))

    # ---- T5: convergence status (final RTO) ----
    t5 = pd.DataFrame({"scenario": SCEN})
    for cdir, sup, rto in CONFIGS:
        c = cfg(sup, rto)
        t5[f"{sup}_{rto}"] = [c.loc[s, "final_converged"] if s in c.index else None for s in SCEN]
    t5.to_csv(TBL / "T5_convergence_status.csv", index=False)
    print("\n### T5_convergence_status (final RTO converged?)\n"); print(_md(t5))

    # ---- digest for interpretation ----
    print("\n\n===== DIGEST =====")
    print("missing runs:", df[~df.present][["supervisor", "rto", "scenario"]].values.tolist() or "none")
    print("\n-- agentic vs baseline (MA), realized xD / xB --")
    am, bm = cfg("agentic", "ma"), cfg("baseline", "ma")
    for s in SCEN:
        if s in am.index and s in bm.index:
            print(f"  {s}: agentic xD={am.loc[s,'rxD']:.4f} xB={am.loc[s,'rxB']:.5f} (act {int(am.loc[s,'n_actions'])}) | "
                  f"baseline xD={bm.loc[s,'rxD']:.4f} xB={bm.loc[s,'rxB']:.5f} | dxD={am.loc[s,'rxD']-bm.loc[s,'rxD']:+.4f}")
    print("\n-- agent actions by scenario (ma / ma-gp) --")
    agp = cfg("agentic", "ma-gp")
    for s in SCEN:
        ma_a = int(am.loc[s, "n_actions"]) if s in am.index else None
        gp_a = int(agp.loc[s, "n_actions"]) if s in agp.index else None
        print(f"  {s}: ma_actions={ma_a} (trig {int(am.loc[s,'n_trigger'])}/upd {int(am.loc[s,'n_update'])})  "
              f"ma-gp_actions={gp_a}")
    print("\n-- MA vs MA-GP final realized xD (per supervisor) --")
    for sup in ("agentic", "baseline", "rule_naive", "rule_smart"):
        cm, cg = cfg(sup, "ma"), cfg(sup, "ma-gp")
        diffs = [f"{s}:{cm.loc[s,'rxD']-cg.loc[s,'rxD']:+.4f}" for s in SCEN
                 if s in cm.index and s in cg.index]
        print(f"  {sup}: dxD(ma-magp) " + " ".join(diffs))
    print("\n-- R5 convergence (all configs) --")
    r5 = present[present.scenario == "R5"]
    for _, r in r5.iterrows():
        print(f"  {r.supervisor}_{r.rto}: final_converged={r.final_converged} any_infeasible={r.any_infeasible} "
              f"realized xD={r.rxD:.4f} xB={r.rxB:.5f}")
    print("\n-- rule_naive vs rule_smart (MA) actions+xD --")
    rn, rs = cfg("rule_naive", "ma"), cfg("rule_smart", "ma")
    for s in SCEN:
        print(f"  {s}: naive act={int(rn.loc[s,'n_actions'])} xD={rn.loc[s,'rxD']:.4f} | "
              f"smart act={int(rs.loc[s,'n_actions'])} xD={rs.loc[s,'rxD']:.4f}")


if __name__ == "__main__":
    main()
