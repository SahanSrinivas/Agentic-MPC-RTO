"""S1 no-LLM CALIBRATION dry-run (free; verifies the four LLM-independent structural invariants
before the LLM arm is built or the metric is run on results).

Invariants to verify (and freeze magnitudes against):
  1. measured xD stays < 1.0 throughout Event B  (so a [0,1] check cannot catch it).
  2. innovation at the A-cycle (t=120) and B-cycle (t=180) both clear the naive 5e-4 threshold and
     are comparable (a single-threshold rule fires on both).
  3. trigger-on-A REDUCES regret (real load, re-optimization helps) AND trigger-on-B INCREASES
     regret (sensor fault, re-optimizing corrupts MA) -- regret on the TRUE state.
  4. the guarded gain-residual rho separates A (coupled: both rho large) from B (single-channel:
     rho_xD large, rho_xB ~0).

Scripted (deterministic, no LLM) variants on the SAME seed:
  never  : never trigger (baseline)
  trigA  : trigger only at the A-cycle (t=120)
  trigAB : trigger at A-cycle and B-cycle (the naive path: fires on both)
  trigB  : trigger only at the B-cycle (t=180)

Run:  python experiments/s1_calibration_dryrun.py
"""
from __future__ import annotations

import warnings

import numpy as np

from agentic_mpc.agent.tools import AgentContext, make_tool_registry
from agentic_mpc.controllers import ClassicalMPC
from agentic_mpc.plants import WoodBerryPlant
from agentic_mpc.rto import ModifierAdaptation, RTOMPCLoop, WoodBerryEconomics, WoodBerryRTO
from agentic_mpc.safety import BoxSafetyEnvelope
from agentic_mpc.scenarios import S1ConflictingSignals

warnings.filterwarnings("ignore")

K = np.array([[12.8, -18.9], [6.6, -19.4]])      # nominal Wood-Berry SS gains [out, in]
CYCLES = {60, 120, 180, 240}
EVENT_B_T = 160.0
DT = 1.0


def p_opt() -> float:
    econ = WoodBerryEconomics()
    pl = WoodBerryPlant(dt=DT, seed=0)
    return float(WoodBerryRTO(economics=econ, plant_params=pl.params, seed=0).solve()["objective"])


def rho_from_history(hist: dict) -> tuple[float, float]:
    """Guarded-rule gain residual: denoised recent change minus the input-explained part."""
    xD = np.asarray(hist["y"]["xD"], float); xB = np.asarray(hist["y"]["xB"], float)
    R = np.asarray(hist["u"]["R"], float); S = np.asarray(hist["u"]["S"], float)
    if len(xD) < 10:
        return 0.0, 0.0
    f = lambda a: float(a[-5:].mean() - a[:5].mean())          # denoised window change
    dxD, dxB, dR, dS = f(xD), f(xB), f(R), f(S)
    dxD_pred = K[0, 0] * dR + K[0, 1] * dS
    dxB_pred = K[1, 0] * dR + K[1, 1] * dS
    return dxD - dxD_pred, dxB - dxB_pred


def build():
    """Fresh plant+mpc+MA, MA warmed to convergence at the nominal plant, plant reset clean."""
    seed = 1
    plant = WoodBerryPlant(dt=DT, seed=seed)
    mpc = ClassicalMPC(dt=DT)
    rto = ModifierAdaptation(economics=WoodBerryEconomics(), plant=plant,
                             plant_params=plant.params, seed=seed)
    rto.run_until_convergence(max_iterations=30)            # warm MA past its probe phase (nominal)
    plant.reset()                                           # restore clean seed/state for the run
    loop = RTOMPCLoop(plant, mpc, rto, rto_interval_min=1e9, dt=DT,
                      trigger_rto_at_start=False)           # supervisor is the SOLE RTO trigger
    ctx = AgentContext(plant=plant, controller=mpc, safety=BoxSafetyEnvelope(), rto=rto, rto_loop=loop)
    return plant, mpc, rto, loop, make_tool_registry(ctx)


def run_variant(trigger_at: set[int]) -> dict:
    plant, mpc, rto, loop, reg = build()
    scen = S1ConflictingSignals()
    log = {"innov": {}, "rho": {}, "triggers": [], "rto_status": {}}

    def on_step(lp, t, y):
        scen.on_step(lp, t, y)
        ti = int(round(t))
        if ti in CYCLES:
            h = reg["get_mpc_health"]()
            im = h.get("innovation_mean", {})
            st = reg["get_process_state"]()
            rxD, rxB = rho_from_history(st["history"])
            log["innov"][ti] = (im.get("xD"), im.get("xB"))
            log["rho"][ti] = (rxD, rxB)
            if ti in trigger_at:
                res = reg["trigger_rto_run"](rationale=f"scripted trigger at t={ti}")
                log["triggers"].append((ti, res.get("status"), res.get("commanded_setpoints")))

    res = loop.run(240.0, on_step=on_step)
    h = res["history"]
    log["history"] = h
    return log


def regret_true(hist, P: float) -> float:
    econ = WoodBerryEconomics()
    xD = np.asarray(hist["xD_true"], float); xB = np.asarray(hist["xB_true"], float)
    inst = np.array([max(P - econ.profit(float(a), float(b)), 0.0) for a, b in zip(xD, xB)])
    return float(inst.sum() * DT)


def main() -> None:
    P = p_opt()
    print(f"P_opt = {P:.6f}  (S1: supervisor-sole-trigger; periodic RTO OFF; NOT comparable to T6/T7)\n")
    variants = {"never": set(), "trigA": {120}, "trigAB": {120, 180}, "trigB": {180}}
    R = {name: run_variant(at) for name, at in variants.items()}

    # ---- invariant 1: measured xD < 1.0 during Event B (t >= 160) ----
    print("== Invariant 1: measured xD < 1.0 throughout Event B (t>=160) ==")
    for name, lg in R.items():
        h = lg["history"]; t = np.asarray(h["t"]); xD = np.asarray(h["xD"])
        mx = float(xD[t >= EVENT_B_T].max())
        print(f"  {name:7} max measured xD in B-window = {mx:.4f}  -> {'OK (<1)' if mx < 1.0 else 'FAIL (>=1)'}")

    # ---- invariant 2: innovation at A-cycle and B-cycle clears 5e-4, comparable ----
    print("\n== Invariant 2: innovation fires naive 5e-4 at A-cycle(120) and B-cycle(180) ==")
    for name in ("never", "trigA"):
        for c in (120, 180):
            ix, ib = R[name]["innov"][c]
            fires = abs(ix) > 5e-4 or abs(ib) > 5e-4
            print(f"  {name:7} t={c}: innov_xD={ix:+.3e} innov_xB={ib:+.3e}  "
                  f"naive-fires={fires}")

    # ---- invariant 3: trigger-on-A reduces regret; trigger-on-B (on top of A) raises it ----
    print("\n== Invariant 3: regret on TRUE state (lower=better) ==")
    g = {name: regret_true(R[name]["history"], P) for name in R}
    for name in ("never", "trigA", "trigAB", "trigB"):
        print(f"  {name:7} regret={g[name]:.4f}  triggers={[(t,s) for t,s,_ in R[name]['triggers']]}")
    print(f"  A-benefit:  never({g['never']:.4f}) -> trigA({g['trigA']:.4f})  "
          f"delta={g['trigA']-g['never']:+.4f}  -> {'OK (A helps)' if g['trigA'] < g['never'] else 'FAIL'}")
    print(f"  B-harm:     trigA({g['trigA']:.4f}) -> trigAB({g['trigAB']:.4f}) "
          f"delta={g['trigAB']-g['trigA']:+.4f}  -> {'OK (B hurts)' if g['trigAB'] > g['trigA'] else 'FAIL'}")

    # ---- invariant 4: rho separates A (coupled) from B (single-channel) ----
    print("\n== Invariant 4: guarded rho-residual (trigA path: A handled at 120, B judged at 180) ==")
    for c, label in ((120, "A-cycle"), (180, "B-cycle")):
        rxD, rxB = R["trigA"]["rho"][c]
        print(f"  t={c} ({label}): rho_xD={rxD:+.4f} rho_xB={rxB:+.4f}  "
              f"|rho_xD|/|rho_xB|={abs(rxD)/max(abs(rxB),1e-9):.1f}")
    print("  expect A-cycle: BOTH large (coupled);  B-cycle: rho_xD large, rho_xB ~0 (single-channel)")


if __name__ == "__main__":
    main()
