"""Shared runner for the Phase-1.5 scenario scripts (R1-R7).

Builds the WoodBerry plant + classical MPC + an RTO variant (nominal / MA / MA-GP) inside an
:class:`RTOMPCLoop`, optionally with the LLM supervisory agent on top, applies one scenario's
perturbation through the loop's ``on_step`` hook, and writes a plot + JSON log to a
model-name-suffixed output directory. All randomness is seeded.

The 7 ``phase1_5_r*.py`` scripts are thin wrappers that call :func:`run_scenario` with their
scenario id and an ``argparse`` ``--model`` (plus optional ``--rto`` / ``--agentic`` / ``--t-end``).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agentic_mpc.agent import SYSTEM_PROMPT_RTO, SupervisoryAgent
from agentic_mpc.agent.llm_config import LLMConfig
from agentic_mpc.controllers import ClassicalMPC
from agentic_mpc.plants import WoodBerryPlant
from agentic_mpc.rto import (MAGaussianProcess, ModifierAdaptation, RTOMPCLoop, WoodBerryEconomics,
                             WoodBerryRTO)
from agentic_mpc.safety import BoxSafetyEnvelope
from agentic_mpc.scenarios import SCENARIOS

warnings.filterwarnings("ignore")
_OUT_ROOT = pathlib.Path(__file__).parent / "outputs" / "phase1_5"


def _make_rto(variant: str, plant: WoodBerryPlant, econ: WoodBerryEconomics, seed: int):
    variant = variant.lower()
    if variant == "nominal":
        return WoodBerryRTO(economics=econ, plant_params=plant.params, seed=seed)
    if variant == "ma":
        return ModifierAdaptation(economics=econ, plant=plant, plant_params=plant.params, seed=seed)
    if variant in ("ma-gp", "magp"):
        return MAGaussianProcess(economics=econ, plant=plant, plant_params=plant.params, seed=seed)
    raise ValueError(f"unknown rto variant {variant!r} (use nominal | ma | ma-gp)")


def run_scenario(scenario_id: str, model: str = "qwen3:4b", rto_variant: str = "ma",
                 agentic: bool = True, t_end: float = 240.0, rto_interval_min: float = 60.0,
                 agent_interval_min: float = 60.0, seed: int = 0) -> dict:
    """Run one Phase-1.5 scenario and persist outputs under outputs/phase1_5/<model>/<scenario>/."""
    scenario = SCENARIOS[scenario_id]()
    dt = 1.0
    plant = WoodBerryPlant(dt=dt, seed=seed)
    mpc = ClassicalMPC(dt=dt)
    rto = _make_rto(rto_variant, plant, WoodBerryEconomics(), seed)
    loop = RTOMPCLoop(plant, mpc, rto, rto_interval_min=rto_interval_min, dt=dt)

    agent = None
    agent_log: list[dict] = []
    if agentic:
        agent = SupervisoryAgent(plant, mpc, safety=BoxSafetyEnvelope(), rto=rto, rto_loop=loop,
                                 config=LLMConfig(model=model), system_prompt=SYSTEM_PROMPT_RTO)
    agent_steps = max(1, int(round(agent_interval_min / dt)))

    def on_step(lp, t, y):
        scenario.on_step(lp, t, y)
        if agent is not None and int(round(t)) > 0 and int(round(t)) % agent_steps == 0:
            msg = (f"Supervisory cycle at t={t:.0f} min on the Wood-Berry RTO/MPC stack. Assess "
                   f"the process, MPC health, economic context, and RTO status; act if warranted.")
            try:
                out = agent.run_cycle(msg, max_iterations=6)
            except Exception as e:  # never let one flaky LLM cycle kill the run
                out = {"final": f"[agent error: {type(e).__name__}: {e}]", "actions": [], "iterations": 0}
            agent_log.append({"t": t, "final": out["final"], "iterations": out.get("iterations"),
                              "actions": out["actions"]})

    result = loop.run(t_end, on_step=on_step)
    out_dir = _OUT_ROOT / model.replace(":", "_") / scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot(result, scenario, out_dir, scenario_id)
    summary = _summary(result, scenario, agent_log, model, rto_variant, agentic)
    (out_dir / "log.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    _print_summary(summary, out_dir)
    return summary


def _summary(result, scenario, agent_log, model, rto_variant, agentic) -> dict:
    h = result["history"]
    return {
        "scenario": scenario.describe(),
        "config": {"model": model, "rto_variant": rto_variant, "agentic": agentic},
        "rto_handoffs": result["handoffs"],
        "agent_decisions": agent_log,
        "final_realized": {"xD": float(h["xD"][-1]), "xB": float(h["xB"][-1])},
        "final_setpoint": {"xD": float(h["xD_sp"][-1]), "xB": float(h["xB_sp"][-1])},
        "n_rto_commands": len(result["handoffs"]),
        "n_agent_decisions": len(agent_log),
        "n_agent_actions": sum(1 for d in agent_log for a in d["actions"]
                               if a.get("tool") in ("trigger_rto_run", "update_mpc_target")
                               and a.get("status") == "executed"),
    }


def _plot(result, scenario, out_dir, scenario_id) -> None:
    h = result["history"]; t = h["t"]
    rto_times = [hd["t"] for hd in result["handoffs"]]
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t, h["xD_sp"], "k--", lw=0.8, label="xD setpoint (RTO)")
    ax[0].plot(t, h["xD"], color="C2", lw=1.0, label="xD measured")
    ax[0].set_ylabel("xD"); ax[0].legend(loc="best", fontsize=8)
    ax[0].set_title(f"Phase 1.5 {scenario_id} -- {scenario.REGIME}")
    ax[1].plot(t, h["xB_sp"], "k--", lw=0.8, label="xB setpoint (RTO)")
    ax[1].plot(t, h["xB"], color="C3", lw=1.0, label="xB measured")
    ax[1].set_ylabel("xB"); ax[1].legend(loc="best", fontsize=8)
    ax[2].plot(t, h["R"], color="C0", lw=0.9, label="R")
    ax[2].plot(t, h["S"], color="C1", lw=0.9, label="S")
    ax[2].set_ylabel("inputs [lb/min]"); ax[2].set_xlabel("time [min]"); ax[2].legend(loc="best", fontsize=8)
    for a in ax:
        for i, rt in enumerate(rto_times):
            a.axvline(rt, color="purple", ls=":", lw=0.6, label="RTO command" if i == 0 else None)
    fig.tight_layout(); fig.savefig(out_dir / f"{scenario_id}.png", dpi=110); plt.close(fig)


def _print_summary(s, out_dir) -> None:
    sc = s["scenario"]
    print("=" * 74)
    print(f"PHASE 1.5 {sc['scenario_id']} ({sc['regime']}) -- model={s['config']['model']} "
          f"rto={s['config']['rto_variant']} agentic={s['config']['agentic']}")
    print(f"  mechanism: {sc['mechanism']}")
    print(f"  final realized: xD={s['final_realized']['xD']:.4f} xB={s['final_realized']['xB']:.5f}"
          f"  (setpoint xD={s['final_setpoint']['xD']:.4f} xB={s['final_setpoint']['xB']:.5f})")
    print(f"  RTO commands: {s['n_rto_commands']}  | agent cycles: {s['n_agent_decisions']}  | "
          f"agent actions: {s['n_agent_actions']}")
    print(f"  outputs -> {out_dir}")
    print("=" * 74)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(scenario_id: str) -> None:
    p = argparse.ArgumentParser(description=f"Phase 1.5 scenario {scenario_id}")
    p.add_argument("--model", default="qwen3:4b", help="Ollama model id for the agent")
    p.add_argument("--rto", default="ma", choices=["nominal", "ma", "ma-gp", "magp"],
                   help="RTO variant")
    p.add_argument("--no-agent", action="store_true", help="run the baseline (no LLM agent)")
    p.add_argument("--t-end", type=float, default=240.0, help="run length [min]")
    p.add_argument("--rto-interval", type=float, default=60.0, help="RTO cadence [min]")
    p.add_argument("--agent-interval", type=float, default=60.0, help="agent cadence [min]")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run_scenario(scenario_id, model=a.model, rto_variant=a.rto, agentic=not a.no_agent,
                 t_end=a.t_end, rto_interval_min=a.rto_interval, agent_interval_min=a.agent_interval,
                 seed=a.seed)
