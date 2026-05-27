"""Tests for the rule-based supervisor baselines (Naive + Smart), seed propagation, and the
config-aware output-directory separation. All LLM-free and deterministic.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

from agentic_mpc.agent import RuleBasedSupervisorNaive, RuleBasedSupervisorSmart
from agentic_mpc.controllers import ClassicalMPC
from agentic_mpc.plants import WoodBerryPlant
from agentic_mpc.rto import RTOMPCLoop, WoodBerryRTO

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
from _phase1_5_runner import output_dir_for  # noqa: E402


def _settled_nominal_stack(noise: float = 0.0, run_steps: int = 40, **disturbance):
    """Build a WoodBerry+MPC+nominal-RTO loop, optionally with a disturbance, and run it."""
    plant = WoodBerryPlant(dt=1.0, seed=42, meas_noise_std=noise)
    mpc = ClassicalMPC(dt=1.0)
    rto = WoodBerryRTO(plant_params=plant.params)            # nominal RTO: does not adapt
    loop = RTOMPCLoop(plant, mpc, rto, rto_interval_min=30.0)
    if disturbance:
        plant.set_disturbance(**disturbance)
    loop.run(float(run_steps))
    return plant, mpc, rto, loop


def _action_tools(out: dict) -> list[str]:
    return [a["tool"] for a in out["actions"]
            if a["tool"] in ("trigger_rto_run", "update_mpc_target") and a["status"] == "executed"]


# --- nominal: no action ----------------------------------------------------------------
def test_both_variants_hold_when_nominal():
    plant, mpc, rto, loop = _settled_nominal_stack()
    for cls in (RuleBasedSupervisorNaive, RuleBasedSupervisorSmart):
        sup = cls(plant, mpc, rto=rto, rto_loop=loop)
        out = sup.run_cycle("nominal cycle")
        assert _action_tools(out) == [], (cls.__name__, out["final"])
        assert out["final"].startswith("No action")


# --- Rule 1: biased innovation -> trigger RTO (both variants) ---------------------------
def test_biased_innovation_triggers_rto():
    # load disturbance biases the MPC innovation negative (~-1.5e-3); nominal RTO does not adapt,
    # so the bias persists and Rule 1 must fire.
    plant, mpc, rto, loop = _settled_nominal_stack(run_steps=30, output_bias={"xD": -0.03})
    health = mpc.get_health()
    assert abs(health["innovation_mean"]["xD"]) > 5e-4, health  # precondition: biased
    for cls in (RuleBasedSupervisorNaive, RuleBasedSupervisorSmart):
        sup = cls(plant, mpc, rto=rto, rto_loop=loop)
        out = sup.run_cycle("post-disturbance cycle")
        assert "trigger_rto_run" in _action_tools(out), (cls.__name__, out["final"])


# --- Rule 2: sustained offset -> variant-specific action -------------------------------
def _force_sustained_offset(loop):
    """Craft a settled, low-innovation state with the RTO-commanded setpoint far from the plant."""
    loop.last_handoff = {"t": 0.0, "setpoints": {"xD": 0.97, "xB": 0.005}, "rto_type": "nominal",
                         "rto_status": {}}
    loop._settled_since_command = True


def test_rule2_naive_retargets_to_measurement():
    plant, mpc, rto, loop = _settled_nominal_stack()      # y ~ 0.96, innovation ~ 0
    _force_sustained_offset(loop)                          # commanded 0.97 -> offset 0.01 > 5e-3
    sup = RuleBasedSupervisorNaive(plant, mpc, rto=rto, rto_loop=loop)
    out = sup.run_cycle("offset cycle")
    assert _action_tools(out) == ["update_mpc_target"], out["final"]
    # retargeted to the current measurement (~0.96), i.e. accepted the drift
    assert abs(mpc.targets["xD"] - 0.96) < 0.01


def test_rule2_smart_triggers_rto():
    plant, mpc, rto, loop = _settled_nominal_stack()
    _force_sustained_offset(loop)
    sup = RuleBasedSupervisorSmart(plant, mpc, rto=rto, rto_loop=loop)
    out = sup.run_cycle("offset cycle")
    assert _action_tools(out) == ["trigger_rto_run"], out["final"]


# --- log format parity with the LLM supervisor ----------------------------------------
def test_run_cycle_log_format_matches_llm():
    plant, mpc, rto, loop = _settled_nominal_stack()
    out = RuleBasedSupervisorNaive(plant, mpc, rto=rto, rto_loop=loop).run_cycle("x")
    assert set(out) == {"final", "actions", "iterations"}
    assert isinstance(out["final"], str) and out["iterations"] == 1
    for entry in out["actions"]:                           # same shape as supervisor._execute_tool_call
        assert {"iteration", "tool", "status", "args", "result"} <= set(entry)
        assert entry["status"] == "executed"


# --- seed propagation ------------------------------------------------------------------
def test_seed_reproducibility_in_plant_noise():
    u = np.array([1.95, 1.71])
    a = WoodBerryPlant(dt=1.0, seed=42).step(u, 1.0)
    b = WoodBerryPlant(dt=1.0, seed=42).step(u, 1.0)
    c = WoodBerryPlant(dt=1.0, seed=43).step(u, 1.0)
    assert np.allclose(a, b) and not np.allclose(a, c)


# --- output-directory separation (the overwrite-bug fix) ------------------------------
def test_output_dir_separation():
    paths = {
        ("llm", "ma", True): output_dir_for("R3", "qwen3:30b", "ma", "llm", True),
        ("llm", "ma-gp", True): output_dir_for("R3", "qwen3:30b", "ma-gp", "llm", True),
        ("baseline", "ma", False): output_dir_for("R3", "qwen3:30b", "ma", "llm", False),
        ("naive", "ma", True): output_dir_for("R3", "x", "ma", "rule-based-naive", True),
        ("smart", "ma", True): output_dir_for("R3", "x", "ma", "rule-based-smart", True),
    }
    s = {str(p) for p in paths.values()}
    assert len(s) == 5, s                                  # all five configs are distinct
    assert paths[("llm", "ma", True)].as_posix().endswith("qwen3_30b/agentic_ma/R3")
    assert paths[("baseline", "ma", False)].as_posix().endswith("qwen3_30b/baseline_ma/R3")
    assert paths[("naive", "ma", True)].as_posix().endswith("rule_based_naive_ma/R3")
    assert paths[("smart", "ma", True)].as_posix().endswith("rule_based_smart_ma/R3")
