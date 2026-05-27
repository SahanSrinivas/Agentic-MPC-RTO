"""Tests for the refactored agent wiring (Step 6) -- no LLM required.

Covers the argument validation/repair, the tools acting on REAL Plant + Controller
objects through the interfaces, and the safety-envelope projection on the agent's one
mutating action.
"""
from __future__ import annotations

import numpy as np

from agentic_mpc.agent.tools import (AgentContext, get_mpc_health, get_process_state,
                                     make_tool_registry, update_mpc_target)
from agentic_mpc.agent.validation import validate_and_repair_args
from agentic_mpc.controllers import ClassicalMPC
from agentic_mpc.plants import WoodBerryPlant
from agentic_mpc.safety import BoxSafetyEnvelope


def _ctx(with_safety: bool = True) -> AgentContext:
    plant = WoodBerryPlant(dt=1.0, seed=0)
    ctrl = ClassicalMPC(dt=1.0)
    # warm the controller so health/state are populated
    ctrl.compute_control(np.array([0.96, 0.005]), np.array([0.96, 0.005]), t=0.0)
    return AgentContext(plant=plant, controller=ctrl,
                        safety=BoxSafetyEnvelope() if with_safety else None)


# --- validation / repair ---------------------------------------------------------------
def test_repair_rationale_dict():
    raw = {"targets": {"xD": 0.95}, "rationale": {"description": "lower xD purity",
                                                  "type": "string"}}
    clean, err = validate_and_repair_args("update_mpc_target", raw)
    assert err is None
    assert clean["rationale"] == "lower xD purity"
    assert clean["targets"] == {"xD": 0.95}


def test_validation_rejects_short_rationale():
    raw = {"targets": {"xD": 0.95}, "rationale": "no"}  # < min_length
    clean, err = validate_and_repair_args("update_mpc_target", raw)
    assert clean is None and err is not None


def test_unknown_tool_passes_through():
    args, err = validate_and_repair_args("get_process_state", {})
    assert err is None and args == {}


# --- tools on real objects -------------------------------------------------------------
def test_get_process_state_shape():
    ctx = _ctx()
    st = get_process_state(ctx)
    assert {"t", "y", "u", "history"} <= set(st)
    assert set(st["y"]) == {"xD", "xB"}
    assert set(st["u"]) == {"R", "S"}


def test_get_mpc_health_shape():
    ctx = _ctx()
    h = get_mpc_health(ctx)
    assert {"innovation_mean", "innovation_std", "active_constraints", "ise_recent"} <= set(h)


def test_update_mpc_target_applies_to_controller():
    ctx = _ctx(with_safety=False)
    res = update_mpc_target(ctx, {"xD": 0.95}, "lower overhead purity target for test")
    assert res["status"] == "ok"
    assert ctx.controller.targets["xD"] == 0.95
    assert res["clipped_by_safety"] is False


def test_update_mpc_target_rejects_unknown_cv():
    ctx = _ctx(with_safety=False)
    res = update_mpc_target(ctx, {"not_a_cv": 1.0}, "this should be rejected cleanly")
    assert res["status"] == "error"


def test_registry_dispatch():
    ctx = _ctx()
    reg = make_tool_registry(ctx)
    assert set(reg) == {"get_process_state", "get_mpc_health", "update_mpc_target"}
    assert "history" in reg["get_process_state"]()
    out = reg["update_mpc_target"](targets={"xD": 0.97}, rationale="nudge xD up for test")
    assert out["status"] == "ok" and ctx.controller.targets["xD"] == 0.97


# --- safety envelope -------------------------------------------------------------------
def test_safety_envelope_clips_out_of_range():
    env = BoxSafetyEnvelope()  # xD in [0.90, 0.99], xB in [0.001, 0.05]
    safe, violated = env.project({"targets": {"xD": 1.5, "xB": 0.005}})
    assert violated is True
    assert safe["targets"]["xD"] == 0.99
    assert safe["targets"]["xB"] == 0.005


def test_safety_envelope_passes_in_range():
    env = BoxSafetyEnvelope()
    safe, violated = env.project({"targets": {"xD": 0.95, "xB": 0.004}})
    assert violated is False
    assert safe["targets"] == {"xD": 0.95, "xB": 0.004}


def test_update_mpc_target_clips_via_safety():
    ctx = _ctx(with_safety=True)
    res = update_mpc_target(ctx, {"xD": 2.0}, "agent proposed an out-of-spec target")
    assert res["clipped_by_safety"] is True
    assert ctx.controller.targets["xD"] == 0.99  # clipped to the safe upper bound
