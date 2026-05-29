"""Tests for LLM/rules setpoint resolution and apply wiring."""
from __future__ import annotations

from agentic_mpc.mcp_sandbox import MPCSandbox
from agentic_mpc.scenario2_agent import Decision
from agentic_mpc.supervisory_apply import apply_supervisory_setpoint, reconcile, resolve_setpoint_targets


def _propose_dec(xD=0.95, xB=0.011):
    return Decision(
        "REAL_DISTURBANCE",
        "PROPOSE_SETPOINT",
        "rules rationale",
        proposed_targets={"xD": xD, "xB": xB},
    )


def test_reconcile_rules_win_on_action_mismatch():
    rec = reconcile("PROPOSE_SETPOINT", "VETO_HOLD")
    assert rec["final_action"] == "VETO_HOLD" and rec["overridden"]


def test_llm_targets_used_when_match_and_in_envelope():
    dec = _propose_dec(xD=0.940, xB=0.006)
    llm = {"action": "PROPOSE_SETPOINT", "xD_sp": 0.955, "xB_sp": 0.011, "rationale": "llm"}
    resolved = resolve_setpoint_targets("PROPOSE_SETPOINT", dec, llm)
    assert resolved["source"] == "llm"
    assert resolved["xD"] == 0.955 and resolved["xB"] == 0.011


def test_rules_targets_when_llm_outside_envelope():
    dec = _propose_dec()
    llm = {"action": "PROPOSE_SETPOINT", "xD_sp": 1.2, "xB_sp": 0.011, "rationale": "llm bad xD"}
    resolved = resolve_setpoint_targets("PROPOSE_SETPOINT", dec, llm)
    assert resolved["source"] == "rules"
    assert resolved["xD"] == 0.95


def test_rules_targets_when_action_overridden():
    dec = _propose_dec()
    llm = {"action": "PROPOSE_SETPOINT", "xD_sp": 0.955, "xB_sp": 0.011}
    resolved = resolve_setpoint_targets("ESCALATE", dec, llm)
    assert resolved is None


def test_apply_supervisory_setpoint_calls_sandbox():
    sb = MPCSandbox(seed=0)
    sb.advance(30)
    dec = _propose_dec(xD=0.94, xB=0.004)
    out = apply_supervisory_setpoint(sb, "PROPOSE_SETPOINT", dec, None)
    assert out is not None
    assert out["applied_targets"]["xD"] == 0.94
    assert out["target_source"] == "rules"
    assert sb.mpc.targets["xD"] == 0.94
