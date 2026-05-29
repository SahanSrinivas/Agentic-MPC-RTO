"""Tests for the three-arm comparison guardrail: the deterministic validator (rules) always wins,
and the canonical decisions per case are correct -- both WITHOUT calling the LLM."""
from __future__ import annotations

import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.three_arm_comparison import INJECT, reconcile, snapshot_at_decision  # noqa: E402
from agentic_mpc.scenario2_agent import DiagnosticSupervisor  # noqa: E402

_EXPECTED = {"nominal": "HOLD", "sensor_fault": "VETO_HOLD",
             "coupled_load": "PROPOSE_SETPOINT", "ambiguous_load": "ESCALATE"}


def test_validator_always_wins_even_if_llm_disagrees():
    # a rogue LLM proposing to chase a sensor fault is overridden; the canonical VETO stands.
    rec = reconcile("PROPOSE_SETPOINT", "VETO_HOLD")
    assert rec["final_action"] == "VETO_HOLD" and rec["overridden"] and not rec["match"]
    # a rogue LLM proposing to act on an ambiguous single-channel mismatch is overridden -> ESCALATE.
    rec = reconcile("PROPOSE_SETPOINT", "ESCALATE")
    assert rec["final_action"] == "ESCALATE" and rec["overridden"]
    # agreement is not an override.
    rec = reconcile("HOLD", "HOLD")
    assert rec["final_action"] == "HOLD" and rec["match"] and not rec["overridden"]


def test_canonical_decisions_are_correct_without_any_llm():
    for case, expected in _EXPECTED.items():
        _, diag, snap = snapshot_at_decision(case)
        assert DiagnosticSupervisor().assess(diag, snap).action == expected, case
