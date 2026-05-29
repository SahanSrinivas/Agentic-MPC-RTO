"""Apply supervisory decisions to the MCP sandbox (set_target wiring).

Guardrail pattern:
  final_action = canonical (rules) action; if the LLM disagrees on *action*, rules win.
  On PROPOSE_SETPOINT with action agreement, use LLM (xD_sp, xB_sp) only when both are present
  and lie inside the safety envelope without clipping; otherwise use rules' proposed_targets.
"""
from __future__ import annotations

from typing import Any

from agentic_mpc.safety import BoxSafetyEnvelope
from agentic_mpc.scenario2_agent import Decision


def reconcile(llm_action: str, canonical_action: str) -> dict[str, Any]:
    """Rules always win on action class; LLM only proposes."""
    return {
        "final_action": canonical_action,
        "match": llm_action == canonical_action,
        "overridden": llm_action != canonical_action,
    }


def resolve_setpoint_targets(
    final_action: str,
    canonical: Decision,
    llm_proposal: dict[str, Any] | None = None,
    *,
    envelope: BoxSafetyEnvelope | None = None,
) -> dict[str, Any] | None:
    """Pick (xD, xB) to apply when final_action is PROPOSE_SETPOINT, or None otherwise."""
    if final_action != "PROPOSE_SETPOINT":
        return None
    rules_tgt = canonical.proposed_targets
    if rules_tgt is None:
        return None

    env = envelope or BoxSafetyEnvelope()
    llm = llm_proposal or {}
    llm_action = str(llm.get("action", "")).strip().upper()
    if llm_action == "PROPOSE_SETPOINT" and final_action == "PROPOSE_SETPOINT":
        xD_sp, xB_sp = llm.get("xD_sp"), llm.get("xB_sp")
        if xD_sp is not None and xB_sp is not None:
            proposed = {"targets": {"xD": float(xD_sp), "xB": float(xB_sp)}}
            safe, clipped = env.project(proposed)
            if not clipped:
                return {
                    "xD": safe["targets"]["xD"],
                    "xB": safe["targets"]["xB"],
                    "source": "llm",
                    "rationale": str(llm.get("rationale") or canonical.rationale),
                }

    return {
        "xD": float(rules_tgt["xD"]),
        "xB": float(rules_tgt["xB"]),
        "source": "rules",
        "rationale": canonical.rationale,
    }


def apply_supervisory_setpoint(
    sandbox,
    final_action: str,
    canonical: Decision,
    llm_proposal: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Call sandbox.set_target when the resolved supervisory action warrants it."""
    resolved = resolve_setpoint_targets(final_action, canonical, llm_proposal)
    if resolved is None:
        return None
    out = sandbox.set_target(
        xD=resolved["xD"],
        xB=resolved["xB"],
        rationale=resolved["rationale"],
    )
    return {
        **out,
        "target_source": resolved["source"],
        "requested_targets": {"xD": resolved["xD"], "xB": resolved["xB"]},
    }
