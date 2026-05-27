"""Pydantic schemas + repair logic for validating LLM tool-call arguments.

Carried over from the original ``test.py`` skeleton, including the rationale-dict repair
for a known small-model failure mode (returning the schema definition object instead of a
plain string for ``rationale``).
"""
from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field, ValidationError


class UpdateMpcTargetArgs(BaseModel):
    """Arguments for the ``update_mpc_target`` tool."""

    targets: Dict[str, float] = Field(..., description="CV name -> target value, "
                                      "e.g. {'xD': 0.96, 'xB': 0.005}")
    rationale: str = Field(..., min_length=10,
                           description="Operator-readable explanation (plain string)")


class TriggerRtoRunArgs(BaseModel):
    """Arguments for the ``trigger_rto_run`` tool (Phase 1.5)."""

    rationale: str = Field(..., min_length=10,
                           description="Operator-readable reason for re-optimizing (plain string)")


# Tools that take structured arguments register their schema here.
TOOL_ARG_SCHEMAS: dict[str, type[BaseModel]] = {
    "update_mpc_target": UpdateMpcTargetArgs,
    "trigger_rto_run": TriggerRtoRunArgs,
}


def validate_and_repair_args(tool_name: str, raw_args: dict):
    """Return ``(validated_args, error_message)``.

    If ``error_message`` is not ``None`` the agent should be told the call was rejected
    and asked to retry with corrected arguments. Tools with no registered schema pass
    through unchanged.
    """
    schema = TOOL_ARG_SCHEMAS.get(tool_name)
    if schema is None:
        return raw_args, None

    # --- repair pass: known small-model failure modes ---
    # Small models sometimes return `rationale` as the schema definition itself,
    # e.g. {"description": "...", "type": "string"} instead of a plain string. This
    # affects any tool that takes a rationale (update_mpc_target, trigger_rto_run).
    r = raw_args.get("rationale")
    if isinstance(r, dict) and "description" in r:
        raw_args = {**raw_args, "rationale": r["description"]}

    # --- validate against the schema ---
    try:
        validated = schema(**raw_args)
        return validated.model_dump(), None
    except ValidationError as e:
        return None, f"Tool call rejected: arguments did not match schema. Details: {e}"
