"""Tool schemas + registry for the supervisory agent.

The three Phase-1 tools wrap REAL :class:`~agentic_mpc.interfaces.Plant` and
:class:`~agentic_mpc.interfaces.Controller` objects (replacing the stubs in the original
``test.py``). Dependencies are injected via a small :class:`AgentContext`, and
:func:`make_tool_registry` binds that context into the callables the agent loop invokes.

The agent talks ONLY through the interface ABCs on the context -- never a concrete plant
or controller class -- so the same tool set works for any process/controller pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_mpc.interfaces import Controller, Optimizer, Plant, SafetyEnvelope


@dataclass
class AgentContext:
    """Injected dependencies the tools act on (all via the interface ABCs)."""

    plant: Plant
    controller: Controller
    safety: SafetyEnvelope | None = None   # last-line guard on update_mpc_target
    rto: Optimizer | None = None           # Phase 1.5 RTO layer (nominal / MA / MA-GP)
    rto_loop: Any = None                   # RTOMPCLoop, for the get_rto_status handoff snapshot


# OpenAI function-calling schemas (the "universal interface" the agent calls).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_process_state",
            "description": ("Returns the current measured process state: controlled "
                            "variables (xD, xB), manipulated variables (R, S), the "
                            "current time, and a recent history window."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mpc_health",
            "description": ("Returns MPC health metrics: innovation statistics (measured "
                            "minus the MPC's one-step-ahead prediction -- the model-plant "
                            "mismatch signal), active constraints, and recent ISE."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_mpc_target",
            "description": ("Push new setpoint targets to the MPC. Must include a "
                            "rationale. Targets are clipped to a safe range before being "
                            "applied."),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "description": ("Map of controlled-variable name to target value, "
                                        "e.g. {\"xD\": 0.95, \"xB\": 0.006}"),
                    },
                    "rationale": {
                        "type": "string",
                        "description": ("Operator-readable plain-string explanation of why "
                                        "this change is being made. Example: 'Reducing xD "
                                        "target due to reflux constraint pressure'. Do NOT "
                                        "pass an object here."),
                    },
                },
                "required": ["targets", "rationale"],
            },
        },
    },
]


# --- tool implementations (operate on the injected context) ---
def get_process_state(ctx: AgentContext) -> dict:
    return ctx.plant.get_state()


def get_mpc_health(ctx: AgentContext) -> dict:
    return ctx.controller.get_health()


def update_mpc_target(ctx: AgentContext, targets: dict, rationale: str) -> dict:
    """Project the proposed targets through the safety envelope, then apply to the MPC."""
    clipped_by_safety = False
    if ctx.safety is not None:
        safe_action, clipped_by_safety = ctx.safety.project({"targets": targets})
        targets = safe_action.get("targets", targets)
    try:
        ctx.controller.set_targets(targets, rationale)
    except KeyError as e:
        return {"status": "error", "message": str(e),
                "hint": "targets must be a subset of the controlled variables (xD, xB)."}
    return {"status": "ok", "applied_targets": targets,
            "clipped_by_safety": clipped_by_safety}


# --- Phase 1.5 RTO tool schemas (added only when the context carries an RTO) ---
RTO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "trigger_rto_run",
            "description": ("Run the real-time optimizer (RTO) to recompute the economically-"
                            "optimal composition setpoints and command them to the MPC. Use after "
                            "you detect an economic shift or sustained model-plant mismatch. Must "
                            "include a rationale."),
            "parameters": {
                "type": "object",
                "properties": {
                    "rationale": {"type": "string", "description": ("Operator-readable plain-"
                                  "string reason for re-optimizing, e.g. 'Steam price rose; "
                                  "re-optimize economics'. Do NOT pass an object.")},
                },
                "required": ["rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_economic_context",
            "description": ("Returns the current economic parameters (prices/costs, feed, spec), "
                            "the last RTO objective, and the model-plant economic gap. Use to spot "
                            "an economic shift (a price/cost change vs nominal)."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rto_status",
            "description": ("Returns the latest RTO handoff: which setpoints were commanded, when, "
                            "by which RTO variant (nominal/MA/MA-GP), the RTO status at command "
                            "time, and whether the plant has settled since."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def trigger_rto_run(ctx: AgentContext, rationale: str) -> dict:
    """Run the RTO, project the recommended setpoints through safety, command them to the MPC."""
    if ctx.rto is None:
        return {"status": "error", "message": "no RTO is configured in this context."}
    res = ctx.rto.solve()
    sp = res.get("setpoints", {})
    if not res.get("converged", True):
        return {"status": "rto_infeasible", "rto_variant": res.get("type"),
                "active_constraints": res.get("active_constraints", []),
                "message": "RTO returned infeasible; setpoints unchanged."}
    targets = {"xD": float(sp["xD"]), "xB": float(sp["xB"])}
    clipped = False
    if ctx.safety is not None:
        safe, clipped = ctx.safety.project({"targets": targets})
        targets = safe.get("targets", targets)
    ctx.controller.set_targets(targets, rationale)
    if ctx.rto_loop is not None:                # keep get_rto_status consistent with the command
        ctx.rto_loop.note_external_command(res)
    return {"status": "ok", "commanded_setpoints": targets, "rto_variant": res.get("type"),
            "objective": res.get("objective"), "active_constraints": res.get("active_constraints", []),
            "model_plant_gap": res.get("model_plant_gap"), "clipped_by_safety": clipped}


def get_economic_context(ctx: AgentContext) -> dict:
    if ctx.rto is None:
        return {"status": "error", "message": "no RTO is configured in this context."}
    st = ctx.rto.get_status()
    return {"prices": ctx.rto.economics.prices(),
            "last_rto_objective": st.get("objective"),
            "model_plant_gap": st.get("model_plant_gap"),
            "current_rto_setpoints": st.get("setpoints")}


def get_rto_status(ctx: AgentContext) -> dict:
    if ctx.rto_loop is not None:
        return ctx.rto_loop.get_rto_status()
    if ctx.rto is None:
        return {"status": "error", "message": "no RTO is configured in this context."}
    st = ctx.rto.get_status()
    return {"rto_has_run": bool(st), "rto_variant": st.get("type"),
            "commanded_setpoints": st.get("setpoints"),
            "rto_status_at_command": st}


def tool_schemas(ctx: AgentContext) -> list:
    """The OpenAI tool schemas available for this context (RTO tools only if an RTO is present)."""
    return TOOLS + (RTO_TOOLS if ctx.rto is not None else [])


def make_tool_registry(ctx: AgentContext) -> dict:
    """Bind ``ctx`` into the tool callables the agent loop invokes by name."""
    registry = {
        "get_process_state": lambda **kw: get_process_state(ctx),
        "get_mpc_health": lambda **kw: get_mpc_health(ctx),
        "update_mpc_target": lambda **kw: update_mpc_target(ctx, **kw),
    }
    if ctx.rto is not None:
        registry.update({
            "trigger_rto_run": lambda **kw: trigger_rto_run(ctx, **kw),
            "get_economic_context": lambda **kw: get_economic_context(ctx),
            "get_rto_status": lambda **kw: get_rto_status(ctx),
        })
    return registry
