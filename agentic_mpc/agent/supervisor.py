"""The supervisory-agent loop, refactored from ``test.py`` onto real Plant + Controller.

Same OpenAI-compatible client config (via :mod:`agentic_mpc.agent.llm_config`) and the
same system-prompt structure as the original skeleton, but the constructor now takes a
:class:`~agentic_mpc.interfaces.Plant` and :class:`~agentic_mpc.interfaces.Controller`
(and an optional :class:`~agentic_mpc.interfaces.SafetyEnvelope`), wires them into the
tools via an :class:`AgentContext`, and records a structured decision log for the
end-to-end experiment.
"""
from __future__ import annotations

import json
import re
from typing import Any

from agentic_mpc.agent.llm_config import LLM_CONFIG, LLMConfig, make_client
from agentic_mpc.agent.tools import AgentContext, make_tool_registry, tool_schemas
from agentic_mpc.agent.validation import validate_and_repair_args
from agentic_mpc.interfaces import Controller, Optimizer, Plant, SafetyEnvelope

SYSTEM_PROMPT = """You are a supervisory controller sitting above a Model Predictive \
Controller (MPC) that regulates a Wood-Berry binary distillation column.

The process has:
  - Two controlled variables (measured outputs): xD (overhead/distillate composition,
    nominally ~0.96 mole fraction) and xB (bottoms composition, nominally ~0.005).
  - Two manipulated variables (handled by the MPC): R (reflux flow) and S (steam flow),
    in lb/min.

Your job each supervisory cycle:
  1. Observe the process state (get_process_state) and the MPC health (get_mpc_health).
  2. Diagnose any issue: disturbances, model-plant mismatch (biased or growing
     innovation), constraint pressure, or degraded tracking (rising ISE).
  3. Take a supervisory action ONLY when warranted -- adjust the MPC setpoints via
     update_mpc_target -- and always give a clear, operator-readable rationale.

Reference values under healthy nominal operation (use these to judge the magnitudes you
read -- the raw numbers are small, so compare against these baselines):
  - innovation mean magnitude is ~1e-5 (essentially zero) for each output;
  - recent ISE is ~1e-6;
  - measured xD sits at ~0.96 and xB at ~0.005 (the nominal targets).
A SUSTAINED innovation-mean magnitude above ~5e-4 (tens of times nominal), OR a recent
ISE above ~1e-5 (about 10x nominal), OR a persistent offset of the measurement from its
nominal target, indicates a disturbance or model-plant mismatch that warrants attention.

Guidance:
  - The MPC innovation statistics (measured minus the MPC's prediction) are your main
    mismatch signal. Judge them against the reference values above, not in the abstract.
  - If you detect mismatch or a persistent tracking offset, take a supervisory action via
    update_mpc_target and explain your reasoning. In this phase your ONLY lever is the
    setpoints -- you cannot re-identify the model or change constraints -- so choose a
    setpoint adjustment that best trades off the situation, and say so.
  - When calling tools, provide CONCRETE VALUES, not schema definitions. The `rationale`
    field expects a plain string like "Lowering xD target to relieve reflux constraint
    pressure.", NOT an object like {"description": "...", "type": "string"}.
  - If the process is genuinely healthy and tracking well (magnitudes near the nominal
    references above), it is correct to take no action and simply report your assessment.

When you have finished your assessment and any actions, respond with a final summary
message (no further tool calls)."""

SYSTEM_PROMPT_RTO = SYSTEM_PROMPT + """

PHASE 1.5 -- you also supervise a real-time-optimization (RTO) layer that sets the MPC's
economically-optimal composition setpoints. Additional tools:
  - get_economic_context(): current prices/costs, last RTO objective, model-plant economic gap.
  - get_rto_status(): the last RTO command (setpoints, when, which variant, status, settled?).
  - trigger_rto_run(rationale): re-run the RTO and command fresh setpoints to the MPC.

Diagnose the regime before acting:
  * economic shift (a price/cost changed vs nominal -- check get_economic_context): the right
    response is to trigger_rto_run so the RTO re-optimizes for the new economics.
  * model-plant mismatch / load disturbance (biased innovation, rising ISE, persistent offset):
    a trigger_rto_run lets an adaptive RTO (MA/MA-GP) correct it; or adjust targets directly.
  * constraint change / RTO infeasibility (get_rto_status shows active constraints / infeasible):
    report it; do not command an out-of-spec target.
Prefer trigger_rto_run for economic re-optimization; reserve update_mpc_target for direct
supervisory setpoint nudges. Always give a concrete plain-string rationale."""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class SupervisoryAgent:
    """LLM supervisory agent that observes a closed loop and retunes the MPC."""

    def __init__(self, plant: Plant, controller: Controller,
                 safety: SafetyEnvelope | None = None,
                 config: LLMConfig = LLM_CONFIG,
                 system_prompt: str = SYSTEM_PROMPT,
                 rto: Optimizer | None = None, rto_loop: Any = None) -> None:
        self.ctx = AgentContext(plant=plant, controller=controller, safety=safety,
                                rto=rto, rto_loop=rto_loop)
        self.registry = make_tool_registry(self.ctx)
        self.tools = tool_schemas(self.ctx)     # RTO tools included iff an RTO is present
        self.config = config
        self.client = make_client(config)
        self.system_prompt = system_prompt

    def run_cycle(self, user_message: str, max_iterations: int = 8) -> dict[str, Any]:
        """One supervisory cycle: observe, reason, act.

        Returns ``{"final": str, "actions": list[dict], "iterations": int}`` where
        ``actions`` is the structured decision log (one entry per executed/ rejected tool
        call, with arguments and result) for downstream logging.
        """
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        actions: list[dict] = []

        for iteration in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.config.model, messages=messages, tools=self.tools,
                tool_choice="auto", temperature=self.config.temperature,
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                final = _strip_think(msg.content or "")
                return {"final": final, "actions": actions, "iterations": iteration + 1}

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                result, entry = self._execute_tool_call(tool_call, name, iteration)
                actions.append(entry)
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": json.dumps(result)})

        return {"final": "Agent hit max iterations without finishing.",
                "actions": actions, "iterations": max_iterations}

    # -- internals --------------------------------------------------------------------
    def _execute_tool_call(self, tool_call, name: str, iteration: int):
        base = {"iteration": iteration, "tool": name}
        # 1) parse JSON arguments
        try:
            raw_args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            err = f"Could not parse tool arguments as JSON: {e}"
            return ({"error": err, "hint": "Return valid JSON for the tool arguments."},
                    {**base, "status": "rejected", "error": err})

        # 2) validate / repair against the schema
        clean_args, err = validate_and_repair_args(name, raw_args)
        if err is not None:
            return ({"error": err, "hint": "Please retry with corrected argument types."},
                    {**base, "status": "rejected", "args": raw_args, "error": err})

        # 3) execute
        if name not in self.registry:
            err = f"Unknown tool: {name}"
            return {"error": err}, {**base, "status": "rejected", "error": err}
        result = self.registry[name](**clean_args)
        return result, {**base, "status": "executed", "args": clean_args, "result": result}


def _strip_think(text: str) -> str:
    """Remove qwen3-style <think>...</think> reasoning blocks from a final message."""
    return _THINK_RE.sub("", text).strip()
