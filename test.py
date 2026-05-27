"""
Minimal agentic controller skeleton.
Points at Ollama for development/paper; swap base_url for Claude in production.
"""

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Dict
import json

# --- LLM client: this is the only place the backend is configured ---
LLM_CONFIG = {
    "base_url": "http://localhost:11434/v1",  # Ollama default
    "api_key": "ollama",                       # Ollama ignores this but the SDK requires a value
    "model": "qwen3:4b",                       # Whatever you pulled
}

client = OpenAI(
    base_url=LLM_CONFIG["base_url"],
    api_key=LLM_CONFIG["api_key"],
)


# --- Tool definitions (OpenAI function-calling schema) ---
# These are the "universal interface" methods your agent calls.
# Right now they're stubs; later they'll wrap your real Plant / Controller objects.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_process_state",
            "description": "Returns the current measured state of the process (controlled variables, manipulated variables, recent history).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mpc_health",
            "description": "Returns MPC health metrics: innovation statistics, active constraints, recent ISE.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_mpc_target",
            "description": "Push new setpoint targets to the MPC. Must include a rationale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "description": "Dictionary of controlled variable names to target values, e.g. {\"xD\": 0.95, \"xB\": 0.05}",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Operator-readable plain-string explanation of why this change is being made. Example: 'Reducing R target due to constraint pressure'. Do NOT pass an object here.",
                    },
                },
                "required": ["targets", "rationale"],
            },
        },
    },
]


# --- Argument schemas (Pydantic) for validating LLM tool calls ---

class UpdateMpcTargetArgs(BaseModel):
    targets: Dict[str, float] = Field(..., description="CV name -> target value")
    rationale: str = Field(..., min_length=10, description="Operator-readable explanation")


TOOL_SCHEMAS = {
    "update_mpc_target": UpdateMpcTargetArgs,
    # Other tools that take args will be registered here later.
}


def validate_and_repair_args(tool_name: str, raw_args: dict):
    """
    Returns (validated_args, error_message).
    If error_message is not None, the agent should be told the call was
    rejected and asked to retry with corrected arguments.
    """
    schema = TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        # No schema registered; pass through as-is.
        return raw_args, None

    # --- Repair pass: known small-model failure modes ---
    if tool_name == "update_mpc_target":
        r = raw_args.get("rationale")
        # Small models sometimes return rationale as the schema definition itself,
        # e.g. {"description": "...", "type": "string"} instead of a plain string.
        if isinstance(r, dict) and "description" in r:
            raw_args = {**raw_args, "rationale": r["description"]}

    # --- Validate against schema ---
    try:
        validated = schema(**raw_args)
        return validated.model_dump(), None
    except ValidationError as e:
        return None, f"Tool call rejected: arguments did not match schema. Details: {e}"


# --- Tool implementations (stubs for now) ---

def get_process_state():
    return {
        "y": {"xD": 0.92, "xB": 0.08},   # xD drifted down from 0.96, xB up from 0.05
        "u": {"R": 2.35, "S": 1.71},     # R has climbed from 1.95
        "t": 1800.0,
        "history_note": "xD has been trending downward over the last 20 minutes; R has been increasing.",
    }

def get_mpc_health():
    return {
        "innovation_mean": 0.018,         # was 0.002 — biased now
        "innovation_std": 0.042,          # was 0.015 — variance up
        "active_constraints": ["R_upper"],  # MPC is hitting reflux limit
        "ise_last_300s": 0.029,           # was 0.0048 — 6x worse
    }

def update_mpc_target(targets, rationale):
    # Later: call your Controller.set_targets()
    print(f"  [ACTION] Setting targets to {targets}")
    print(f"  [RATIONALE] {rationale}")
    return {"status": "ok", "applied_targets": targets}


TOOL_REGISTRY = {
    "get_process_state": get_process_state,
    "get_mpc_health": get_mpc_health,
    "update_mpc_target": update_mpc_target,
}


# --- The agent loop ---

SYSTEM_PROMPT = """You are a supervisory controller for a chemical process.
You sit above an MPC controller. Your job is to:
1. Observe the process state and MPC health.
2. Diagnose any issues (disturbances, model mismatch, degradation).
3. Take supervisory actions (adjust targets, constraints) when warranted.
4. Always provide a clear rationale for every action.

When calling tools, provide CONCRETE VALUES for each argument, not schema definitions.
For example, the `rationale` field expects a plain string like:
  "Reducing R target due to constraint pressure and rising innovation variance."
NOT an object like:
  {"description": "...", "type": "string"}

Use the provided tools. When you have completed your assessment and taken any
necessary actions, respond with a final summary message (no further tool calls)."""


def run_agent_step(user_message, max_iterations=8):
    """One supervisory cycle: observe, reason, act."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # If no tool calls, the agent is done.
        if not msg.tool_calls:
            print(f"\n[AGENT FINAL]: {msg.content}\n")
            return msg.content

        # Execute each tool call (with validation).
        for tool_call in msg.tool_calls:
            name = tool_call.function.name

            # 1) Parse the JSON the LLM produced.
            try:
                raw_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                err = f"Could not parse tool arguments as JSON: {e}"
                print(f"  [TOOL REJECTED] {name}: {err}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "error": err,
                        "hint": "Return valid JSON for the tool arguments.",
                    }),
                })
                continue

            # 2) Validate / repair against the schema.
            clean_args, err = validate_and_repair_args(name, raw_args)
            if err is not None:
                print(f"  [TOOL REJECTED] {name}: {err}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "error": err,
                        "hint": "Please retry with corrected argument types.",
                    }),
                })
                continue

            print(f"  [TOOL CALL] {name}({clean_args})")

            # 3) Execute.
            if name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[name](**clean_args)
            else:
                result = {"error": f"Unknown tool: {name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    return "Agent hit max iterations without finishing."


if __name__ == "__main__":
    trigger = (
        "A new supervisory cycle has started. "
        "Please assess the process and MPC, and take any actions you deem necessary."
    )
    run_agent_step(trigger)