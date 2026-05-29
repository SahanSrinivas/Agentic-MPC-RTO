"""MCP server exposing the Wood-Berry MPC sim sandbox (Milestone 1, MPC-only -- no RTO, no LLM).

A client (Claude Desktop, Cursor, custom) connects over stdio and can drive the column and inspect
MPC health: reset the sim, advance it, read diagnostics / a plant snapshot, and nudge the setpoint
(the only write -- and it is clipped through the safety envelope, on a simulation). The regulator
(condensed-QP SLSQP MPC) is never bypassed: no tool sets R/S directly.

Run:
    python -m agentic_mpc.mcp_server          # stdio server
    # or, after `pip install -e .[mcp]`:
    agentic-mpc-mcp

Register with an MCP client by pointing it at that command (stdio transport).
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from agentic_mpc.mcp_sandbox import MPCSandbox

mcp = FastMCP("woodberry-mpc")
_sandbox = MPCSandbox(seed=0)


@mcp.tool()
def info() -> dict[str, Any]:
    """Describe the hosted sim: plant, controller, dt, seed, armed scenario, the safety box, and the
    scenarios this MPC-only server supports."""
    return _sandbox.info()


@mcp.tool()
def reset_sim(seed: int = 0, scenario: str | None = None) -> dict[str, Any]:
    """Rebuild the Wood-Berry plant + MPC at the nominal operating point (xD=0.96, xB=0.005) and clear
    history. Optionally arm a plant-disturbance scenario: one of R1 (slow feed drift), R2 (abrupt
    efficiency loss), R6 (xD analyzer gross error), R7 (xD load disturbance), S1 (real load then sensor
    fault). Economic scenarios (R3/R4/R5) need an RTO layer and are not available on this MPC-only
    server."""
    try:
        return _sandbox.reset(seed=seed, scenario=scenario)
    except ValueError as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def advance(minutes: float = 30.0) -> dict[str, Any]:
    """Step the closed loop forward by `minutes` (dt = 1 min). The MPC tracks the current setpoint,
    the plant responds, and any armed scenario perturbs the plant. Returns the new time and latest
    measured (xD, xB). Call get_mpc_diagnostics afterward to see how innovation/offset evolved."""
    return _sandbox.advance(minutes)


@mcp.tool()
def get_mpc_diagnostics() -> dict[str, Any]:
    """MPC health: innovation_mean/std per output (measured minus the MPC's prediction -- the
    model-plant mismatch signal), recent ISE, active constraints, the steady-state offset (y - y_sp),
    and the current setpoints. This is the supervisory evidence the no-integral MPC deliberately
    exposes rather than absorbing."""
    return _sandbox.get_mpc_diagnostics()


@mcp.tool()
def get_plant_snapshot() -> dict[str, Any]:
    """Current measured state: controlled vars (xD, xB), manipulated vars (R, S), time, and the recent
    history window. `is_simulation` is always true (this is a sim sandbox)."""
    return _sandbox.get_plant_snapshot()


@mcp.tool()
def set_target(xD: float, xB: float, rationale: str) -> dict[str, Any]:
    """Propose a new composition setpoint (xD, xB) for the MPC to track. It is clipped through the
    safety envelope (xD in [0.90, 0.99], xB in [0.001, 0.05]) before being applied; the response flags
    whether clipping occurred. This is the ONLY write -- it never sets R/S directly. Provide a plain
    string rationale."""
    return _sandbox.set_target(xD, xB, rationale)


def main() -> None:
    """Entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
