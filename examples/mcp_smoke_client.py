"""Smoke client for the Wood-Berry MPC MCP server (exercises the real stdio protocol).

Spawns `python -m agentic_mpc.mcp_server` over stdio, lists its tools, and runs a realistic
sequence: reset with the R7 load scenario, advance across the disturbance, read diagnostics, and
attempt an out-of-box setpoint (to show the safety clip). No GUI / Node needed.

Run:  python examples/mcp_smoke_client.py
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = pathlib.Path(__file__).resolve().parents[1]


def _payload(result) -> dict:
    """Pull the JSON dict out of a CallToolResult (FastMCP fills structuredContent for dict returns)."""
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    for block in result.content:
        if getattr(block, "text", None):
            return json.loads(block.text)
    return {}


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "agentic_mpc.mcp_server"],
                                   cwd=str(REPO))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print("tools:", sorted(t.name for t in tools))

            print("info:", _payload(await session.call_tool("info", {}))["controller"])
            print("reset:", _payload(await session.call_tool(
                "reset_sim", {"seed": 1, "scenario": "R7"}))["status"], "(scenario R7, seed 1)")

            await session.call_tool("advance", {"minutes": 90})        # pre-event (load fires at t=100)
            d0 = _payload(await session.call_tool("get_mpc_diagnostics", {}))
            print(f"t=90  pre-event : innov_xD={d0['innovation_mean']['xD']:+.2e} "
                  f"offset_xD={d0['steady_state_offset']['xD']:+.4f}")

            await session.call_tool("advance", {"minutes": 60})        # cross the load disturbance
            d1 = _payload(await session.call_tool("get_mpc_diagnostics", {}))
            print(f"t=150 post-load : innov_xD={d1['innovation_mean']['xD']:+.2e} "
                  f"offset_xD={d1['steady_state_offset']['xD']:+.4f}  (MPC surfaces the load)")

            clip = _payload(await session.call_tool(
                "set_target", {"xD": 1.20, "xB": 0.004, "rationale": "smoke: out-of-box"}))
            print(f"set_target(1.20) -> applied={clip['applied_targets']} "
                  f"clipped={clip['clipped_by_safety']}")

            snap = _payload(await session.call_tool("get_plant_snapshot", {}))
            print(f"snapshot: y={ {k: round(v,4) for k,v in snap['y'].items()} } "
                  f"is_simulation={snap['is_simulation']}")
    print("OK - server spoke MCP over stdio.")


if __name__ == "__main__":
    asyncio.run(main())
