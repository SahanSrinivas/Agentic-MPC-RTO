"""Tests for the MCP MPC sandbox (the hostable sim) + a server-registration smoke test.

The MCP transport itself isn't exercised (that's stdio integration); we test the sandbox logic the
server wraps, plus that the FastMCP server registers the expected tools.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from agentic_mpc.mcp_sandbox import MPCSandbox

warnings.filterwarnings("ignore")


def test_reset_starts_at_nominal_optimum():
    sb = MPCSandbox(seed=1)
    r = sb.reset(seed=1)
    assert r["t"] == 0.0 and r["scenario"] is None
    assert abs(r["targets"]["xD"] - 0.96) < 1e-9 and abs(r["targets"]["xB"] - 0.005) < 1e-9


def test_advance_evolves_state_and_fills_diagnostics():
    sb = MPCSandbox(seed=1)
    sb.advance(30)
    assert sb.t == 30.0 and len(sb.history["t"]) == 30
    d = sb.get_mpc_diagnostics()
    for k in ("innovation_mean", "innovation_std", "ise_recent", "active_constraints",
              "steady_state_offset", "setpoints"):
        assert k in d
    assert set(d["innovation_mean"]) == {"xD", "xB"}
    # near nominal, tracking the optimum: offset stays small.
    assert abs(d["steady_state_offset"]["xD"]) < 0.02


def test_snapshot_shape_and_is_simulation():
    sb = MPCSandbox(seed=1); sb.advance(10)
    s = sb.get_plant_snapshot()
    assert s["is_simulation"] is True
    assert set(s["y"]) == {"xD", "xB"} and set(s["u"]) == {"R", "S"}
    assert "history" in s and "t" in s["history"]


def test_set_target_clips_through_safety_box():
    sb = MPCSandbox(seed=1)
    r = sb.set_target(xD=1.5, xB=0.5, rationale="push out of range")   # both out of box
    assert r["clipped_by_safety"] is True
    assert r["applied_targets"]["xD"] == 0.99 and r["applied_targets"]["xB"] == 0.05
    r2 = sb.set_target(xD=0.95, xB=0.004, rationale="in range")
    assert r2["clipped_by_safety"] is False
    assert r2["applied_targets"] == {"xD": 0.95, "xB": 0.004}


def test_set_target_takes_effect_in_tracking():
    sb = MPCSandbox(seed=1); sb.advance(20)
    sb.set_target(xD=0.94, xB=0.004, rationale="lower xD")
    sb.advance(120)                                   # let the MPC drive to the new target
    s = sb.get_plant_snapshot()
    assert abs(s["y"]["xD"] - 0.94) < 0.01            # measured xD tracked the new setpoint


def test_mpc_only_scenario_arms_a_plant_disturbance():
    sb = MPCSandbox(seed=1)
    sb.reset(seed=1, scenario="R7")                   # xD load disturbance fires at t>=100
    sb.advance(130)
    assert abs(sb.plant._output_bias[0] - (-0.03)) < 1e-9   # the real load is active


def test_economic_scenarios_are_rejected_on_mpc_only_server():
    sb = MPCSandbox(seed=1)
    with pytest.raises(ValueError):
        sb.reset(seed=1, scenario="R3")               # needs an RTO layer
    # the tool wrapper turns that into a clean error dict (not an exception to the client):
    from agentic_mpc import mcp_server
    assert mcp_server.reset_sim(seed=1, scenario="R3")["status"] == "error"


def test_sandbox_is_seed_deterministic():
    a, b = MPCSandbox(seed=7), MPCSandbox(seed=7)
    a.advance(40); b.advance(40)
    assert np.allclose(a.history["xD"], b.history["xD"])
    assert np.allclose(a.history["xB_true"], b.history["xB_true"])


def test_server_registers_expected_tools():
    import asyncio

    from agentic_mpc.mcp_server import mcp
    names = sorted(t.name for t in asyncio.run(mcp.list_tools()))
    assert names == ["advance", "get_mpc_diagnostics", "get_plant_snapshot",
                     "info", "reset_sim", "set_target"]
