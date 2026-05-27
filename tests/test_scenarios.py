"""Unit tests for the Phase-1.5 scenarios R1-R7 (perturbation logic; no LLM).

Each scenario is a pure perturbation of the running stack. These tests drive each scenario's
``on_step`` against a real (baseline, agent-free) RTO-MPC loop and assert the documented regime
effect, plus one end-to-end baseline loop smoke that the stack integrates.
"""
from __future__ import annotations

import warnings

import numpy as np

from agentic_mpc.controllers import ClassicalMPC
from agentic_mpc.plants import WoodBerryPlant
from agentic_mpc.rto import RTOMPCLoop, WoodBerryRTO
from agentic_mpc.scenarios import SCENARIOS

warnings.filterwarnings("ignore")


def _loop(rto_interval_min=30.0):
    plant = WoodBerryPlant(dt=1.0, seed=0)
    mpc = ClassicalMPC(dt=1.0)
    rto = WoodBerryRTO(plant_params=plant.params)
    loop = RTOMPCLoop(plant, mpc, rto, rto_interval_min=rto_interval_min)
    return loop


def test_scenarios_registry_complete():
    assert set(SCENARIOS) == {f"R{i}" for i in range(1, 8)}
    for sid, cls in SCENARIOS.items():
        d = cls().describe()
        assert d["scenario_id"] == sid and d["regime"] and d["mechanism"]


def test_r1_ramps_output_bias():
    loop = _loop(); sc = SCENARIOS["R1"](t_start=50, t_end=150, final_xD_bias=-0.03)
    sc.on_step(loop, 100.0, None)                       # halfway -> bias ~ -0.015
    assert abs(loop.plant._output_bias[0] - (-0.015)) < 1e-6


def test_r2_steps_gain_and_bias():
    loop = _loop(); sc = SCENARIOS["R2"](t_event=100)
    sc.on_step(loop, 99.0, None); assert loop.plant._gain_mult[0, 0] == 1.0   # not yet
    sc.on_step(loop, 100.0, None)
    assert loop.plant._gain_mult[0, 0] == 0.90 and loop.plant._output_bias[0] == -0.015


def test_r3_spikes_steam_cost_and_requests_recompute():
    loop = _loop(); base = loop.optimizer.economics.params.c_S
    SCENARIOS["R3"](t_event=100, factor=2.0).on_step(loop, 100.0, None)
    assert abs(loop.optimizer.economics.params.c_S - 2 * base) < 1e-12
    assert loop._pending_rto is True


def test_r4_imposes_demand_cap():
    loop = _loop()
    SCENARIOS["R4"](t_event=100, D_max=0.50).on_step(loop, 100.0, None)
    assert loop.optimizer.economics.params.D_max == 0.50 and loop._pending_rto is True


def test_r5_tightens_spec_to_infeasible():
    loop = _loop()
    SCENARIOS["R5"](t_event=100, xB_max=0.0008).on_step(loop, 100.0, None)
    assert loop.optimizer.economics.params.xB_max == 0.0008
    res = loop.optimizer.solve()                        # RTO must report infeasible
    assert res["converged"] is False and res["status"] == "infeasible"


def test_r6_applies_then_clears_sensor_bias():
    loop = _loop(); sc = SCENARIOS["R6"](t_start=100, t_end=130, bias=0.05)
    sc.on_step(loop, 100.0, None); assert loop.plant._sensor_bias[0] == 0.05
    sc.on_step(loop, 130.0, None); assert loop.plant._sensor_bias[0] == 0.0
    assert loop.plant._output_bias[0] == 0.0            # true state untouched (sensor-only)


def test_r7_steps_load_disturbance():
    loop = _loop()
    SCENARIOS["R7"](t_event=100, xD_bias=-0.03).on_step(loop, 100.0, None)
    assert loop.plant._output_bias[0] == -0.03


def test_r5_infeasible_does_not_crash_through_loop_with_ma():
    """Reproduces the reported crash path: R5 (sub-achievable xB_max) + MA RTO through the loop.
    Must NOT raise ValueError(high<low), and the RTO must report infeasibility to the supervisor."""
    from agentic_mpc.rto import ModifierAdaptation
    plant = WoodBerryPlant(dt=1.0, seed=42)
    mpc = ClassicalMPC(dt=1.0)
    rto = ModifierAdaptation(plant=plant, plant_params=plant.params, seed=0)
    loop = RTOMPCLoop(plant, mpc, rto, rto_interval_min=30.0)
    sc = SCENARIOS["R5"](t_event=60, xB_max=0.0008)
    loop.run(150.0, on_step=sc.on_step)                        # must NOT raise
    st = loop.get_rto_status()
    assert st["rto_converged"] is False                        # surfaced infeasibility to the agent
    assert st["rto_solve_status"] == "infeasible"


def test_baseline_loop_smoke_runs_end_to_end():
    """A full baseline (no-agent) RTO-MPC loop with R7 integrates and stays physical."""
    loop = _loop(rto_interval_min=30.0)
    sc = SCENARIOS["R7"](t_event=60)
    res = loop.run(120.0, on_step=sc.on_step)
    h = res["history"]
    assert len(h["t"]) == 121
    assert res["handoffs"], "RTO should have commanded at least once"
    assert 0.85 < h["xD"][-1] < 1.0 and -0.01 < h["xB"][-1] < 0.06   # physical
