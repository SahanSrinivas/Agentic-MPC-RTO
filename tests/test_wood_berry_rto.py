"""Standalone validation for the classical Wood-Berry NLP-RTO (Phase 1.5, Gate 2).

Verifies the nominal optimum reproduces Phase 1's control targets and that the three
economic/constraint scenarios (r3 steam-cost spike, r4 demand cap, r5 infeasibility) produce
the intended RTO behavior -- BEFORE the MA/MA-GP comparators are layered on top.
"""
from __future__ import annotations

import numpy as np

from agentic_mpc.interfaces import Optimizer
from agentic_mpc.rto import WoodBerryRTO


def test_implements_optimizer_interface():
    rto = WoodBerryRTO()
    assert isinstance(rto, Optimizer)
    md = rto.metadata
    assert md["setpoint_names"] == ["xD", "xB"] and md["type"] == "nominal"


def test_nominal_optimum_matches_phase1_targets():
    """Nominal economic optimum must land on Phase 1's hardcoded targets (continuity)."""
    rto = WoodBerryRTO()
    res = rto.solve()
    assert res["converged"]
    assert abs(res["setpoints"]["xD"] - 0.96) < 2e-3, res["setpoints"]
    assert abs(res["setpoints"]["xB"] - 0.005) < 5e-4, res["setpoints"]
    assert abs(res["inputs"]["R"] - 1.95) < 5e-3, res["inputs"]
    assert abs(res["inputs"]["S"] - 1.71) < 5e-3, res["inputs"]
    assert res["active_constraints"] == []  # interior optimum, nothing binding


def test_r3_steam_cost_spike_backs_off_purity():
    rto = WoodBerryRTO()
    base = rto.solve()["setpoints"]["xD"]
    econ2 = rto.economics.with_overrides(c_S=2 * rto.economics.params.c_S)
    spiked = rto.solve({"economics": econ2})["setpoints"]
    assert spiked["xD"] < base - 0.01, (base, spiked)   # energy expensive -> less separation
    assert spiked["xB"] > 0.005                          # bottoms less pure


def test_r4_demand_cap_drives_higher_purity():
    rto = WoodBerryRTO()
    base = rto.solve()["setpoints"]["xD"]
    econ2 = rto.economics.with_overrides(D_max=0.50)
    res = rto.solve({"economics": econ2})
    assert res["setpoints"]["xD"] > base + 0.01         # demand-capped -> purer distillate
    assert "D_max" in res["active_constraints"]


def test_r5_tightened_spec_pins_to_boundary():
    rto = WoodBerryRTO()
    econ2 = rto.economics.with_overrides(xB_max=0.002)
    res = rto.solve({"economics": econ2})
    assert res["converged"]
    assert abs(res["setpoints"]["xB"] - 0.002) < 2e-4    # pinned to the tightened spec
    assert "xB_max" in res["active_constraints"]


def test_r5_infeasible_below_achievable():
    rto = WoodBerryRTO()
    econ2 = rto.economics.with_overrides(xB_max=0.0008)  # below the physical box minimum
    res = rto.solve({"economics": econ2})
    assert res["converged"] is False and res["status"] == "infeasible"


def test_setpoint_to_inputs_nominal():
    rto = WoodBerryRTO()
    R, S = rto.setpoint_to_inputs(0.96, 0.005)
    assert abs(R - 1.95) < 1e-9 and abs(S - 1.71) < 1e-9


def test_get_status_after_solve():
    rto = WoodBerryRTO()
    rto.solve()
    st = rto.get_status()
    assert {"setpoints", "objective", "converged", "active_constraints"} <= set(st)
