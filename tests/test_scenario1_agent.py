"""Unit tests for the Scenario-1-lite diagnostics-gated economic supervisor."""
from __future__ import annotations

from agentic_mpc.scenario1_agent import EconWeights, Scenario1Agent, margin, steady_state_inputs

_NOISE = 1e-5


def _diag(iD=_NOISE, iB=_NOISE, oD=0.0, oB=0.0, constraints=None):
    return {"innovation_mean": {"xD": iD, "xB": iB},
            "steady_state_offset": {"xD": oD, "xB": oB},
            "active_constraints": constraints or [], "setpoints": {"xD": 0.96, "xB": 0.005}}


def _snap(xD=0.96, xB=0.005, hist_xD=None, n=30):
    hxD = hist_xD if hist_xD is not None else [0.96] * n
    return {"y": {"xD": xD, "xB": xB},
            "history": {"t": list(range(n)), "y": {"xD": hxD, "xB": [0.005] * n},
                        "u": {"R": [1.95] * n, "S": [1.71] * n}}}


def test_optimizer_returns_feasible_point_beating_nominal():
    opt = Scenario1Agent().optimize()
    assert 0.90 <= opt["xD"] <= 0.99 and 0.001 <= opt["xB"] <= 0.05      # inside the envelope
    w = EconWeights()
    _, S_nom = steady_state_inputs(0.96, 0.005)
    assert opt["J"] >= margin(0.96, 0.005, S_nom, w) - 1e-9              # not worse than nominal


def test_green_state_runs_economics():
    res = Scenario1Agent().step(_diag(), _snap())
    assert res["gated"] is False and res["s2_action"] == "OPTIMIZE"
    assert res["proposed"] is not None


def test_sensor_fault_blocks_economics():
    s = _snap(xD=0.985, hist_xD=[0.96] * 28 + [1.013, 0.99])            # implausible excursion in window
    res = Scenario1Agent().step(_diag(iD=4.5e-3), s)
    assert res["gated"] is True and res["s2_state"] == "SENSOR_FAULT" and res["proposed"] is None


def test_coupled_disturbance_blocks_economics():
    res = Scenario1Agent().step(_diag(iD=-2e-3, iB=1.2e-3, oD=0.01), _snap())
    assert res["gated"] is True and res["s2_state"] == "REAL_DISTURBANCE"


def test_single_channel_blocks_economics_via_escalation():
    res = Scenario1Agent().step(_diag(iD=-2e-3, iB=_NOISE, oD=0.01), _snap())
    assert res["gated"] is True and res["s2_state"] == "AMBIGUOUS"


def test_gate_latches_after_event_even_when_signature_clears():
    agent = Scenario1Agent()
    assert agent.step(_diag(iD=-2e-3, iB=1.2e-3), _snap())["gated"] is True   # coupled -> blocked + latched
    # signal clears, but the latched Scenario-2 finding keeps economics gated.
    assert agent.step(_diag(), _snap())["gated"] is True
