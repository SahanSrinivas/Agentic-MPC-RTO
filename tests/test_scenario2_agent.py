"""Unit tests for the Scenario-2 DiagnosticSupervisor classification paths (synthetic diagnostics,
no sim) + its latching behavior. The end-to-end sim behavior is covered by the benchmark."""
from __future__ import annotations

from agentic_mpc.scenario2_agent import DiagnosticSupervisor

_NOISE = 1e-5


def diag(iD=_NOISE, iB=_NOISE, oD=0.0, oB=0.0, constraints=None, sp=None):
    return {"innovation_mean": {"xD": iD, "xB": iB},
            "steady_state_offset": {"xD": oD, "xB": oB},
            "active_constraints": constraints or [], "setpoints": sp or {"xD": 0.96, "xB": 0.005}}


def snap(xD=0.96, xB=0.005, hist_xD=None, hist_xB=None, n=30):
    hxD = hist_xD if hist_xD is not None else [0.96] * n
    hxB = hist_xB if hist_xB is not None else [0.005] * n
    return {"y": {"xD": xD, "xB": xB}, "is_simulation": True,
            "history": {"t": list(range(n)), "y": {"xD": hxD, "xB": hxB},
                        "u": {"R": [1.95] * n, "S": [1.71] * n}}}


def test_nominal_holds():
    d = DiagnosticSupervisor().assess(diag(), snap())
    assert d.state == "NOMINAL" and d.action == "HOLD"


def test_implausible_reading_is_vetoed_as_sensor_fault():
    # current reading is back in range (MPC pulled it down) but the window holds a >1 excursion.
    s = snap(xD=0.985, hist_xD=[0.96] * 27 + [1.012, 1.008, 0.99])
    d = DiagnosticSupervisor().assess(diag(iD=4.5e-3), s)
    assert d.state == "SENSOR_FAULT" and d.action == "VETO_HOLD"


def test_coupled_innovation_proposes_bounded_setpoint():
    d = DiagnosticSupervisor().assess(diag(iD=-2e-3, iB=1.2e-3, oD=0.01, oB=0.004), snap())
    assert d.state == "REAL_DISTURBANCE" and d.action == "PROPOSE_SETPOINT"
    assert d.proposed_targets is not None and set(d.proposed_targets) == {"xD", "xB"}


def test_single_channel_innovation_escalates():
    # xD innovation biased, xB at its nominal floor -> load vs in-range sensor bias not separable.
    d = DiagnosticSupervisor().assess(diag(iD=-2e-3, iB=_NOISE, oD=0.01), snap())
    assert d.state == "AMBIGUOUS" and d.action == "ESCALATE"


def test_offset_without_innovation_is_a_tracking_transient_not_a_fault():
    # raw offset with clean innovation = setpoint-tracking lag (e.g. right after a setpoint change),
    # NOT a model-plant fault -> must not trigger (otherwise it false-blocks the supervisor's own move).
    d = DiagnosticSupervisor().assess(diag(iD=_NOISE, iB=_NOISE, oD=0.01), snap())
    assert d.state == "NOMINAL" and d.action == "HOLD"


def test_diagnosis_latches_after_signature_decays():
    sup = DiagnosticSupervisor()
    s = snap(xD=0.985, hist_xD=[0.96] * 27 + [1.012, 1.008, 0.99])
    assert sup.assess(diag(iD=4.5e-3), s).state == "SENSOR_FAULT"
    # signature gone (clean nominal cycle) -> the serious finding is NOT forgotten.
    assert sup.assess(diag(), snap()).state == "SENSOR_FAULT"


def test_latch_upgrades_by_severity_then_resets():
    sup = DiagnosticSupervisor()
    assert sup.assess(diag(iD=-2e-3, iB=_NOISE), snap()).state == "AMBIGUOUS"
    # a coupled signature later upgrades the latch (REAL_DISTURBANCE > AMBIGUOUS) ...
    assert sup.assess(diag(iD=-2e-3, iB=1.2e-3), snap()).state == "REAL_DISTURBANCE"
    # ... and a sensor fault upgrades it again (highest severity).
    s = snap(hist_xD=[0.96] * 29 + [1.02])
    assert sup.assess(diag(iD=-2e-3, iB=1.2e-3), s).state == "SENSOR_FAULT"
    sup.reset()
    assert sup.assess(diag(), snap()).state == "NOMINAL"
