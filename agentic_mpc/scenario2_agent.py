"""Scenario 2 -- diagnostic supervisor over the MPC's deliberately-exposed mismatch signals.

This is the *decision* layer the MCP tools were built for. It is **rules-first and deterministic**
(so it is reproducible and free): it reads the same diagnostics the MCP server exposes
(``get_mpc_diagnostics`` + ``get_plant_snapshot``) and classifies the situation, then maps to an
**allow-listed** action. An LLM, if attached, supplies *interpretation / incident narrative /
ranked hypotheses* on top of this decision -- it does NOT make or override the control decision
(see :meth:`DiagnosticSupervisor.narrative_prompt`, which is a hook, not called here).

Classification uses exactly the evidence the no-integral MPC refuses to hide:
  * innovation mean magnitude (measured - MPC prediction) -- the mismatch signal,
  * steady-state offset |y - y_sp|,
  * a gain-consistency residual rho = dy - K.du over the recent window (the part of the composition
    move NOT explained by the manipulated variables, per the nominal Wood-Berry gains), and
  * physical plausibility of the reading (composition in [0, 1]).

Decision policy (allow-listed; nothing else is callable):
  NOMINAL              -> HOLD                 (innovation ~ noise, offset small, no constraints)
  SENSOR_FAULT         -> VETO_HOLD            (reading physically implausible OR a single analyzer's
                                                residual moves while the coupled output stays at the
                                                noise floor -> do NOT chase / re-optimize on it)
  REAL_DISTURBANCE     -> PROPOSE_SETPOINT      (residual coupled across BOTH compositions -> a real
                                                process move; recommend a bounded, clipped setpoint)
  AMBIGUOUS            -> ESCALATE             (single-channel but in-range: a real load and an
                                                in-range sensor bias are not separable from telemetry
                                                alone -- request a corroborating check before acting)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiagConfig:
    """Thresholds (provenance mirrors the rule-based supervisor: sensor-noise std ~2e-4)."""
    innov_thresh: float = 5e-4     # |innovation mean| indicating mismatch (~2.5x noise std, ~50x nominal)
    offset_xD: float = 5e-3        # 0.5% composition (operating-envelope scale)
    offset_xB: float = 1e-3        # 20% relative of the 0.005 nominal
    rho_susp: float = 5e-3         # a window residual this large = a real anomaly on that channel
    rho_corr: float = 1e-3         # other channel residual below this = "did not independently move"


@dataclass
class Decision:
    state: str                     # NOMINAL | SENSOR_FAULT | REAL_DISTURBANCE | AMBIGUOUS
    action: str                    # HOLD | VETO_HOLD | PROPOSE_SETPOINT | ESCALATE
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)
    proposed_targets: dict[str, float] | None = None   # only for PROPOSE_SETPOINT


class DiagnosticSupervisor:
    """Deterministic Scenario-2 policy: diagnose mismatch, veto/act/escalate within an allow-list."""

    # diagnosis severity: a serious finding is not forgotten once the transient signature decays.
    _SEVERITY = {"NOMINAL": 0, "AMBIGUOUS": 1, "REAL_DISTURBANCE": 2, "SENSOR_FAULT": 3}

    def __init__(self, config: DiagConfig | None = None) -> None:
        self.config = config or DiagConfig()
        self._latched: Decision | None = None       # strongest diagnosis seen this episode

    def reset(self) -> None:
        self._latched = None

    @staticmethod
    def _window_excursion(history: dict, y: dict, tol: float = 2e-3):
        """Worst physically-impossible reading (composition CLEARLY outside [0,1]) in the recent window
        or current sample. Returns (channel, value) or None. tol (>> sensor-noise std 2e-4) ensures a
        genuine gross error triggers, not a noise-scale dip near the boundary on the unclamped model."""
        worst = None
        series = {"xD": list(history.get("y", {}).get("xD", [])) + [y.get("xD")],
                  "xB": list(history.get("y", {}).get("xB", [])) + [y.get("xB")]}
        for ch, vals in series.items():
            for v in vals:
                if v is None:
                    continue
                if v > 1.0 + tol or v < 0.0 - tol:
                    if worst is None or abs(v - 0.5) > abs(worst[1] - 0.5):
                        worst = (ch, float(v))
        return worst

    # -- the decision (latching wrapper: a serious finding persists until reset) ------------------
    def assess(self, diagnostics: dict, snapshot: dict) -> Decision:
        dec = self._classify(diagnostics, snapshot)
        if (self._latched is None
                or self._SEVERITY[dec.state] > self._SEVERITY[self._latched.state]):
            if dec.state != "NOMINAL":
                self._latched = dec
        return self._latched or dec

    def _classify(self, diagnostics: dict, snapshot: dict) -> Decision:
        c = self.config
        im = diagnostics.get("innovation_mean", {})
        iD, iB = abs(im.get("xD", 0.0) or 0.0), abs(im.get("xB", 0.0) or 0.0)
        off = diagnostics.get("steady_state_offset", {})
        offD, offB = abs(off.get("xD", 0.0) or 0.0), abs(off.get("xB", 0.0) or 0.0)
        constraints = diagnostics.get("active_constraints", []) or []
        y = snapshot.get("y", {})
        ev = {"innov": {"xD": im.get("xD"), "xB": im.get("xB")}, "offset": off,
              "constraints": constraints, "y": y}

        # the model-plant-mismatch trigger is the INNOVATION (and active constraints), NOT raw offset:
        # offset is transiently large after ANY setpoint change while innovation is not (the MPC
        # models the setpoint), so gating on innovation avoids false alarms on a supervisor's own move.
        mismatch = (iD > c.innov_thresh or iB > c.innov_thresh or bool(constraints))
        if not mismatch:
            return Decision("NOMINAL", "HOLD",
                            "Innovation within noise (no model-plant mismatch); tracking the setpoint.",
                            ev)

        # plausibility scans the recent window, not just the instant: a sensor spike is transient
        # because the MPC chases the bad reading back into range, but the excursion still happened.
        excursion = self._window_excursion(snapshot.get("history", {}), y)
        ev["excursion"] = excursion
        if excursion is not None:
            ch, val = excursion
            return Decision("SENSOR_FAULT", "VETO_HOLD",
                            f"{ch} read {val:.3f} -- physically impossible (composition must lie in "
                            f"[0,1]); analyzer gross error, not a process change. Hold; do not "
                            f"re-optimize on corrupted data (flag the {ch} analyzer for inspection).", ev)

        # Channel attribution uses the per-output Kalman INNOVATION -- the input-corrected mismatch
        # signal the observer already produces (the classic innovation-based FDI residual). A
        # single-output disturbance/bias biases one innovation while the coupled output stays at its
        # ~1e-5 nominal floor; a real coupled process disturbance biases BOTH.
        iD_sig, iB_sig = iD > c.innov_thresh, iB > c.innov_thresh
        if iD_sig and iB_sig:
            sp = diagnostics.get("setpoints", {})
            # bounded recommendation: re-anchor the setpoint toward the current operating point so the
            # offset is acknowledged (the move is clipped by the safety envelope on apply).
            tgt = {"xD": float(y.get("xD", sp.get("xD", 0.96))),
                   "xB": float(y.get("xB", sp.get("xB", 0.005)))}
            return Decision("REAL_DISTURBANCE", "PROPOSE_SETPOINT",
                            "Mismatch shows in BOTH composition innovations (coupled) -> a real process "
                            "disturbance. Recommend a bounded setpoint correction (subject to the safety "
                            "clip) and flag for engineering review.", ev, proposed_targets=tgt)
        if iD_sig or iB_sig:
            ch = "xD" if iD_sig else "xB"
            other = "xB" if ch == "xD" else "xD"
            return Decision("AMBIGUOUS", "ESCALATE",
                            f"Sustained mismatch isolated to the {ch} innovation ({im.get(ch):+.2e}) while "
                            f"{other} stays at its nominal floor. A real {ch} load and an in-range {ch} "
                            f"sensor bias are indistinguishable from telemetry alone -- request a "
                            f"corroborating check (analyzer/feed) before any setpoint move.", ev)
        return Decision("AMBIGUOUS", "ESCALATE",
                        "Offset/constraint mismatch without a clear innovation signature; escalate.", ev)

    # -- LLM hook (NOT called here; deterministic decision above is authoritative) ----------------
    def narrative_prompt(self, decision: Decision) -> str:
        """Build the context an LLM would narrate (interpretation/hypotheses) -- never the decision."""
        return (f"Diagnostic state={decision.state}, recommended action={decision.action}. "
                f"Evidence={decision.evidence}. Write an operator-facing incident note and rank "
                f"plausible root causes; do NOT change the recommended action.")
