"""Phase-1.5 disturbance scenarios R1-R7 for the Wood-Berry RTO/MPC stack.

Each scenario is a pure perturbation of the running stack (the plant, the RTO economics, or the
analyzer) -- it contains NO agent or classification logic (that is the agent's job). Each exposes
``on_step(loop, t, y)`` (usable directly as :meth:`RTOMPCLoop.run`'s ``on_step`` hook) and a
``describe()`` with its regime and expected supervisory response. The R1-R7 regimes mirror the
phase_2 Williams-Otto scenarios, recast for a linear distillation column.

Wood-Berry note carried through: a pure multiplicative gain change is ~invisible at the economic
optimum (it sits where the input deviation is 0), so the scenarios that must move the economics or
degrade tracking use a load / output disturbance (a feed-composition bias) or an economic-parameter
change; gain changes appear only as an *equipment-fault mechanism* (R2), paired with the load term
that actually carries the signal.
"""
from __future__ import annotations

from typing import Any


class _Scenario:
    SCENARIO_ID = "R?"
    REGIME = "?"
    MECHANISM = ""
    EXPECTED = ""

    def on_step(self, loop: Any, t: float, y: Any) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"scenario_id": self.SCENARIO_ID, "regime": self.REGIME,
                "mechanism": self.MECHANISM, "expected_supervisory_response": self.EXPECTED}


class R1SlowFeedDrift(_Scenario):
    """R1 -- slow feed-quality drift (analog of catalyst deactivation): a gradual load drift."""

    SCENARIO_ID, REGIME = "R1", "model-mismatch (slow drift)"
    MECHANISM = "feed-composition load drift: xD output bias ramps linearly over the window"
    EXPECTED = "track with a faster RTO cadence (MA/MA-GP follow the drift); nominal RTO lags"

    def __init__(self, t_start: float = 50.0, t_end: float = 250.0,
                 final_xD_bias: float = -0.03) -> None:
        self.t_start, self.t_end, self.final = t_start, t_end, final_xD_bias

    def on_step(self, loop, t, y) -> None:
        if t < self.t_start:
            return
        frac = min(1.0, (t - self.t_start) / max(1e-9, self.t_end - self.t_start))
        loop.plant.set_disturbance(output_bias={"xD": self.final * frac})


class R2EfficiencyLoss(_Scenario):
    """R2 -- abrupt tray-efficiency / fouling loss (analog of heat-transfer degradation)."""

    SCENARIO_ID, REGIME = "R2", "model-mismatch (abrupt)"
    MECHANISM = ("equipment fault: step -10% R->xD gain (the fault) + xD output bias -0.015 "
                 "(the load term that carries the RTO-detectable signal)")
    EXPECTED = "MA/MA-GP correct the load gap; the gain change alone is sub-threshold at the optimum"

    def __init__(self, t_event: float = 100.0, gain_mult: float = 0.90,
                 xD_bias: float = -0.015) -> None:
        self.t_event, self.gain_mult, self.xD_bias, self._fired = t_event, gain_mult, xD_bias, False

    def on_step(self, loop, t, y) -> None:
        if not self._fired and t >= self.t_event:
            loop.plant.set_disturbance(gain_multiplier={("xD", "R"): self.gain_mult},
                                       output_bias={"xD": self.xD_bias})
            self._fired = True


class R3SteamPriceSpike(_Scenario):
    """R3 -- steam / utility price spike (economic shift): a cost coefficient jumps."""

    SCENARIO_ID, REGIME = "R3", "economic-shift"
    MECHANISM = "steam (reboiler-energy) cost c_S multiplied by `factor` at the event"
    EXPECTED = "trigger_rto_run -> RTO re-optimizes to a lower-purity, lower-energy operating point"

    def __init__(self, t_event: float = 100.0, factor: float = 2.0) -> None:
        self.t_event, self.factor, self._fired = t_event, factor, False

    def on_step(self, loop, t, y) -> None:
        if not self._fired and t >= self.t_event:
            loop.optimizer.economics = loop.optimizer.economics.with_overrides(
                c_S=self.factor * loop.optimizer.economics.params.c_S)
            loop.request_rto_recompute()
            self._fired = True


class R4DemandShift(_Scenario):
    """R4 -- product demand shift (economic / constraint): a distillate-demand cap appears."""

    SCENARIO_ID, REGIME = "R4", "economic-shift (constraint-driven)"
    MECHANISM = "distillate-demand cap D <= D_max imposed at the event"
    EXPECTED = "trigger_rto_run -> RTO re-optimizes to a higher-purity (demand-capped) point"

    def __init__(self, t_event: float = 100.0, D_max: float = 0.50) -> None:
        self.t_event, self.D_max, self._fired = t_event, D_max, False

    def on_step(self, loop, t, y) -> None:
        if not self._fired and t >= self.t_event:
            loop.optimizer.economics = loop.optimizer.economics.with_overrides(D_max=self.D_max)
            loop.request_rto_recompute()
            self._fired = True


class R5SpecTightening(_Scenario):
    """R5 -- bottoms-spec tightening below the achievable minimum (RTO infeasibility)."""

    SCENARIO_ID, REGIME = "R5", "constraint-change (infeasibility)"
    MECHANISM = "xB_max tightened below the physical minimum -> RTO returns infeasible"
    EXPECTED = "get_rto_status shows infeasible; agent reports, does NOT command an out-of-spec target"

    def __init__(self, t_event: float = 100.0, xB_max: float = 0.0008) -> None:
        self.t_event, self.xB_max, self._fired = t_event, xB_max, False

    def on_step(self, loop, t, y) -> None:
        if not self._fired and t >= self.t_event:
            loop.optimizer.economics = loop.optimizer.economics.with_overrides(xB_max=self.xB_max)
            loop.request_rto_recompute()
            self._fired = True


class R6AnalyzerGrossError(_Scenario):
    """R6 -- composition-analyzer gross error (data/sensor): a transient measurement bias."""

    SCENARIO_ID, REGIME = "R6", "data / sensor (gross error)"
    MECHANISM = "xD analyzer reads biased by `bias` over a window (sensor-only; true state unchanged)"
    EXPECTED = "ideally recognized as a sensor fault, not a process change; do not over-react"

    def __init__(self, t_start: float = 100.0, t_end: float = 140.0, bias: float = 0.05) -> None:
        self.t_start, self.t_end, self.bias = t_start, t_end, bias
        self._on = False

    def on_step(self, loop, t, y) -> None:
        if not self._on and self.t_start <= t < self.t_end:
            loop.plant.set_sensor_bias({"xD": self.bias}); self._on = True
        elif self._on and t >= self.t_end:
            loop.plant.set_sensor_bias({"xD": 0.0}); self._on = False


class R7LoadDisturbance(_Scenario):
    """R7 -- load disturbance stressing MPC tracking (analog of MPC tracking degradation)."""

    SCENARIO_ID, REGIME = "R7", "mpc-tracking degradation"
    MECHANISM = "step xD output (load) disturbance -0.03 the MPC must reject; offset + ISE rise"
    EXPECTED = "agent sees biased innovation / rising ISE; MA/MA-GP or a target nudge restores it"

    def __init__(self, t_event: float = 100.0, xD_bias: float = -0.03) -> None:
        self.t_event, self.xD_bias, self._fired = t_event, xD_bias, False

    def on_step(self, loop, t, y) -> None:
        if not self._fired and t >= self.t_event:
            loop.plant.set_disturbance(output_bias={"xD": self.xD_bias})
            self._fired = True


SCENARIOS = {
    "R1": R1SlowFeedDrift, "R2": R2EfficiencyLoss, "R3": R3SteamPriceSpike,
    "R4": R4DemandShift, "R5": R5SpecTightening, "R6": R6AnalyzerGrossError,
    "R7": R7LoadDisturbance,
}

__all__ = ["SCENARIOS"] + [c.__name__ for c in SCENARIOS.values()]
