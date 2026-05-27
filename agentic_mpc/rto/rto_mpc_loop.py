"""RTO -> MPC -> plant orchestration with explicit dual-rate timing (Phase 1.5).

Layered closed loop (adapted from the phase_2 ``RTOMPCLoop`` structure):

    RTO (slow, every rto_interval_min) commands composition setpoints
        -> MPC (fast, every dt) tracks them
        -> plant responds; a steady-state detector watches the tracking
        -> the next RTO trigger re-optimizes on the (now-settled) plant.

Two design properties (Phase-1.5 design asks):

1. **Explicit, configurable cadence.** The RTO and the MPC run on separate, explicit timers.
   Defaults: MPC every ``dt`` = 1 min (from the controller metadata), RTO every
   ``rto_interval_min`` = 60 min -- the standard process-control separation (an RTO layer
   re-optimizes economics on a tens-of-minutes cadence while the MPC regulates minute-by-minute).
   Scenarios may override ``rto_interval_min`` (e.g. a slow drift wants a faster RTO cadence to
   track) and may force an immediate re-optimization on an event via :meth:`request_rto_recompute`
   (e.g. a price-spike event should recompute now, not wait for the next tick).

2. **Clean handoff capture for the agent.** Every time the RTO commands a setpoint, the loop
   records a handoff with (a) the setpoints commanded, (b) the time, (c) which RTO variant
   produced them (nominal / MA / MA-GP, from ``optimizer.metadata['type']``), and (d) the RTO's
   ``get_status()`` snapshot at command time -- plus whether the plant had settled since the
   previous command. This is exactly what the agent's ``get_rto_status`` tool surfaces, so the
   agent can reason about "the RTO commanded a target T minutes ago; has the plant settled, and
   what did the RTO believe when it did?"
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from agentic_mpc.interfaces import Controller, Optimizer, Plant
from agentic_mpc.rto.modifier_adaptation import SteadyStateDetector


class RTOMPCLoop:
    """Drives RTO (slow) over MPC (fast) over the plant, with explicit timers + handoff log."""

    def __init__(self, plant: Plant, controller: Controller, optimizer: Optimizer,
                 rto_interval_min: float = 60.0, dt: float | None = None,
                 settle_detector: SteadyStateDetector | None = None,
                 trigger_rto_at_start: bool = True) -> None:
        self.plant = plant
        self.controller = controller
        self.optimizer = optimizer
        self.dt = float(dt if dt is not None else controller.metadata["dt"])
        self.rto_interval_min = float(rto_interval_min)
        self.rto_interval_steps = max(1, int(round(self.rto_interval_min / self.dt)))
        self.detector = settle_detector or SteadyStateDetector.for_wood_berry(dt=self.dt)
        self._trigger_rto_at_start = bool(trigger_rto_at_start)
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self._k = 0
        self._pending_rto = False
        self._settled_since_command = False
        self.handoffs: list[dict[str, Any]] = []
        self.last_handoff: dict[str, Any] | None = None
        self.detector.reset()
        self.history: dict[str, list] = {k: [] for k in ("t", "R", "S", "xD", "xB",
                                                          "xD_sp", "xB_sp", "settled")}

    # -- public controls --------------------------------------------------------------
    def request_rto_recompute(self) -> None:
        """Force an RTO re-optimization on the next step (event trigger, e.g. a price change)."""
        self._pending_rto = True

    def note_external_command(self, res: dict[str, Any]) -> None:
        """Record a handoff for an RTO run commanded outside the periodic tick (e.g. by the agent
        via ``trigger_rto_run``), so :meth:`get_rto_status` reflects it."""
        sp = res.get("setpoints", {})
        handoff = {"t": self.t, "setpoints": dict(sp), "rto_type": res.get("type", "?"),
                   "rto_status": self.optimizer.get_status(),
                   "settled_since_prev": self._settled_since_command, "source": "agent"}
        self.handoffs.append(handoff)
        self.last_handoff = handoff
        self.detector.reset()
        self._settled_since_command = False

    def get_rto_status(self) -> dict[str, Any]:
        """The agent-facing handoff snapshot: last RTO command + settle state (or 'no RTO yet')."""
        if self.last_handoff is None:
            return {"rto_has_run": False, "minutes_since_command": None,
                    "settled_since_command": False}
        return {"rto_has_run": True,
                "commanded_setpoints": self.last_handoff["setpoints"],
                "commanded_at_min": self.last_handoff["t"],
                "minutes_since_command": round(self.t - self.last_handoff["t"], 3),
                "rto_variant": self.last_handoff["rto_type"],
                "rto_status_at_command": self.last_handoff["rto_status"],
                "settled_since_command": self._settled_since_command,
                "n_rto_commands": len(self.handoffs)}

    # -- the loop ---------------------------------------------------------------------
    def run(self, t_end: float, on_step: Callable[["RTOMPCLoop", float, np.ndarray], None] | None = None
            ) -> dict[str, Any]:
        """Run the dual-rate loop to ``t_end``.

        ``on_step(loop, t, y)`` is called at the start of every MPC step -- scenarios use it to
        inject disturbances / price changes and to call :meth:`request_rto_recompute`.
        """
        n = int(round(t_end / self.dt))
        y = np.array([self.plant.get_state()["y"]["xD"], self.plant.get_state()["y"]["xB"]])
        for k in range(n + 1):
            self.t = k * self.dt
            self._k = k
            if on_step is not None:
                on_step(self, self.t, y)

            # RTO trigger: at start (optional), on the periodic tick, or on an event request.
            periodic = (k > 0 and k % self.rto_interval_steps == 0)
            at_start = (k == 0 and self._trigger_rto_at_start)
            if at_start or periodic or self._pending_rto:
                self._trigger_rto()
                self._pending_rto = False

            y_sp = self.controller.target_vector()
            u = self.controller.compute_control(y, y_sp, t=self.t)
            self._record(self.t, u, y, y_sp)
            y = self.plant.step(u, self.dt)
            self._settled_since_command = self.detector.update(y)

        return {"history": {k: np.asarray(v) for k, v in self.history.items()},
                "handoffs": self.handoffs}

    # -- internals --------------------------------------------------------------------
    def _trigger_rto(self) -> None:
        res = self.optimizer.solve()
        sp = res.get("setpoints", {})
        if res.get("converged", True) and np.isfinite(sp.get("xD", np.nan)):
            self.controller.set_targets(
                {"xD": float(sp["xD"]), "xB": float(sp["xB"])},
                rationale=f"RTO ({res.get('type', '?')}) economic setpoint at t={self.t:.0f} min")
        handoff = {"t": self.t, "setpoints": dict(sp), "rto_type": res.get("type", "?"),
                   "rto_status": self.optimizer.get_status(),
                   "settled_since_prev": self._settled_since_command}
        self.handoffs.append(handoff)
        self.last_handoff = handoff
        self.detector.reset()                 # restart settle tracking for the new setpoint
        self._settled_since_command = False

    def _record(self, t: float, u: np.ndarray, y: np.ndarray, y_sp: np.ndarray) -> None:
        self.history["t"].append(t)
        self.history["R"].append(float(u[0])); self.history["S"].append(float(u[1]))
        self.history["xD"].append(float(y[0])); self.history["xB"].append(float(y[1]))
        self.history["xD_sp"].append(float(y_sp[0])); self.history["xB_sp"].append(float(y_sp[1]))
        self.history["settled"].append(bool(self._settled_since_command))
