"""A hostable, steppable Wood-Berry + MPC simulation sandbox (no RTO, no LLM).

This is the state an MCP server holds so a client can drive the column and inspect MPC health
end-to-end: reset, advance N minutes, read diagnostics / snapshot, and nudge the setpoint (clipped
through the safety envelope). It mirrors the inner per-minute loop of :class:`RTOMPCLoop` but with
NO economic/RTO layer -- "just the MPC" -- and exposes the same diagnostics the agent tools do.

The sandbox is plain Python (no ``mcp`` dependency) so it is unit-testable on its own; the MCP
server (``mcp_server/mpc_server.py``) is a thin wrapper that binds these methods to MCP tools.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from agentic_mpc.controllers import ClassicalMPC
from agentic_mpc.plants import WoodBerryPlant
from agentic_mpc.safety import BoxSafetyEnvelope
from agentic_mpc.scenarios import SCENARIOS

# scenarios that perturb ONLY the plant (work without an RTO/economics layer). The economic /
# constraint scenarios (R3, R4, R5) drive loop.optimizer and are out of scope for an MPC-only server.
_MPC_ONLY_SCENARIOS = ("R1", "R2", "R6", "R7", "S1")


class MPCSandbox:
    """A live, steppable Wood-Berry column under the classical MPC (the MCP server's held state)."""

    def __init__(self, seed: int = 0, dt: float = 1.0) -> None:
        self.dt = float(dt)
        self.reset(seed=seed)

    # -- lifecycle --------------------------------------------------------------------
    def reset(self, seed: int = 0, scenario: str | None = None) -> dict[str, Any]:
        """(Re)build the plant + MPC at the nominal operating point; optionally arm a scenario."""
        if scenario is not None and scenario not in _MPC_ONLY_SCENARIOS:
            raise ValueError(f"scenario {scenario!r} needs an RTO layer; this server is MPC-only. "
                             f"choose one of {_MPC_ONLY_SCENARIOS} or null.")
        self.seed = int(seed)
        self.plant = WoodBerryPlant(dt=self.dt, seed=self.seed)
        self.mpc = ClassicalMPC(dt=self.dt)
        self.safety = BoxSafetyEnvelope()
        self.scenario_id = scenario
        self._scenario = SCENARIOS[scenario]() if scenario else None
        self.t = 0.0
        st = self.plant.get_state()
        self.y = np.array([st["y"]["xD"], st["y"]["xB"]], dtype=float)
        self.history: dict[str, list] = {k: [] for k in ("t", "R", "S", "xD", "xB",
                                                          "xD_true", "xB_true", "xD_sp", "xB_sp")}
        return {"status": "reset", "seed": self.seed, "scenario": self.scenario_id,
                "t": self.t, "targets": dict(self.mpc.targets)}

    def advance(self, minutes: float) -> dict[str, Any]:
        """Step the closed loop forward ``minutes`` (MPC tracks; plant responds; scenario perturbs)."""
        n = max(1, int(round(minutes / self.dt)))
        for _ in range(n):
            t = self.t
            if self._scenario is not None:
                self._scenario.on_step(self, t, self.y)   # sandbox acts as the "loop" (has .plant)
            y_sp = self.mpc.target_vector()
            u = self.mpc.compute_control(self.y, y_sp, t=t)
            yt = self.plant.last_true_output()
            self.history["t"].append(t)
            self.history["R"].append(float(u[0])); self.history["S"].append(float(u[1]))
            self.history["xD"].append(float(self.y[0])); self.history["xB"].append(float(self.y[1]))
            self.history["xD_true"].append(float(yt[0])); self.history["xB_true"].append(float(yt[1]))
            self.history["xD_sp"].append(float(y_sp[0])); self.history["xB_sp"].append(float(y_sp[1]))
            self.y = self.plant.step(u, self.dt)
            self.t += self.dt
        return {"status": "advanced", "minutes": n * self.dt, "t": self.t,
                "y": {"xD": float(self.y[0]), "xB": float(self.y[1])}}

    # -- reads ------------------------------------------------------------------------
    def get_mpc_diagnostics(self) -> dict[str, Any]:
        """Innovation stats, ISE, active constraints, and the steady-state offset (y - y_sp)."""
        h = self.mpc.get_health()
        sp = self.mpc.targets
        offset = {"xD": float(self.y[0] - sp["xD"]), "xB": float(self.y[1] - sp["xB"])}
        return {"t": self.t, **h, "steady_state_offset": offset, "setpoints": dict(sp)}

    def get_plant_snapshot(self) -> dict[str, Any]:
        """Latest measurements + manipulated vars + the rolling history window (a simulation)."""
        return {**self.plant.get_state(), "is_simulation": True}

    # -- the one (bounded) write ------------------------------------------------------
    def set_target(self, xD: float, xB: float, rationale: str) -> dict[str, Any]:
        """Project a proposed (xD, xB) setpoint through the safety box, then apply to the MPC."""
        proposed = {"targets": {"xD": float(xD), "xB": float(xB)}}
        safe, clipped = self.safety.project(proposed)
        targets = safe["targets"]
        self.mpc.set_targets(targets, rationale)
        return {"status": "applied", "applied_targets": targets, "clipped_by_safety": clipped,
                "rationale": rationale}

    def info(self) -> dict[str, Any]:
        return {"plant": "WoodBerry 2x2 FOPDT", "controller": "ClassicalMPC (condensed QP, no integral)",
                "dt_min": self.dt, "seed": self.seed, "scenario": self.scenario_id,
                "t": self.t, "rto": "none (MPC-only server)",
                "available_scenarios": list(_MPC_ONLY_SCENARIOS),
                "safety_box": self.safety.target_bounds}
