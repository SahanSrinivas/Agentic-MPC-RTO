"""Scenario 1 (LITE) -- diagnostics-GATED supervisory economics over the offset-prone MPC.

This is deliberately thin (one paper subsection, not a product): a static economic objective and a
grid optimizer over the safety envelope, **gated by Scenario 2**. The agent proposes an economic
setpoint ONLY when the diagnostic supervisor reports a clean (NOMINAL/green) state; if Scenario 2
flags anything (sensor fault -> veto, coupled disturbance, or single-channel ambiguity -> escalate),
the economic move is BLOCKED and Scenario 2 governs. This realizes the "Scenario 2 vetoes
Scenario 1" interlock without any RTO, feed model, or LLM-in-the-loop control.

Explicit non-goals (kept out on purpose): real prices, an RTO cycle, an LLM choosing setpoints, any
R/S write. The objective is linear and the steady state is the nominal Kdc map, so the optimum sits
on the envelope (a vertex) -- that is fine for the gate demo; richer economics is future work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentic_mpc.scenario2_agent import DiagnosticSupervisor

# nominal Wood-Berry steady-state gains and operating point (deviation map y = y_nom + K (u - u_nom)).
_K = np.array([[12.8, -18.9], [6.6, -19.4]])
_KINV = np.linalg.inv(_K)
_Y_NOM = np.array([0.96, 0.005])
_U_NOM = np.array([1.95, 1.71])
# safety-envelope box (matches BoxSafetyEnvelope): the HARD clip on any setpoint.
_BOX = {"xD": (0.90, 0.99), "xB": (0.001, 0.05)}
# economic OPERATING band: strictly inside the hard envelope so the optimum keeps control headroom
# from the physical limit (running at xD=0.99 lets a disturbance + tracking push the measurement
# past 1.0 on this unclamped linear plant -- standard practice is to optimize with margin to spec).
_ECON_BOX = {"xD": (0.90, 0.975), "xB": (0.005, 0.05)}


@dataclass(frozen=True)
class EconWeights:
    """Illustrative static economic weights for J = w_D*xD - w_B*xB - w_S*S (config, not real prices)."""
    w_D: float = 1.0      # value of overhead purity
    w_B: float = 1.0      # penalty on bottoms impurity
    w_S: float = 0.3      # steam (reboiler-energy) cost on the steady-state S


def steady_state_inputs(xD_sp: float, xB_sp: float) -> tuple[float, float]:
    """Steady inputs (R, S) the MPC settles at for a setpoint, via the nominal Kdc map."""
    u = _U_NOM + _KINV @ (np.array([xD_sp, xB_sp]) - _Y_NOM)
    return float(u[0]), float(u[1])


def margin(xD: float, xB: float, S: float, w: EconWeights) -> float:
    """Static economic objective J (higher is better)."""
    return w.w_D * xD - w.w_B * xB - w.w_S * S


class Scenario1Agent:
    """Diagnostics-gated economic supervisor: optimize setpoints ONLY when Scenario 2 is green."""

    def __init__(self, weights: EconWeights | None = None,
                 supervisor: DiagnosticSupervisor | None = None) -> None:
        self.w = weights or EconWeights()
        self.s2 = supervisor or DiagnosticSupervisor()

    def reset(self) -> None:
        self.s2.reset()

    # -- the (config) economic optimizer: grid over the envelope, score J at the Kdc steady state ---
    def optimize(self) -> dict:
        best = None
        for xD in np.round(np.arange(_ECON_BOX["xD"][0], _ECON_BOX["xD"][1] + 1e-9, 0.005), 4):
            for xB in np.round(np.arange(_ECON_BOX["xB"][0], _ECON_BOX["xB"][1] + 1e-9, 0.002), 4):
                _, S = steady_state_inputs(float(xD), float(xB))
                J = margin(float(xD), float(xB), S, self.w)
                if best is None or J > best["J"]:
                    best = {"xD": float(xD), "xB": float(xB), "S": S, "J": J}
        return best

    # -- one supervisory cycle: gate, then (if green) propose+apply an economic setpoint ------------
    def step(self, diagnostics: dict, snapshot: dict, sandbox=None) -> dict:
        dec = self.s2.assess(diagnostics, snapshot)
        if dec.state != "NOMINAL":
            return {"gated": True, "s2_state": dec.state, "s2_action": dec.action,
                    "rationale": f"Economics blocked: diagnostics not green ({dec.state}). "
                                 f"Scenario 2 governs: {dec.action}.", "applied": None,
                    "proposed": None}
        opt = self.optimize()
        targets = {"xD": opt["xD"], "xB": opt["xB"]}
        applied = None
        if sandbox is not None:
            applied = sandbox.set_target(rationale=f"Economic optimum (J={opt['J']:.4f}); diagnostics "
                                         f"green.", **targets)
        return {"gated": False, "s2_state": dec.state, "s2_action": "OPTIMIZE",
                "rationale": f"Diagnostics green -> economic optimization. Target J={opt['J']:.4f}.",
                "proposed": targets, "predicted_J": opt["J"], "applied": applied}

    def narrative_prompt(self, result: dict) -> str:
        """LLM hook (off by default): a one-paragraph tradeoff explanation, never the decision."""
        return (f"Economic supervisory result: {result}. Write a one-paragraph operator-facing "
                f"explanation of the tradeoff (purity value vs steam cost) or of why the move was "
                f"gated; do NOT change the decision.")
