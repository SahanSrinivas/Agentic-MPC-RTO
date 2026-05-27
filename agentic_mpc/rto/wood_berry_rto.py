"""Classical (no-adaptation) NLP-RTO for the Wood-Berry column -- the nominal comparator.

Adapted from the prior-KIRA Williams-Otto RTO
(``operating-region-aware-agentic-cstr-control/phase_2/src/rto/nlp_rto.py``): same
CasADi-Opti + IPOPT + multi-start pattern, generalized to the Wood-Berry economic problem.
What changes: the decision variables are the two composition setpoints (xD, xB) the MPC will
track (vs Williams-Otto's F_B, T_R), the objective is :mod:`agentic_mpc.rto.economics`
(vs the Williams-Otto profit), and the model is linear -- the optimal setpoint maps to the
manipulated inputs (R, S) through the nominal Wood-Berry steady-state gain (same
:class:`WoodBerryParams` the MPC uses, so the RTO and MPC share one nominal model).

Because it optimizes the (nominal) economic model with no plant feedback, this RTO is
plant-suboptimal under disturbance by construction -- exactly the failure mode the
modifier-adaptation comparators (``modifier_adaptation.py``) close. It is the non-adaptive
baseline in the comparison.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from agentic_mpc.interfaces import Optimizer
from agentic_mpc.plants.wood_berry import WoodBerryParams
from agentic_mpc.rto.economics import WoodBerryEconomics


class WoodBerryRTO(Optimizer):
    """Classical NLP-RTO: maximize economic profit over the nominal model (CasADi/IPOPT)."""

    def __init__(self, economics: WoodBerryEconomics | None = None,
                 plant_params: WoodBerryParams | None = None,
                 n_multistart: int = 12, seed: int | None = 0,
                 verbose: bool = False) -> None:
        self.economics = economics if economics is not None else WoodBerryEconomics()
        self.plant_params = plant_params if plant_params is not None else WoodBerryParams()
        self.n_multistart = int(n_multistart)
        self._rng = np.random.default_rng(seed)
        self.verbose = bool(verbose)
        # nominal steady-state gain (single source of truth, shared with the MPC's model)
        self._Kdc = np.asarray(self.plant_params.gain, dtype=float)
        self._u_nom = np.asarray(self.plant_params.u_nominal, dtype=float)
        self._y_nom = np.asarray(self.plant_params.y_nominal, dtype=float)
        self._last_status: dict[str, Any] = {}

    # -- Optimizer interface ----------------------------------------------------------
    @property
    def metadata(self) -> dict[str, Any]:
        return {"setpoint_names": ["xD", "xB"],
                "setpoint_units": ["mole fraction", "mole fraction"],
                "type": "nominal"}

    def solve(self, context: dict | None = None) -> dict[str, Any]:
        """Solve the economic NLP over the nominal model with multi-start.

        ``context`` may carry ``{"economics": WoodBerryEconomics}`` to optimize a different
        economic parameterization (e.g. a scenario's price/constraint override); otherwise the
        RTO's configured economics is used. Returns the optimal setpoints, the mapped inputs,
        objective, convergence, and active constraints.
        """
        econ = (context or {}).get("economics", self.economics)
        result = self._solve_nlp(econ)
        result["type"] = self.metadata["type"]
        self._last_status = result
        return result

    def get_status(self) -> dict[str, Any]:
        """Diagnostics from the last :meth:`solve` (for the get_rto_status tool)."""
        return dict(self._last_status)

    # -- helpers ----------------------------------------------------------------------
    def setpoint_to_inputs(self, xD: float, xB: float) -> tuple[float, float]:
        """Map a composition setpoint to the steady-state (R, S) via the nominal gain."""
        u = self._u_nom + np.linalg.solve(self._Kdc, np.array([xD, xB]) - self._y_nom)
        return float(u[0]), float(u[1])

    # -- internals --------------------------------------------------------------------
    def _solve_once(self, econ: WoodBerryEconomics, xD0: float, xB0: float) -> dict[str, float]:
        import casadi as ca  # lazy import

        p = econ.params
        opti = ca.Opti()
        xD = opti.variable()
        xB = opti.variable()
        opti.minimize(-econ.profit_symbolic(xD, xB))
        # physical composition box (linear-model-validity envelope)
        opti.subject_to(opti.bounded(p.xD_bounds[0], xD, p.xD_bounds[1]))
        opti.subject_to(opti.bounded(p.xB_bounds[0], xB, min(p.xB_bounds[1], p.xB_max)))
        opti.subject_to(xD - xB >= 0.01)        # material-balance validity (xB < z_F < xD)
        D = p.F * (p.z_F - xB) / (xD - xB)
        if p.D_max is not None:                  # distillate-demand cap (r4 lever)
            opti.subject_to(D <= p.D_max)
        opti.set_initial(xD, xD0)
        opti.set_initial(xB, xB0)
        opti.solver("ipopt", {"print_time": self.verbose},
                    {"print_level": 5 if self.verbose else 0, "sb": "yes", "tol": 1e-9})
        sol = opti.solve()                       # raises ca.OptiSolError on failure
        return {"xD": float(sol.value(xD)), "xB": float(sol.value(xB))}

    def _solve_nlp(self, econ: WoodBerryEconomics) -> dict[str, Any]:
        p = econ.params
        xD_lo, xD_hi = p.xD_bounds
        xB_lo, xB_hi = p.xB_bounds[0], min(p.xB_bounds[1], p.xB_max)
        # Empty feasible box (e.g. r5: spec tightened below the achievable minimum) -> infeasible.
        if xB_hi < xB_lo or xD_hi < xD_lo:
            return self._infeasible()
        starts = [(0.96, 0.005), (0.95, 0.01), (0.97, 0.004), (0.93, 0.02), (0.985, 0.0025)]
        # add random multi-starts inside the (possibly tightened) box
        for _ in range(max(0, self.n_multistart - len(starts))):
            starts.append((float(self._rng.uniform(xD_lo, xD_hi)),
                           float(self._rng.uniform(xB_lo, xB_hi))))
        best: dict[str, float] | None = None
        n_ok = 0
        for xD0, xB0 in starts:
            xB0 = min(max(xB0, xB_lo), xB_hi)
            try:
                vals = self._solve_once(econ, xD0, xB0)
            except Exception:  # noqa: BLE001 -- failed/infeasible start is expected; skip
                continue
            prof = econ.profit(vals["xD"], vals["xB"])
            if not np.isfinite(prof):
                continue
            n_ok += 1
            if best is None or prof > best["profit"]:
                best = {**vals, "profit": prof}

        if best is None:                         # all starts failed -> infeasible (e.g. r5)
            return self._infeasible()

        R, S = self.setpoint_to_inputs(best["xD"], best["xB"])
        return {
            "setpoints": {"xD": best["xD"], "xB": best["xB"]},
            "inputs": {"R": R, "S": S},
            "objective": best["profit"],
            "converged": True,
            "active_constraints": self._active_constraints(best, econ),
            "n_successful_starts": n_ok,
            "model_plant_gap": None,             # non-adaptive: gap only known to MA/MA-GP
            "status": "optimal",
        }

    @staticmethod
    def _infeasible() -> dict[str, Any]:
        return {"setpoints": {"xD": float("nan"), "xB": float("nan")},
                "inputs": {"R": float("nan"), "S": float("nan")},
                "objective": float("nan"), "converged": False,
                "active_constraints": [], "n_successful_starts": 0,
                "model_plant_gap": None, "status": "infeasible"}

    def _active_constraints(self, best: dict, econ: WoodBerryEconomics,
                            tol: float = 1e-4) -> list[str]:
        p = econ.params
        xD, xB = best["xD"], best["xB"]
        active: list[str] = []
        if abs(xD - p.xD_bounds[1]) <= tol:
            active.append("xD_upper")
        if abs(xD - p.xD_bounds[0]) <= tol:
            active.append("xD_lower")
        if abs(xB - min(p.xB_bounds[1], p.xB_max)) <= tol:
            active.append("xB_max" if p.xB_max <= p.xB_bounds[1] else "xB_upper")
        if abs(xB - p.xB_bounds[0]) <= tol:
            active.append("xB_lower")
        if p.D_max is not None:
            D = p.F * (p.z_F - xB) / (xD - xB)
            if abs(D - p.D_max) <= tol:
                active.append("D_max")
        return active
