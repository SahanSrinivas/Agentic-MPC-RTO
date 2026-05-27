"""Wood-Berry distillation economics for the Phase-1.5 RTO layer.

Single source of truth for the economic objective, shared by the nominal NLP-RTO and the
MA / MA-GP comparators (numeric form for evaluation, CasADi-symbolic form for IPOPT). The
RTO decision variables are the composition setpoints (xD, xB) handed to the MPC; a column
material balance turns them into distillate/bottoms flows, and the profit trades product
value against separation effort.

Objective (to maximize)
-----------------------
    profit(xD, xB) =  D * (p_D * xD  -  p_pen * (1 - xD))     # purity-premium distillate revenue
                   +  p_B * B * (1 - xB)                       # heavy-bottoms (heavy-key) credit
                   -  c_S * D * sep(xD, xB)                    # reboiler-duty / separation-effort cost
  with the column material balance (constant feed F, light-key feed fraction z_F):
    F = D + B ,   F * z_F = D * xD + B * xB
    =>  D = F (z_F - xB) / (xD - xB) ,  B = F - D            (valid for xB < z_F < xD)
  and the separation-effort proxy
    sep(xD, xB) = ln[ (xD / (1 - xD)) * ((1 - xB) / xB) ]      # Fenske-type minimum-stages / min-reflux

Why the separation-effort term is necessary (documented modeling finding)
-------------------------------------------------------------------------
A purely bilinear profit with the material balance produces a CORNER solution on
Wood-Berry's FOPDT model: the linear gain structure has essentially no cost gradient across
the composition envelope (the manipulated reflux/steam barely move for any realizable
composition change because the steady-state gains are large), and the light-key recovery
D*xD ~ F*z_F is pinned by the material balance -- so nothing penalizes over-purification and
the optimum collapses to the sharpest-separation corner. We augment the objective with a
separation-effort cost based on the ln-separation-factor proxy, which captures the physical
reality that minimum reflux (hence reboiler duty) grows toward infinity as either product
approaches purity. The augmented objective preserves the bilinear-with-material-balance
benchmark structure on the revenue side. References for the ln-separation-factor / minimum-
reflux scaling:
  [Fenske 1932] M.R. Fenske, Ind. Eng. Chem. 24(5):482-485 (minimum stages / separation factor).
  [Skogestad 1997] S. Skogestad, "Dynamics and control of distillation columns -- a tutorial
      introduction," Chem. Eng. Res. Des. 75(6):539-562 (separation factor S = (xD/(1-xD))*
      ((1-xB)/xB); reflux/energy grow as products approach purity).
  [Seborg, Edgar, Mellichamp & Doyle, Process Dynamics and Control (Wiley)] -- Wood-Berry benchmark.

Two further Wood-Berry findings that shape the scenario levers (see scenarios.py):
  * the distillate price p_D is recovery-pinned, so doubling it barely moves the optimum
    (~+0.001 in xD); the effective economic-shift lever is the STEAM/utility cost c_S;
  * profit is homogeneous in the feed F, so scaling F is argmax-invariant; the effective
    demand-shift lever is a distillate-demand constraint D <= D_max.

Nominal parameters (calibration)
--------------------------------
F = 1.0 (normalized feed basis), z_F = 0.5 (a balanced binary feed), p_pen = 2.0, p_B = 0.3
are chosen as round, defensible base values; p_D = 29.3513 and c_S = 0.1306 are then SOLVED
from the two first-order optimality (gradient = 0) conditions so that the unconstrained
interior optimum lands exactly on Phase 1's hardcoded operating point xD = 0.96, xB = 0.005
(which maps through the nominal Wood-Berry gain to R = 1.95, S = 1.71 lb/min). This continuity
-- the RTO's nominal economic answer reproduces the Phase-1 control targets -- is intentional
for the paper narrative. (profit is linear in (p_D, p_pen, p_B, c_S), so the gradient
conditions are two linear equations with a unique (p_D, c_S) solution; verified numerically
against both a scipy prototype and CasADi/IPOPT.)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WoodBerryEconParams:
    """Economic + material-balance parameters. See module docstring for justification."""

    # Material balance
    F: float = 1.0            # feed molar flow [normalized basis]
    z_F: float = 0.5          # feed light-key mole fraction
    # Prices / costs (p_D, c_S calibrated to place the nominal optimum at xD=0.96, xB=0.005)
    p_D: float = 29.3513      # distillate value coefficient
    p_pen: float = 2.0        # off-spec (impurity) penalty on the distillate
    p_B: float = 0.3          # heavy-bottoms product credit
    c_S: float = 0.1306       # separation-effort (reboiler-duty) cost coefficient
    # Operating constraints
    xB_max: float = 0.05      # bottoms impurity spec (RTO inequality; tightened in r5)
    D_max: float | None = None  # distillate demand cap (None = uncapped; set in r4)
    # Physical composition box (the linear Wood-Berry model's validity envelope)
    xD_bounds: tuple = (0.905, 0.994)
    xB_bounds: tuple = (0.0011, 0.049)


class WoodBerryEconomics:
    """Wood-Berry economic objective: material balance + profit (numeric and CasADi-symbolic)."""

    def __init__(self, params: WoodBerryEconParams | None = None) -> None:
        self.params = params if params is not None else WoodBerryEconParams()

    # -- material balance -------------------------------------------------------------
    def flows(self, xD: float, xB: float) -> tuple[float, float]:
        """Distillate D and bottoms B flows from the column material balance (numeric)."""
        p = self.params
        D = p.F * (p.z_F - xB) / (xD - xB)
        return D, p.F - D

    def separation_factor(self, xD: float, xB: float) -> float:
        """ln-separation-factor sep = ln[(xD/(1-xD))((1-xB)/xB)] (numeric)."""
        return float(np.log((xD / (1.0 - xD)) * ((1.0 - xB) / xB)))

    # -- profit -----------------------------------------------------------------------
    def profit(self, xD: float, xB: float) -> float:
        """Instantaneous profit at (xD, xB) [arbitrary $/time], numeric (to maximize)."""
        p = self.params
        D, B = self.flows(xD, xB)
        sep = self.separation_factor(xD, xB)
        return float(
            D * (p.p_D * xD - p.p_pen * (1.0 - xD))
            + p.p_B * B * (1.0 - xB)
            - p.c_S * D * sep
        )

    def profit_symbolic(self, xD: Any, xB: Any) -> Any:
        """Symbolic profit for the CasADi/IPOPT NLP (identical algebra to :meth:`profit`).

        ``xD``/``xB`` are CasADi symbols; uses ``casadi.log`` so IPOPT can differentiate it.
        """
        import casadi as ca  # lazy import (module imports without casadi)

        p = self.params
        D = p.F * (p.z_F - xB) / (xD - xB)
        B = p.F - D
        sep = ca.log((xD / (1.0 - xD)) * ((1.0 - xB) / xB))
        return (
            D * (p.p_D * xD - p.p_pen * (1.0 - xD))
            + p.p_B * B * (1.0 - xB)
            - p.c_S * D * sep
        )

    # -- helpers for the agent's economic-context tool --------------------------------
    def prices(self) -> dict[str, Any]:
        """The current economic coefficients + constraints (for get_economic_context)."""
        p = self.params
        return {
            "F": p.F, "z_F": p.z_F, "p_D": p.p_D, "p_pen": p.p_pen, "p_B": p.p_B,
            "c_S": p.c_S, "xB_max": p.xB_max, "D_max": p.D_max,
        }

    def with_overrides(self, **overrides: Any) -> "WoodBerryEconomics":
        """Return a new economics with some parameters changed (used by scenarios)."""
        return WoodBerryEconomics(replace(self.params, **overrides))
