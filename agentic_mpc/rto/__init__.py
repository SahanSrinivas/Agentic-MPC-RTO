"""Phase 1.5 real-time-optimization (RTO) layer for the Wood-Berry column.

An economic optimizer sits above the classical MPC: it picks the economically-optimal
composition setpoints (xD, xB) and hands them to the controller. Two comparators share one
solver path (CasADi + IPOPT): a classical nominal NLP (no adaptation) and modifier
adaptation / MA-GP (which closes model-plant mismatch using plant measurements).
"""
from agentic_mpc.rto.economics import WoodBerryEconomics, WoodBerryEconParams
from agentic_mpc.rto.modifier_adaptation import (MAGaussianProcess, ModifierAdaptation,
                                                 SteadyStateDetector)
from agentic_mpc.rto.rto_mpc_loop import RTOMPCLoop
from agentic_mpc.rto.wood_berry_rto import WoodBerryRTO

__all__ = ["WoodBerryEconomics", "WoodBerryEconParams", "WoodBerryRTO",
           "ModifierAdaptation", "MAGaussianProcess", "SteadyStateDetector", "RTOMPCLoop"]
