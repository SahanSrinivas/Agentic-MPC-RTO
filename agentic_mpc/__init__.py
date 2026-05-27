"""Agentic-MPC: Phase 1 vertical slice (Wood-Berry distillation column).

A classical linear MPC controls a simulated Wood-Berry column; an LLM supervisory
agent observes the closed loop and acts through the universal interfaces in
:mod:`agentic_mpc.interfaces`. The agent talks only to the ABCs, so the same agent
generalizes to other plants/controllers in later phases.
"""
from agentic_mpc.interfaces import Controller, Optimizer, Plant, SafetyEnvelope

__all__ = ["Plant", "Controller", "SafetyEnvelope", "Optimizer"]
__version__ = "0.1.0"
