"""Concrete :class:`~agentic_mpc.interfaces.SafetyEnvelope` implementations.

The :class:`BoxSafetyEnvelope` is the last-line guard on the supervisory agent's actions:
before a proposed setpoint reaches the controller, it is clipped into a box of physically
acceptable composition targets. This protects against the agent proposing an out-of-spec
or nonphysical setpoint regardless of how it reasoned.
"""
from __future__ import annotations

from agentic_mpc.interfaces import SafetyEnvelope


class BoxSafetyEnvelope(SafetyEnvelope):
    """Clip controlled-variable setpoint targets into per-variable safe boxes.

    Parameters
    ----------
    target_bounds : dict[str, tuple[float, float]]
        Map of controlled-variable name -> ``(low, high)`` allowed setpoint range. For
        Wood-Berry the defaults keep xD a high-purity overhead spec and xB a low-purity
        bottoms spec, both strictly inside the physical [0, 1] mole-fraction range.
    """

    def __init__(self, target_bounds: dict[str, tuple[float, float]] | None = None) -> None:
        self.target_bounds = target_bounds if target_bounds is not None else {
            "xD": (0.90, 0.99),
            "xB": (0.001, 0.05),
        }

    def project(self, proposed_action: dict) -> tuple[dict, bool]:
        """Clip ``proposed_action["targets"]`` into the safe boxes.

        Returns ``(safe_action, was_violated)``; ``was_violated`` is True iff any target
        had to be clipped. Targets for unknown variables are passed through unchanged.
        """
        targets = dict(proposed_action.get("targets", {}))
        violated = False
        for name, val in list(targets.items()):
            if name in self.target_bounds:
                lo, hi = self.target_bounds[name]
                clipped = min(max(float(val), lo), hi)
                if clipped != float(val):
                    violated = True
                targets[name] = clipped
        safe_action = {**proposed_action, "targets": targets}
        return safe_action, violated
