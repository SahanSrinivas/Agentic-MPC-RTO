"""Shared control-performance metrics (ISE / IAE / settling time).

Used by both :meth:`agentic_mpc.controllers.classical_mpc.ClassicalMPC.get_health`
(rolling ISE) and the PRBS / end-to-end experiments, so the definitions live in one
place rather than being reimplemented per script.
"""
from __future__ import annotations

import numpy as np


def ise(errors: np.ndarray, dt: float) -> float:
    """Integral of squared error, ISE = sum(e**2) * dt.

    ``errors`` may be 1-D (one signal) or 2-D (T, n_signals); in the 2-D case the
    squared errors are summed across all signals and time.
    """
    e = np.asarray(errors, dtype=float)
    return float(np.sum(e * e) * dt)


def iae(errors: np.ndarray, dt: float) -> float:
    """Integral of absolute error, IAE = sum(|e|) * dt (same shape rules as :func:`ise`)."""
    e = np.asarray(errors, dtype=float)
    return float(np.sum(np.abs(e)) * dt)


def settling_time(t: np.ndarray, y: np.ndarray, y_target: float, tol: float,
                  t_start: float = 0.0) -> float | None:
    """Time (relative to ``t_start``) for ``y`` to enter and *stay within* ``+/- tol``
    of ``y_target``.

    Returns the elapsed time after ``t_start`` past which ``|y - y_target| <= tol`` for
    the remainder of the trace, or ``None`` if it is still outside the band at the end.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.abs(y - y_target) > tol
    if not mask.any():
        return 0.0
    last_outside = int(np.max(np.flatnonzero(mask)))
    if last_outside >= len(t) - 1:
        return None  # never settles within the trace
    return float(t[last_outside + 1] - t_start)
