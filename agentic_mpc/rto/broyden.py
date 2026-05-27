"""Broyden plant-gradient estimation for the RTO comparators (adapted from phase_2).

The "good Broyden" rank-1 update maintains an estimate of the plant Jacobian ``B = dy/du``
(measured outputs y w.r.t. RTO inputs u) from successive (u, y) measurements -- the gradient
information modifier adaptation needs but cannot get analytically from the plant.

Ported from ``operating-region-aware-agentic-cstr-control/phase_2/src/rto/comparators/
broyden.py`` (itself re-derived from deliverable §9.4 eq. 15; no code copied from the
unlicensed omega-icl/ma-gp). One addition for Wood-Berry, per the Phase-1.5 request:

  **Step-conditioning guard.** Beyond the numerical zero-step tolerance, the update is also
  skipped when the input step is small relative to the noise scale (``||du|| < min_step``),
  because a secant taken over a step comparable to the measurement noise is dominated by
  noise and would corrupt the Jacobian. Skips are counted (``n_skipped``) and warned, so a
  caller (or the agent) can see that "the RTO sat quiet and learned no new gradient this
  cycle." Recommended ``min_step ~ 3 * input-noise scale``.
"""
from __future__ import annotations

import warnings

import numpy as np

# Squared-step tolerance below which the denominator (du^T du) is treated as a zero step.
_ZERO_STEP_TOL = 1.0e-14


class BroydenGradientEstimator:
    """Rank-1 ("good") Broyden estimator of a plant Jacobian ``dy/du``.

    Maintains ``B`` (shape ``n_outputs x n_inputs``) from consecutive measurement pairs:

        B <- B + outer( dy - B du, du ) / (du^T du),   du = u_k - u_{k-1}, dy = y_k - y_{k-1}

    exact for affine plants after ``n_inputs`` linearly independent steps. The first
    :meth:`update` only stores its pair (a single point carries no gradient information).

    Parameters
    ----------
    min_step : float
        Minimum ``||du||`` for which a rank-1 update is applied (the conditioning guard).
        ``0.0`` keeps only the numerical zero-step tolerance. Set ~3x the input-noise scale
        so noise-dominated secants are skipped rather than incorporated.
    """

    def __init__(self, n_inputs: int, n_outputs: int,
                 initial_B: np.ndarray | None = None, min_step: float = 0.0) -> None:
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.min_step = float(min_step)
        self._initial_B = (np.zeros((self.n_outputs, self.n_inputs), dtype=float)
                           if initial_B is None
                           else np.asarray(initial_B, float).reshape(self.n_outputs, self.n_inputs))
        self.B: np.ndarray = self._initial_B.copy()
        self._u_prev: np.ndarray | None = None
        self._y_prev: np.ndarray | None = None
        self._n_updates = 0
        self.n_skipped = 0          # conditioning-guard skips (for diagnostics / the agent)

    @property
    def has_estimate(self) -> bool:
        """True once at least one rank-1 update has been applied (>= 2 distinct points)."""
        return self._u_prev is not None and self._n_updates > 0

    def update(self, u_new: np.ndarray, y_new: np.ndarray) -> np.ndarray:
        """Apply the Broyden rank-1 update with a new ``(u, y)`` pair.

        First call stores the pair. A (numerically) zero step, or a step below ``min_step``
        (the conditioning guard), is skipped with a warning and counted in ``n_skipped``.
        """
        u = np.asarray(u_new, dtype=float).reshape(self.n_inputs)
        y = np.asarray(y_new, dtype=float).reshape(self.n_outputs)
        if self._u_prev is None:
            self._u_prev, self._y_prev = u, y
            return self.B

        du = u - self._u_prev
        dy = y - self._y_prev
        step = float(np.linalg.norm(du))
        if (du @ du) <= _ZERO_STEP_TOL:
            warnings.warn("Broyden: zero input step; skipping rank-1 update.",
                          RuntimeWarning, stacklevel=2)
            self.n_skipped += 1
            return self.B
        if step < self.min_step:
            warnings.warn(f"Broyden: input step {step:.2e} < min_step {self.min_step:.2e} "
                          f"(noise-dominated); skipping update.", RuntimeWarning, stacklevel=2)
            self.n_skipped += 1
            # advance the reference point but do NOT corrupt B with a noisy secant
            self._u_prev, self._y_prev = u, y
            return self.B

        self.B = self.B + np.outer(dy - self.B @ du, du) / (du @ du)
        self._u_prev, self._y_prev = u, y
        self._n_updates += 1
        return self.B

    def reset(self) -> None:
        """Clear history and restore ``B`` to its initial value (use after a regime change)."""
        self.B = self._initial_B.copy()
        self._u_prev = self._y_prev = None
        self._n_updates = 0
        self.n_skipped = 0
