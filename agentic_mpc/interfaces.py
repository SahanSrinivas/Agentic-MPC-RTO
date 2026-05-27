"""Universal control-stack interfaces for the Agentic-MPC project (Phase 1).

These abstract base classes are *the contract the supervisory agent talks through*.
The agent in :mod:`agentic_mpc.agent.supervisor` must only ever interact with objects
that implement :class:`Plant`, :class:`Controller`, and :class:`SafetyEnvelope` --
never with a concrete plant or controller class directly. This is the
"one algorithm, multiple processes" claim made concrete: swapping the Wood-Berry
column for the Wu CSTR (Phase 2+), or the classical MPC for an RNN-MPC, must require
*no* change to the agent -- only a different object satisfying these same ABCs.

Design conventions (read before implementing a concrete subclass)
-----------------------------------------------------------------
* All numeric signal I/O uses ``numpy`` arrays in a *fixed, documented channel
  order* given by ``metadata`` (e.g. inputs ``["R", "S"]``, outputs ``["xD", "xB"]``).
* Dict-shaped methods (:meth:`Plant.get_state`, :meth:`Controller.get_health`,
  :meth:`Controller.set_targets`) key on human-readable channel *names* so the LLM
  agent's reasoning and the operator log read clearly.
* ``metadata`` is the single source of truth mapping array index <-> channel name,
  units, and sampling time. Concrete classes MUST populate it.
* Units and sampling time are explicit everywhere; this is research code that will
  be read by reviewers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Plant(ABC):
    """A simulated (later, possibly real) process under control.

    The plant owns its true internal state and dynamics. The controller and agent
    observe it ONLY through :meth:`step` (which returns the *measured* output,
    including sensor noise) and :meth:`get_state`. The plant's true parameters are
    deliberately *not* exposed on this interface: model-plant mismatch is something
    the agent must *infer* from measurements, not read off directly.
    """

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Static descriptor of the plant's I/O. Required keys::

            {
              "input_names":  list[str],   # manipulated vars, in array order  e.g. ["R", "S"]
              "output_names": list[str],   # measured vars,    in array order  e.g. ["xD", "xB"]
              "input_units":  list[str],   # parallel to input_names
              "output_units": list[str],   # parallel to output_names
              "dt": float,                 # nominal sampling time [process time units]
              "time_units": str,           # e.g. "min"
              "history_window_samples":  int,   # length of get_state()'s history window, in samples
              "history_window_duration": float, # = history_window_samples * dt, in `time_units`
            }

        The array order in ``input_names`` / ``output_names`` defines the meaning of
        the ``u`` and ``y`` vectors used by :meth:`step` and the controller. The
        ``history_window_*`` pair lets the agent know the temporal span of the window
        from :meth:`get_state` *before* calling it, in both samples and absolute time
        (so a consumer never has to compute samples * dt itself -- this matters when a
        later plant, e.g. the Wu CSTR, runs at a different ``dt``).
        """
        ...

    @abstractmethod
    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Advance the plant one control interval under zero-order-hold input ``u``.

        Parameters
        ----------
        u : np.ndarray, shape (n_inputs,)
            Manipulated-variable vector, in ``metadata["input_names"]`` order, in
            absolute (not deviation) engineering units.
        dt : float
            Sampling / integration interval in ``metadata["time_units"]``. Normally
            equal to ``metadata["dt"]``.

        Returns
        -------
        y : np.ndarray, shape (n_outputs,)
            The *measured* output vector (true output + sensor noise), in
            ``metadata["output_names"]`` order.
        """
        ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return current measurements plus a recent history window.

        The history window MUST be self-describing: it always carries its own
        timestamps (``history["t"]``), and the configured window length is reported
        in ``metadata["history_window_samples"]`` / ``["history_window_duration"]``.
        The agent therefore never has to guess how much history it received -- it can
        read the timestamps for the actual span and ``metadata`` for the capacity.
        During warm-up the window holds fewer than ``history_window_samples`` samples;
        ``len(history["t"])`` is the count actually present.

        Recommended shape (the agent's ``get_process_state`` tool surfaces this)::

            {
              "t": float,                          # current sim time
              "y": dict[str, float],               # latest measured outputs by name
              "u": dict[str, float],               # last applied inputs by name
              "history": {
                  "t": list[float],                # timestamps -> the window is self-describing
                  "y": dict[str, list[float]],     # per-output recent samples
                  "u": dict[str, list[float]],     # per-input recent samples
              },
            }
        """
        ...

    @abstractmethod
    def reset(self, initial_condition: dict | None = None) -> None:
        """Reset plant state. ``None`` -> the nominal operating point.

        ``initial_condition`` may carry concrete-class-specific keys (e.g. an output
        vector, a time offset); concrete classes document what they accept.
        """
        ...


class Controller(ABC):
    """A feedback controller that drives a :class:`Plant` toward setpoints.

    Phase 1's concrete implementation is a classical linear MPC. The agent never
    sees the controller's internals (internal model, QP, weights) -- only the health
    summary from :meth:`get_health` and the supervisory setpoints/constraints it
    pushes via :meth:`set_targets` / :meth:`set_constraints`.

    Model-plant mismatch (the thing the agent exists to manage)
    -----------------------------------------------------------
    The controller holds a *fixed nominal* internal model. For the Wood-Berry case
    that model is the nominal first-order-plus-deadtime parameter set (gains, delays,
    time constants) from Wood & Berry (1973). The *plant's* parameters, by contrast,
    can be perturbed at runtime (e.g. a feed-composition disturbance changing a gain).
    The controller does NOT update its model in response; the resulting divergence
    between predicted and measured output is the model-plant mismatch the supervisory
    agent observes (via innovation statistics in :meth:`get_health`) and acts on.
    """

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Static descriptor of the controller. Required keys::

            {
              "input_names":  list[str],   # manipulated vars, array order
              "output_names": list[str],   # controlled vars,  array order
              "dt": float,                 # control interval [process time units]
              "horizon": int,              # prediction horizon (in steps)
              "control_horizon": int,      # control/move horizon (in steps)
            }
        """
        ...

    @abstractmethod
    def compute_control(self, y: np.ndarray, y_sp: np.ndarray,
                        t: float | None = None) -> np.ndarray:
        """Solve the controller once and return the input to apply this interval.

        Parameters
        ----------
        y : np.ndarray, shape (n_outputs,)
            Latest measured outputs, in ``metadata["output_names"]`` order.
        y_sp : np.ndarray, shape (n_outputs,)
            Output setpoints used for *this* solve, same order. Lets the caller drive
            a setpoint trajectory; supervisory targets pushed via :meth:`set_targets`
            update the controller's default setpoint between calls.
        t : float | None, optional
            Current simulation time, in the plant's ``time_units``. **Optional by
            design.** The QP solve itself does not need a clock (it works in the
            receding-horizon frame), so a controller may ignore ``t``. It is provided
            so a controller that wants to time-stamp its innovation log or anchor its
            ISE window to the shared simulation clock can do so without inventing its
            own counter. When ``None``, a controller should fall back to an internal
            step counter. (This generalizes the prior ``control(y, setpoint, t)``
            signature, where ``t`` was required but in practice unused.)

        Returns
        -------
        u : np.ndarray, shape (n_inputs,)
            Manipulated-variable vector to apply, in ``metadata["input_names"]``
            order, already projected onto the controller's own hard constraints.
        """
        ...

    @abstractmethod
    def set_targets(self, targets: dict, rationale: str) -> None:
        """Update controlled-variable setpoints.

        ``targets`` maps output name -> value, e.g. ``{"xD": 0.96, "xB": 0.005}``;
        partial dicts update only the named outputs. ``rationale`` is an
        operator-readable string logged alongside the change (the agent must supply it).
        """
        ...

    @abstractmethod
    def set_constraints(self, constraints: dict, rationale: str) -> None:
        """Update hard constraints (input bounds, move-rate limits).

        ``constraints`` maps a constraint name -> value; ``rationale`` is logged as
        for :meth:`set_targets`. Part of the universal contract; NOT exercised by the
        Phase-1 agent tool set (the agent there can only retune targets).
        """
        ...

    @abstractmethod
    def get_health(self) -> dict[str, Any]:
        """Return a controller-health summary. Recommended shape::

            {
              "innovation_mean": dict[str, float] | float,  # measured y - predicted y
              "innovation_std":  dict[str, float] | float,
              "active_constraints": list[str],              # constraints hit last solve
              "ise_recent": float,                          # ISE over recent window
            }

        Surfaced to the agent through the ``get_mpc_health`` tool.
        """
        ...


class SafetyEnvelope(ABC):
    """A last-line guard that projects a proposed supervisory action onto the safe set.

    Independent of the controller's own hard constraints: this is the layer that
    protects against the *agent* proposing something unsafe (e.g. an out-of-spec
    setpoint), before it reaches the controller.
    """

    @abstractmethod
    def project(self, proposed_action: dict) -> tuple[dict, bool]:
        """Clip ``proposed_action`` into the safe set.

        Parameters
        ----------
        proposed_action : dict
            The action the agent wants to take, e.g.
            ``{"targets": {"xD": 0.99, "xB": 0.004}}``.

        Returns
        -------
        (safe_action, was_violated) : tuple[dict, bool]
            ``safe_action`` is ``proposed_action`` with every field clipped into the
            safe set (same shape). ``was_violated`` is True iff any field had to be
            changed.
        """
        ...


class Optimizer(ABC):
    """A real-time-optimization (RTO) layer that computes economically-optimal setpoints.

    Sits ABOVE the :class:`Controller`: it chooses the controlled-variable setpoints (for
    Wood-Berry, xD and xB) that maximize an economic objective subject to operating
    constraints, then hands them to the controller via :meth:`Controller.set_targets`. The
    supervisory agent talks to *any* RTO -- a classical nominal NLP, or a mismatch-correcting
    comparator (modifier adaptation / MA-GP) -- through this single contract, just as it talks
    to any plant/controller through theirs (the "one algorithm, multiple processes" claim,
    now extended up the stack to "one agent, multiple optimizers").
    """

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Static descriptor of the RTO. Required keys::

            {
              "setpoint_names": list[str],   # controlled vars the RTO sets, e.g. ["xD", "xB"]
              "setpoint_units": list[str],   # parallel to setpoint_names
              "type": str,                   # "nominal" | "MA" | "MA-GP"
            }
        """
        ...

    @abstractmethod
    def solve(self, context: dict | None = None) -> dict[str, Any]:
        """Compute the current economically-optimal setpoints.

        For a nominal RTO this optimizes the (fixed) economic model. For an adaptive
        comparator (MA / MA-GP) it performs one adaptation step -- measure the plant, update
        the modifiers / surrogate, re-solve the modified problem. ``context`` may carry
        economic overrides (e.g. a price change) or be ``None`` to use the configured economics.

        Returns
        -------
        dict with at least::

            {
              "setpoints": dict[str, float],     # e.g. {"xD": 0.96, "xB": 0.005}
              "objective": float,                # optimized profit
              "converged": bool,
              "active_constraints": list[str],
              "model_plant_gap": float | None,   # economic mismatch; None if non-adaptive/unmeasured
            }
        """
        ...

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Return diagnostics from the last :meth:`solve` (surfaced via the get_rto_status tool)."""
        ...