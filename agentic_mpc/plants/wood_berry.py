"""Wood-Berry binary distillation column -- a 2x2 first-order-plus-deadtime (FOPDT) plant.

This is the concrete :class:`~agentic_mpc.interfaces.Plant` for Phase 1. It simulates
the classic Wood & Berry (1973) pilot methanol-water column: two manipulated inputs
(reflux flow ``R`` and steam flow ``S``, lb/min) drive two measured outputs (overhead
composition ``xD`` and bottoms composition ``xB``, mole fraction).

Model (deviation variables; absolute output = nominal + deviation + sensor noise)
---------------------------------------------------------------------------------
    [ xD(s) ]   [ 12.8 e^{-1 s}/(16.7 s + 1)   -18.9 e^{-3 s}/(21.0 s + 1) ] [ R(s) ]
    [        ] = [                                                          ] [      ]
    [ xB(s) ]   [  6.6 e^{-7 s}/(10.9 s + 1)   -19.4 e^{-3 s}/(14.4 s + 1) ] [ S(s) ]

with time in MINUTES. Each of the four scalar transfer functions g_ij(s) =
K_ij e^{-theta_ij s} / (tau_ij s + 1) is realized as an independent first-order ZOH
state plus an explicit input transport delay (per-channel input ring buffer), and the
two contributions to each output are summed. Sampling time dt = 1 min (standard for
Wood-Berry).

Sources for the transfer-function and operating-point values (>= 2 independent):
  [1] Wood, R.K. & Berry, M.W. (1973). "Terminal composition control of a binary
      distillation column." Chemical Engineering Science 28(9), 1707-1717.
      doi:10.1016/0009-2509(73)80025-9  -- the original model.
  [2] Seborg, Edgar, Mellichamp & Doyle, "Process Dynamics and Control" (Wiley) --
      the canonical textbook benchmark statement of the same matrix.
  Both report gains/time-constants/delays K = [[12.8, -18.9], [6.6, -19.4]],
  tau = [[16.7, 21.0], [10.9, 14.4]] min, theta = [[1, 3], [7, 3]] min, and nominal
  flows R0 = 1.95, S0 = 1.71 lb/min. Nominal compositions xD0 ~ 0.96 (96 mol%) and
  xB0 ~ 0.005 (~0.5 mol%, the canonical bottoms spec). Because the model is a
  *deviation* model, the absolute operating point (xD0, xB0) only sets the output
  reference/label; it does not affect the dynamics.

Model-plant mismatch hook
-------------------------
The plant exposes a single disturbance-injection point, :meth:`set_disturbance`,
which can scale the channel gains (multiplicative) and/or add an output bias. It is
the seam through which later phases inject the three scenarios (feed-composition
disturbance, economic shift, actuator stiction). For Phase 1 only the hook exists; a
gain change on the R->xD channel is exercised in the Step-7 end-to-end scenario. The
controller's internal model stays at the *nominal* parameters above, so any
disturbance here opens a model-plant gap the supervisory agent must detect.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agentic_mpc.interfaces import Plant

# Output index convention: 0 = xD (overhead), 1 = xB (bottoms).
# Input  index convention: 0 = R  (reflux),   1 = S  (steam).
_OUT_XD, _OUT_XB = 0, 1
_IN_R, _IN_S = 0, 1


@dataclass(frozen=True)
class WoodBerryParams:
    """Nominal Wood & Berry (1973) parameters. See module docstring for citations.

    All matrices are indexed ``[output, input]`` with output order ``[xD, xB]`` and
    input order ``[R, S]``.
    """

    gain: np.ndarray = field(            # K_ij  [(mole frac)/(lb/min)]
        default_factory=lambda: np.array([[12.8, -18.9],
                                          [6.6, -19.4]], dtype=float))
    tau: np.ndarray = field(             # tau_ij  [min]  (time constants)
        default_factory=lambda: np.array([[16.7, 21.0],
                                          [10.9, 14.4]], dtype=float))
    delay_min: np.ndarray = field(       # theta_ij  [min]  (transport delays)
        default_factory=lambda: np.array([[1.0, 3.0],
                                          [7.0, 3.0]], dtype=float))
    u_nominal: np.ndarray = field(       # [R0, S0]  [lb/min]
        default_factory=lambda: np.array([1.95, 1.71], dtype=float))
    y_nominal: np.ndarray = field(       # [xD0, xB0]  [mole fraction]
        default_factory=lambda: np.array([0.96, 0.005], dtype=float))


class WoodBerryPlant(Plant):
    """Stateful Wood-Berry column simulator (Phase 1 concrete :class:`Plant`).

    Parameters
    ----------
    params : WoodBerryParams, optional
        Nominal model parameters (defaults to the canonical values).
    dt : float, default 1.0
        Sampling / control interval in minutes. The transport delays are quantized to
        whole samples at construction, so :meth:`step` must be called with this same
        ``dt`` (a clear error is raised otherwise).
    meas_noise_std : float, default 2e-4
        Std-dev of the zero-mean Gaussian sensor noise added to *each* output
        (chosen with the canonical xB0 ~ 0.005 to keep ~25:1 SNR).
    history_window : int, default 30
        Length, in samples, of the rolling window returned by :meth:`get_state`
        (30 samples = 30 min at dt = 1). Configurable, never hardcoded downstream.
    seed : int | None, default 0
        Seed for the sensor-noise RNG, for reproducible runs.
    """

    def __init__(self, params: WoodBerryParams | None = None, dt: float = 1.0,
                 meas_noise_std: float = 2e-4, history_window: int = 30,
                 seed: int | None = 0) -> None:
        self.params = params if params is not None else WoodBerryParams()
        self.dt = float(dt)
        self.meas_noise_std = float(meas_noise_std)
        self.history_window = int(history_window)
        self._seed = seed

        p = self.params
        # First-order ZOH coefficients per channel (exact for a first-order lag):
        #   x_ij[k+1] = a_ij x_ij[k] + b_ij * du_j[k - d_ij],   a_ij = exp(-dt/tau_ij).
        # a depends only on tau (and dt); b carries the gain and is recomputed from the
        # *effective* (possibly disturbed) gain each step, so disturbances are cheap.
        self._a = np.exp(-self.dt / p.tau)              # (2, 2)
        self._one_minus_a = 1.0 - self._a               # (2, 2)
        self._delay_steps = np.rint(p.delay_min / self.dt).astype(int)  # (2, 2)
        self._max_delay = int(self._delay_steps.max())

        self._n_out, self._n_in = p.gain.shape
        self._md = {
            "input_names": ["R", "S"],
            "output_names": ["xD", "xB"],
            "input_units": ["lb/min", "lb/min"],
            "output_units": ["mole fraction", "mole fraction"],
            "dt": self.dt,
            "time_units": "min",
            "history_window_samples": self.history_window,
            "history_window_duration": self.history_window * self.dt,
        }
        self.reset()

    # -- Plant interface --------------------------------------------------------------
    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._md)  # shallow copy: callers must not mutate our metadata

    def reset(self, initial_condition: dict | None = None) -> None:
        """Reset to the nominal operating point (deviation states = 0).

        Accepted ``initial_condition`` keys (all optional):
          * ``"t0"`` (float): initial simulation time (default 0.0).
          * ``"gain_multiplier"`` / ``"output_bias"``: see :meth:`set_disturbance`.
        ``None`` resets to a clean nominal state with no active disturbance.
        """
        ic = initial_condition or {}
        self._x = np.zeros((self._n_out, self._n_in), dtype=float)  # per-channel FO states
        # Per-INPUT deviation ring buffer (newest at the right end); width = max_delay+1
        # so a lag-d read is buffer[-1 - d] and the largest lag maps to index 0.
        self._du_buf = [deque([0.0] * (self._max_delay + 1), maxlen=self._max_delay + 1)
                        for _ in range(self._n_in)]
        self.t = float(ic.get("t0", 0.0))
        self._rng = np.random.default_rng(self._seed)

        # Disturbance state (the model-plant-mismatch hook).
        self._gain_mult = np.ones((self._n_out, self._n_in), dtype=float)
        self._output_bias = np.zeros(self._n_out, dtype=float)
        # Sensor-only bias (an analyzer gross error): corrupts the MEASUREMENT, not the true
        # state. Distinct from output_bias (a real load disturbance). Used by scenario R6.
        self._sensor_bias = np.zeros(self._n_out, dtype=float)
        if "gain_multiplier" in ic or "output_bias" in ic:
            self.set_disturbance(gain_multiplier=ic.get("gain_multiplier"),
                                 output_bias=ic.get("output_bias"))

        # Rolling history (self-describing via timestamps); seed with the t0 sample.
        self._u_last = self.params.u_nominal.copy()
        self._hist_t: deque = deque(maxlen=self.history_window)
        self._hist_y: dict[str, deque] = {n: deque(maxlen=self.history_window)
                                          for n in self._md["output_names"]}
        self._hist_u: dict[str, deque] = {n: deque(maxlen=self.history_window)
                                          for n in self._md["input_names"]}
        y0 = self._measure(self.params.y_nominal.copy())
        self._y_last = y0
        self._record(self.t, y0, self._u_last)

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Advance one control interval under ZOH input ``u``; return the measured y.

        ``u`` is the absolute input vector ``[R, S]`` in lb/min, in metadata order.
        """
        if abs(dt - self.dt) > 1e-9:
            raise ValueError(
                f"WoodBerryPlant was built with dt={self.dt} min and quantizes its "
                f"transport delays to that grid; step(dt={dt}) is unsupported. "
                f"Build a new plant with the desired dt.")
        u = np.asarray(u, dtype=float).reshape(self._n_in)
        du = u - self.params.u_nominal                      # deviation inputs

        # Push current deviation inputs as the newest buffer entries.
        for j in range(self._n_in):
            self._du_buf[j].append(float(du[j]))

        # Advance each first-order channel using its delayed input, then sum to outputs.
        b = self.params.gain * self._gain_mult * self._one_minus_a   # effective b_ij
        x_next = np.empty_like(self._x)
        for i in range(self._n_out):
            for j in range(self._n_in):
                d = int(self._delay_steps[i, j])
                du_delayed = self._du_buf[j][-1 - d]        # input d steps ago
                x_next[i, j] = self._a[i, j] * self._x[i, j] + b[i, j] * du_delayed
        self._x = x_next

        y_true = self.params.y_nominal + self._x.sum(axis=1) + self._output_bias
        y_meas = self._measure(y_true)

        self.t += self.dt
        self._u_last = u.copy()
        self._y_last = y_meas
        self._record(self.t, y_meas, u)
        return y_meas

    def get_state(self) -> dict[str, Any]:
        """Latest measurements + the rolling, self-describing history window."""
        on, inn = self._md["output_names"], self._md["input_names"]
        return {
            "t": self.t,
            "y": {n: float(self._y_last[k]) for k, n in enumerate(on)},
            "u": {n: float(self._u_last[k]) for k, n in enumerate(inn)},
            "history": {
                "t": list(self._hist_t),
                "y": {n: list(self._hist_y[n]) for n in on},
                "u": {n: list(self._hist_u[n]) for n in inn},
            },
        }

    def steady_state(self, u: np.ndarray, noisy: bool = False) -> np.ndarray:
        """Analytic steady-state measured output for a constant input ``u``.

        For this linear FOPDT plant the transport delays and lags do not affect the steady
        state, so the SS output is ``y_nominal + (K * gain_mult) @ (u - u_nominal) + bias``
        -- including any active disturbance. Used by the RTO comparators (MA / MA-GP) for
        their quasi-steady plant evaluation (deliverable: RTO operates at plant steady state).
        ``noisy=True`` adds one sensor-noise sample.
        """
        u = np.asarray(u, dtype=float).reshape(self._n_in)
        du = u - self.params.u_nominal
        y_true = self.params.y_nominal + (self.params.gain * self._gain_mult) @ du + self._output_bias
        return self._measure(y_true) if noisy else y_true.copy()

    # -- disturbance hook (model-plant-mismatch seam) ---------------------------------
    def set_disturbance(self, gain_multiplier: np.ndarray | dict | None = None,
                        output_bias: np.ndarray | dict | None = None) -> None:
        """Inject / update a plant disturbance (does NOT touch the controller's model).

        Parameters
        ----------
        gain_multiplier : array (2,2) | dict | None
            Multiplicative factor on each channel gain K_ij, indexed [output, input].
            As a dict, keys are ``(output_name, input_name)`` tuples, e.g.
            ``{("xD", "R"): 1.15}`` for a +15% gain on the R->xD channel (Step 7).
            ``None`` leaves the current multiplier unchanged.
        output_bias : array (2,) | dict | None
            Additive bias on each output (an output/load disturbance), e.g. a feed
            composition shift. As a dict, keys are output names. ``None`` leaves it.
        """
        if gain_multiplier is not None:
            self._gain_mult = self._as_io_matrix(gain_multiplier, base=self._gain_mult)
        if output_bias is not None:
            self._output_bias = self._as_out_vector(output_bias, base=self._output_bias)

    def set_sensor_bias(self, sensor_bias: np.ndarray | dict) -> None:
        """Set an analyzer gross-error bias added to the MEASUREMENT only (true state unchanged).

        Keys are output names as a dict, e.g. ``{"xD": 0.05}``; or a length-2 vector. Used by the
        R6 gross-error scenario; reset to zero by passing zeros.
        """
        self._sensor_bias = self._as_out_vector(sensor_bias, base=self._sensor_bias)

    # -- internals --------------------------------------------------------------------
    def _measure(self, y_true: np.ndarray) -> np.ndarray:
        y = y_true + self._sensor_bias                     # analyzer gross error (sensor-only)
        if self.meas_noise_std <= 0.0:
            return y
        return y + self._rng.normal(0.0, self.meas_noise_std, size=self._n_out)

    def _record(self, t: float, y: np.ndarray, u: np.ndarray) -> None:
        self._hist_t.append(float(t))
        for k, n in enumerate(self._md["output_names"]):
            self._hist_y[n].append(float(y[k]))
        for k, n in enumerate(self._md["input_names"]):
            self._hist_u[n].append(float(u[k]))

    def _as_io_matrix(self, val: np.ndarray | dict, base: np.ndarray) -> np.ndarray:
        out = base.copy()
        if isinstance(val, dict):
            on, inn = self._md["output_names"], self._md["input_names"]
            for (o, i), v in val.items():
                out[on.index(o), inn.index(i)] = float(v)
            return out
        return np.asarray(val, dtype=float).reshape(self._n_out, self._n_in)

    def _as_out_vector(self, val: np.ndarray | dict, base: np.ndarray) -> np.ndarray:
        out = base.copy()
        if isinstance(val, dict):
            on = self._md["output_names"]
            for o, v in val.items():
                out[on.index(o)] = float(v)
            return out
        return np.asarray(val, dtype=float).reshape(self._n_out)
