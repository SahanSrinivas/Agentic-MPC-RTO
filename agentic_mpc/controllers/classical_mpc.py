"""Classical linear MPC for the 2x2 Wood-Berry column (Phase 1 concrete Controller).

Adapted from the prior-KIRA mixture-MPC at
``operating-region-aware-agentic-cstr-control/phase_1/src/mixture_mpc.py``. What is
**reused** from that code: the condensed-QP formulation (decision variable is the
control sequence; cost assembled as ``J(V) = V^T H V + 2 f^T V``), the scipy-SLSQP
solve with box + move-rate constraints, receding-horizon warm starting, the
``solve_discrete_are`` usage, and the ``control(...) -> dict``-style health reporting.
What is **dropped**: the CSTR-specific 5-model mixture bank, the unstable-saddle
handling, and the per-model Kalman *bank* -- Wood-Berry is a single linear 2x2 plant,
so it needs ONE linear model and ONE Kalman filter, generalized from SISO to MIMO.

Single source of truth for the nominal model (Step-4 spec 1)
------------------------------------------------------------
The MPC's internal model is built from the SAME :class:`WoodBerryParams` instance the
plant uses for its unperturbed nominal state. The plant may perturb *its* copy at
runtime (:meth:`WoodBerryPlant.set_disturbance`); the controller's model is frozen at
the canonical nominal values for the controller's entire life. The divergence that
opens up under a plant disturbance is exactly the model-plant mismatch the supervisory
agent is meant to detect.

Internal model structure
-------------------------
Each scalar transfer function g_ij(s) = K_ij e^{-theta_ij s}/(tau_ij s + 1) is realized
as a ``d_ij``-step input shift register (the transport delay) feeding a first-order ZOH
lag, identical to the plant's per-channel construction -- so with nominal parameters the
model reproduces the plant's nominal dynamics exactly. The blocks are stacked into one
augmented discrete state-space (A, B, C) in deviation coordinates (input = u - u_nominal,
output = y - y_nominal).

State estimation & innovation (Step-4 spec 2)
---------------------------------------------
A steady-state Kalman filter estimates the augmented state from the 2 measured outputs.
Its one-step-ahead prediction is ``y_pred = y_nominal + C x_pred``, where
``x_pred = A x_est + B v`` was formed last step from the input just applied. The
*innovation* reported by :meth:`get_health` is ``y_measured - y_pred`` -- i.e. measured
minus the model's one-step-ahead prediction, NOT tracking error. Under no mismatch it is
zero-mean with std ~ the sensor noise; under a plant gain change it develops a bias.

Tracking is via a steady-state target ``u_target = Kdc^{-1} (y_sp - y_nominal)`` (so the
input penalty R regularizes toward the offset-free target). There is no integral /
output-disturbance state by design: under a plant gain change the loop therefore shows a
genuine steady-state offset (degradation), which is the signal the agent acts on.

Time handling (Step-4 spec 3)
-----------------------------
The controller keeps an internal step counter as its primary clock (the ISE and
innovation windows are sized in samples off it). The optional ``t`` argument to
:meth:`compute_control` is recorded for log timestamping when supplied but is not needed
by the solve. Hard input/move constraints are enforced inside the QP, and there is no
integrator, so no anti-windup logic is required.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import Bounds, LinearConstraint, minimize

from agentic_mpc.interfaces import Controller
from agentic_mpc.metrics import ise as _ise
from agentic_mpc.plants.wood_berry import WoodBerryParams


@dataclass(frozen=True)
class MPCConfig:
    """Tuning for the classical Wood-Berry MPC (Step-4 specification defaults)."""

    horizon_N: int = 30                 # prediction horizon [steps / min]
    control_horizon_M: int = 5          # control (move) horizon [steps / min]
    Q: tuple = (1.0, 1.0)               # output tracking weights diag, order [xD, xB]
    R: tuple = (0.1, 0.1)               # input penalty (toward target) diag, [R, S]
    S_du: tuple = (0.5, 0.5)            # input-move (Delta u) penalty diag, [R, S]
    u_min: tuple = (0.5, 0.5)           # hard lower input bounds [R, S]  [lb/min]
    u_max: tuple = (3.0, 3.0)           # hard upper input bounds [R, S]  [lb/min]
    du_max: float = 0.5                 # hard per-step |Delta u| limit  [lb/min]
    assumed_noise_std: float = 2e-4     # controller's ASSUMED sensor-noise std (KF R)
    kf_process_var: float = 1e-6        # KF process-noise variance (Q = this * I)
    ise_window_min: float = 5.0         # rolling ISE window [min]  (Step-4 reports ISE/5min)
    innovation_window_min: float = 20.0  # rolling innovation-stats window [min]
    qp_maxiter: int = 200


class ClassicalMPC(Controller):
    """Condensed-QP linear MPC for the Wood-Berry column. See module docstring."""

    def __init__(self, params: WoodBerryParams | None = None, dt: float = 1.0,
                 config: MPCConfig | None = None) -> None:
        self.params = params if params is not None else WoodBerryParams()
        self.dt = float(dt)
        self.config = config if config is not None else MPCConfig()
        cfg = self.config

        # --- build the augmented nominal internal model (deviation coords) ---
        self._A, self._B, self._C, self._n_x = _build_augmented_model(self.params, self.dt)
        self._n_u, self._n_y = 2, 2
        # DC gain Kdc = C (I - A)^-1 B  should equal the FOPDT gain matrix (sanity-tested).
        self._Kdc = self._C @ np.linalg.solve(np.eye(self._n_x) - self._A, self._B)
        self._Kdc_inv = np.linalg.inv(self._Kdc)

        # --- steady-state Kalman gain (current-estimator form) ---
        Qkf = cfg.kf_process_var * np.eye(self._n_x)
        Rkf = (max(cfg.assumed_noise_std, 1e-6) ** 2) * np.eye(self._n_y)
        P = solve_discrete_are(self._A.T, self._C.T, Qkf, Rkf)
        self._L = P @ self._C.T @ np.linalg.inv(self._C @ P @ self._C.T + Rkf)
        # ------------------------------------------------------------------------------
        # TODO (Phase 2+, per project decision 2026-05-27): add an OUTPUT-DISTURBANCE-MODEL
        # variant as a second, stronger baseline (offset-free / "industrial-grade" MPC), so
        # the paper can show the agent adds value even against an MPC that rejects the
        # disturbance on its own. Where the disturbance state goes:
        #   * augment the model with n_d=2 integrating output-disturbance states:
        #       x_aug = [x; d],  A_aug = [[A, 0], [0, I_nd]],  B_aug = [[B], [0]],
        #       C_aug = [C, I_nd]   (constant OUTPUT disturbance d feeding y directly);
        #   * extend this Kalman gain (self._L) to the augmented (A_aug, C_aug) so the
        #     filter estimates d_hat from the innovation;
        #   * in compute_control, subtract C_d @ d_hat from the tracking reference and the
        #     target calc (see the marked line there) so steady-state offset -> 0 under
        #     mismatch.
        # The current NO-integral-action design is intentional for the Phase-1 PRIMARY
        # baseline: under a plant gain change it shows a genuine steady-state offset, which
        # is the degradation signal the supervisory agent acts on.
        # ------------------------------------------------------------------------------

        # --- frozen condensed-prediction + QP matrices ---
        self._build_prediction_matrices()
        self._build_qp_weights()

        self._md = {
            "input_names": ["R", "S"],
            "output_names": ["xD", "xB"],
            "dt": self.dt,
            "horizon": cfg.horizon_N,
            "control_horizon": cfg.control_horizon_M,
        }
        # mutable hard limits (set_constraints may update these)
        self._u_min = np.array(cfg.u_min, dtype=float)
        self._u_max = np.array(cfg.u_max, dtype=float)
        self._du_max = float(cfg.du_max)

        self.targets = {"xD": float(self.params.y_nominal[0]),
                        "xB": float(self.params.y_nominal[1])}
        self.reset()

    # -- Controller interface ---------------------------------------------------------
    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._md)

    def reset(self) -> None:
        self._x_est = np.zeros(self._n_x)          # KF state estimate (deviation)
        self._x_pred = np.zeros(self._n_x)          # one-step-ahead predicted state
        self._v_last = np.zeros(self._n_u)          # last applied input deviation
        self._V_warm: np.ndarray | None = None      # warm-start for the QP
        self._k = 0                                 # internal step counter
        w_inn = max(1, round(self.config.innovation_window_min / self.dt))
        w_ise = max(1, round(self.config.ise_window_min / self.dt))
        self._innov_buf: deque = deque(maxlen=w_inn)   # rows: innovation (n_y,)
        self._err_buf: deque = deque(maxlen=w_ise)     # rows: tracking error (n_y,)
        self._active_constraints: list[str] = []

    def compute_control(self, y: np.ndarray, y_sp: np.ndarray,
                        t: float | None = None) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(self._n_y)
        y_sp = np.asarray(y_sp, dtype=float).reshape(self._n_y)
        y_dev = y - self.params.y_nominal
        y_sp_dev = y_sp - self.params.y_nominal

        # 1) Innovation = measured - one-step-ahead prediction (spec 2), then KF correct.
        y_pred_dev = self._C @ self._x_pred
        innovation = y_dev - y_pred_dev                      # == y - (y_nom + C x_pred)
        self._x_est = self._x_pred + self._L @ innovation
        self._innov_buf.append(innovation.copy())
        self._err_buf.append((y - y_sp).copy())

        # 2) Steady-state target for the current setpoint (offset-free reference).
        #    TODO (output-disturbance variant): subtract estimated output disturbance here,
        #    i.e. v_target = Kdc^-1 (y_sp_dev - Cd @ d_hat), for offset-free tracking under
        #    mismatch. See the matching TODO at the Kalman-gain construction in __init__.
        v_target = self._Kdc_inv @ y_sp_dev

        # 3) Solve the condensed QP from the corrected state estimate.
        V = self._solve_qp(self._x_est, y_sp_dev, v_target, self._v_last)
        v0 = V[: self._n_u]
        u = self.params.u_nominal + v0
        u = np.clip(u, self._u_min, self._u_max)             # belt-and-suspenders
        v_applied = u - self.params.u_nominal

        # 4) Record active constraints, then propagate the model one step (predict).
        self._active_constraints = self._detect_active_constraints(V, self._v_last)
        self._x_pred = self._A @ self._x_est + self._B @ v_applied
        self._v_last = v_applied
        self._k += 1
        self._t_last = t
        return u

    def set_targets(self, targets: dict, rationale: str) -> None:
        """Update controlled-variable setpoints (the agent's supervisory action).

        Partial dicts update only the named outputs; the closed-loop driver reads
        :attr:`targets` / :meth:`target_vector` and passes it as ``y_sp``.
        """
        for name, val in targets.items():
            if name not in self.targets:
                raise KeyError(f"unknown controlled variable {name!r}; "
                               f"expected one of {list(self.targets)}")
            self.targets[name] = float(val)
        self._log("set_targets", {"targets": dict(self.targets), "rationale": rationale})

    def set_constraints(self, constraints: dict, rationale: str) -> None:
        """Update hard input bounds / move-rate (not used by the Phase-1 agent tools).

        Recognized keys: ``R_min, R_max, S_min, S_max, du_max``.
        """
        idx = {"R": 0, "S": 1}
        for key, val in constraints.items():
            if key == "du_max":
                self._du_max = float(val)
            elif key.endswith("_min") and key[:-4] in idx:
                self._u_min[idx[key[:-4]]] = float(val)
            elif key.endswith("_max") and key[:-4] in idx:
                self._u_max[idx[key[:-4]]] = float(val)
            else:
                raise KeyError(f"unrecognized constraint {key!r}")
        self._log("set_constraints", {"constraints": dict(constraints), "rationale": rationale})

    def get_health(self) -> dict[str, Any]:
        """Innovation stats (measured - one-step-ahead prediction), active constraints,
        and rolling ISE -- the summary the agent reads via ``get_mpc_health``."""
        on = self._md["output_names"]
        if self._innov_buf:
            innov = np.array(self._innov_buf)                 # (w, n_y)
            mean = innov.mean(axis=0)
            std = innov.std(axis=0)
        else:
            mean = std = np.zeros(self._n_y)
        ise_total = _ise(np.array(self._err_buf), self.dt) if self._err_buf else 0.0
        ise_by = ({n: _ise(np.array(self._err_buf)[:, k], self.dt)
                   for k, n in enumerate(on)} if self._err_buf else {n: 0.0 for n in on})
        return {
            "innovation_mean": {n: float(mean[k]) for k, n in enumerate(on)},
            "innovation_std": {n: float(std[k]) for k, n in enumerate(on)},
            "active_constraints": list(self._active_constraints),
            "ise_recent": float(ise_total),
            "ise_recent_by_output": ise_by,
            "innovation_window_samples": len(self._innov_buf),
        }

    # -- convenience for the closed-loop driver / agent -------------------------------
    def target_vector(self) -> np.ndarray:
        on = self._md["output_names"]
        return np.array([self.targets[n] for n in on], dtype=float)

    # -- QP internals -----------------------------------------------------------------
    def _build_prediction_matrices(self) -> None:
        """Frozen condensed prediction over horizon N with control blocking at M.

        Predicted output deviations  Y = Sx x0 + Phi V , where V = [v_0..v_{M-1}] are the
        free moves (held constant after M). Y stacks y_dev(1..N).
        """
        A, B, C = self._A, self._B, self._C
        N, M = self.config.horizon_N, self.config.control_horizon_M
        ny, nu, nx = self._n_y, self._n_u, self._n_x

        Apow = [np.eye(nx)]
        for _ in range(N):
            Apow.append(Apow[-1] @ A)
        Sx = np.zeros((ny * N, nx))
        Su = np.zeros((ny * N, nu * N))
        for k in range(1, N + 1):                            # predicted step k
            Sx[ny * (k - 1):ny * k, :] = C @ Apow[k]
            for j in range(k):                               # input at step j (0..k-1)
                Su[ny * (k - 1):ny * k, nu * j:nu * (j + 1)] = C @ Apow[k - 1 - j] @ B
        # blocking: full input sequence (N) = T @ V (M moves), last move held.
        T = np.zeros((nu * N, nu * M))
        for k in range(N):
            m = min(k, M - 1)
            T[nu * k:nu * (k + 1), nu * m:nu * (m + 1)] = np.eye(nu)
        self._Sx, self._Phi = Sx, Su @ T

    def _build_qp_weights(self) -> None:
        cfg = self.config
        N, M, ny, nu = cfg.horizon_N, cfg.control_horizon_M, self._n_y, self._n_u
        Qbar = np.kron(np.eye(N), np.diag(cfg.Q))             # (ny N)
        Rbar = np.kron(np.eye(M), np.diag(cfg.R))             # (nu M)
        Sbar = np.kron(np.eye(M), np.diag(cfg.S_du))          # (nu M)
        # move operator on V: Delta = Dm V - e_move, e_move = [v_last, 0, ...].
        Dm = np.eye(nu * M)
        for k in range(1, M):
            Dm[nu * k:nu * (k + 1), nu * (k - 1):nu * k] = -np.eye(nu)
        self._Qbar, self._Rbar, self._Sbar, self._Dm = Qbar, Rbar, Sbar, Dm
        self._H = self._Phi.T @ Qbar @ self._Phi + Rbar + Dm.T @ Sbar @ Dm
        self._H = 0.5 * (self._H + self._H.T)                 # symmetrize

    def _solve_qp(self, x0: np.ndarray, y_sp_dev: np.ndarray, v_target: np.ndarray,
                  v_last: np.ndarray) -> np.ndarray:
        cfg = self.config
        N, M, nu = cfg.horizon_N, cfg.control_horizon_M, self._n_u
        Ysp = np.tile(y_sp_dev, N)
        Vtar = np.tile(v_target, M)
        e_move = np.zeros(nu * M); e_move[:nu] = v_last
        c = self._Sx @ x0 - Ysp
        f = self._Phi.T @ self._Qbar @ c - self._Rbar @ Vtar - self._Dm.T @ self._Sbar @ e_move
        H = self._H

        def fun(V):
            return float(V @ H @ V + 2.0 * f @ V)

        def jac(V):
            return 2.0 * (H @ V + f)

        # box bounds on each move's resulting input: u = u_nom + v in [u_min, u_max].
        lo = np.tile(self._u_min - self.params.u_nominal, M)
        hi = np.tile(self._u_max - self.params.u_nominal, M)
        bounds = Bounds(lo, hi)
        # move-rate: -du_max <= Dm V - e_move <= du_max.
        lc = LinearConstraint(self._Dm, e_move - self._du_max, e_move + self._du_max)
        V0 = self._V_warm if self._V_warm is not None else Vtar.copy()
        V0 = np.clip(V0, lo, hi)
        res = minimize(fun, V0, jac=jac, method="SLSQP", bounds=bounds,
                       constraints=[lc], options=dict(maxiter=cfg.qp_maxiter, ftol=1e-12))
        V = np.clip(res.x, lo, hi)
        self._V_warm = np.r_[V[nu:], V[-nu:]]                 # receding-horizon shift
        return V

    def _detect_active_constraints(self, V: np.ndarray, v_last: np.ndarray,
                                   tol: float = 1e-5) -> list[str]:
        """Constraints active anywhere in the solved horizon (deduped, by name)."""
        names = self._md["input_names"]
        nu, M = self._n_u, self.config.control_horizon_M
        active: list[str] = []
        Vm = V.reshape(M, nu)
        for k in range(M):
            u_k = self.params.u_nominal + Vm[k]
            for j in range(nu):
                if u_k[j] >= self._u_max[j] - tol and f"{names[j]}_upper" not in active:
                    active.append(f"{names[j]}_upper")
                if u_k[j] <= self._u_min[j] + tol and f"{names[j]}_lower" not in active:
                    active.append(f"{names[j]}_lower")
            v_prev = v_last if k == 0 else Vm[k - 1]
            du = Vm[k] - v_prev
            for j in range(nu):
                if abs(du[j]) >= self._du_max - tol and f"d{names[j]}_max" not in active:
                    active.append(f"d{names[j]}_max")
        return active

    def _log(self, event: str, payload: dict) -> None:
        # Lightweight in-memory event log (the experiments persist their own JSON).
        if not hasattr(self, "_event_log"):
            self._event_log: list[dict] = []
        self._event_log.append({"k": getattr(self, "_k", 0), "event": event, **payload})


def _build_augmented_model(params: WoodBerryParams, dt: float):
    """Build the augmented discrete (A, B, C) for the 2x2 FOPDT model in deviation coords.

    Per channel (i, j): a ``d_ij``-step input shift register feeds a first-order ZOH lag
    ``x[k+1] = a x[k] + b * v_j[k - d_ij]`` (a = exp(-dt/tau), b = K (1 - a)); each output
    is the sum of its two lag states. Identical realization to the plant's per-channel
    construction, so the nominal model matches the plant's nominal dynamics exactly.
    """
    a = np.exp(-dt / params.tau)
    b = params.gain * (1.0 - a)
    d = np.rint(params.delay_min / dt).astype(int)
    n_x = int((d + 1).sum())
    A = np.zeros((n_x, n_x)); B = np.zeros((n_x, 2)); C = np.zeros((2, n_x))
    cur = 0
    for i in range(2):
        for j in range(2):
            dij = int(d[i, j])
            if dij > 0:
                ds = list(range(cur, cur + dij))     # delay states w_1..w_d
                lag = cur + dij
                B[ds[0], j] = 1.0                    # w_1 <- v_j
                for m in range(1, dij):
                    A[ds[m], ds[m - 1]] = 1.0        # w_{m+1} <- w_m
                A[lag, lag] = a[i, j]
                A[lag, ds[-1]] = b[i, j]             # lag fed by w_d = v_j[k-d]
            else:
                lag = cur
                A[lag, lag] = a[i, j]
                B[lag, j] = b[i, j]
            C[i, lag] = 1.0
            cur += dij + 1
    return A, B, C, n_x
