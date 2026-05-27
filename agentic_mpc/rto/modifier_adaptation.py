"""Modifier-adaptation comparators for the Wood-Berry RTO: MA (Broyden) and MA-GP (BO).

Both close the model-plant economic mismatch the nominal :class:`WoodBerryRTO` cannot: the RTO
operates over the (nominal) Wood-Berry model, while the plant carries a disturbance, so the
model optimum is plant-suboptimal. MA / MA-GP measure the plant and correct.

Decision variables and plant feedback
-------------------------------------
The decision is the commanded composition setpoint u = (xD, xB) the RTO hands to the MPC --
the same decision space as the nominal RTO. (Optimizing in composition space, rather than in
manipulated-input (R, S) space, keeps the objective inside the valid composition box: the
Wood-Berry gains are so large that a modest (R, S) move sends the linear-model compositions far
outside [0, 1], where the ln-separation-factor cost is NaN. Composition-space decisions avoid
that entirely.) The disturbed plant is read at quasi-steady state by mapping the commanded
setpoint through the nominal inverse gain to inputs and calling
:meth:`WoodBerryPlant.steady_state`; profit and the bottoms spec xB are evaluated on the
*realized* compositions. Under a load / output disturbance the realized compositions differ
from the commanded setpoint, opening the gap MA/MA-GP close: they shift the commanded setpoint
so the *realized* operating point reaches the economic optimum.

Wood-Berry caveat (documented): a pure +x% gain perturbation is INVISIBLE at the economic
optimum, because that optimum sits at the nominal compositions where the manipulated-input
deviation is ~0 (so scaling the gain scales nothing -- the nominal RTO is already optimal). The
disturbance that creates a real RTO gap for MA/MA-GP to close is a load / output disturbance
(a feed-composition bias) -- see the standalone validation.

Steady-state detection (Phase-1.5 request 1)
--------------------------------------------
:class:`SteadyStateDetector` makes "the plant has settled" explicit and configurable for the
integrated RTO->MPC->plant loop (``rto_mpc_loop``): settled when ||y(t) - y(t-tau)|| < eps over
a window tau. For Wood-Berry, tau = 5 x the slowest time constant (~5 x 21 = ~105 min) and
eps = 3 sigma (3x the sensor-noise std, ~6e-4), so a settle is distinguishable from noise. (In
the direct/quasi-steady standalone path the plant is already at steady state, so the detector is
exercised by the integrated loop, not the direct comparator iterations.)

Modifier filter gains (Phase-1.5 request 3)
-------------------------------------------
The first-order modifier filter gain K (shared by the zeroth-order eps and first-order lambda
modifiers and by the input step) defaults to 0.5 -- the value used in the phase_2 Williams-Otto
MA (``.../phase_2/src/rto/comparators/ma.py``), a standard MA starting value ([Marchetti 2009]).
It is CSTR-tuned there; for Wood-Berry's slower, delay-dominated dynamics a moderate gain (<=0.5)
remains conservative (smaller steps -> Broyden secants stay locally valid), so 0.5 is reasonable;
it is exposed as a constructor argument for retuning.

MA-GP kernel / prior / acquisition (Phase-1.5 request 4)
--------------------------------------------------------
Ported from the phase_2 MA+GP/BO (del Rio Chanona et al. 2021, deliverable §9.5):
  * GP surrogate of the plant-model gap per quantity (profit, xB): sklearn
    ``GaussianProcessRegressor``, kernel ``Matern(length_scale=0.5, bounds=(1e-2,1e1), nu=2.5)
    + WhiteKernel(noise_level=1e-4, bounds=(1e-6,1e-1))``, ``normalize_y=True``, ``alpha=1e-6``.
    Matern-5/2 (twice-differentiable paths) is a standard mildly-smooth RTO-surface prior; the
    WhiteKernel absorbs measurement noise.
  * inputs scaled to [0,1]^2 over the composition box; Latin-hypercube initial sampling.
  * trust-region BO with merit ratio (dRC21 Alg. 1): eta_1=0.2, eta_2=0.8, gamma_red=0.8,
    gamma_inc=1.2, Delta_0=0.2, Delta_max=1.0.
  * acquisition: Expected Improvement (default) or Lower Confidence Bound (beta=3.0), maximized
    by grid search over the scaled trust region (the GP posterior is not CasADi-differentiable).
These hyperparameters are from the dRC21 CSTR-class case study; reasonable defaults for the
2-input Wood-Berry surface, exposed for retuning.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from agentic_mpc.interfaces import Optimizer
from agentic_mpc.plants.wood_berry import WoodBerryParams
from agentic_mpc.rto.broyden import BroydenGradientEstimator
from agentic_mpc.rto.economics import WoodBerryEconomics

_NAMES = ("profit", "xB")


class SteadyStateDetector:
    """Declares the plant settled when ||y(t) - y(t-tau)|| < eps over a window of tau steps."""

    def __init__(self, tau_steps: int = 105, eps: float = 6e-4,
                 max_wait_steps: int | None = None) -> None:
        self.tau_steps = int(tau_steps)
        self.eps = float(eps)
        self.max_wait_steps = max_wait_steps
        self.reset()

    def reset(self) -> None:
        self._buf: deque = deque(maxlen=self.tau_steps + 1)
        self._n = 0

    def update(self, y: np.ndarray) -> bool:
        """Push a measurement; return True once settled (or max_wait_steps exceeded)."""
        self._buf.append(np.asarray(y, dtype=float))
        self._n += 1
        if self.max_wait_steps is not None and self._n >= self.max_wait_steps:
            return True
        if len(self._buf) <= self.tau_steps:
            return False
        return bool(np.linalg.norm(self._buf[-1] - self._buf[0]) < self.eps)

    @classmethod
    def for_wood_berry(cls, dt: float = 1.0, max_tau_min: float = 21.0, n_tau: float = 5.0,
                       noise_std: float = 2e-4, max_wait_min: float | None = 300.0
                       ) -> "SteadyStateDetector":
        """Default detector: tau = n_tau x slowest-time-constant, eps = 3 x sensor-noise std."""
        return cls(tau_steps=int(round(n_tau * max_tau_min / dt)), eps=3.0 * noise_std,
                   max_wait_steps=None if max_wait_min is None else int(round(max_wait_min / dt)))


class _RTOComparatorBase(Optimizer):
    """Shared plant/model/economics machinery; decision = commanded setpoint sp = (xD, xB)."""

    def __init__(self, economics: WoodBerryEconomics | None, plant: Any,
                 plant_params: WoodBerryParams | None, seed: int | None) -> None:
        self.economics = economics if economics is not None else WoodBerryEconomics()
        self.plant = plant
        self.params = plant_params if plant_params is not None else WoodBerryParams()
        self.Kdc = np.asarray(self.params.gain, dtype=float)
        self._Kdc_inv = np.linalg.inv(self.Kdc)
        self.u_nom = np.asarray(self.params.u_nominal, dtype=float)
        self.y_nom = np.asarray(self.params.y_nominal, dtype=float)
        self._rng = np.random.default_rng(seed)

    def _to_inputs(self, sp: np.ndarray) -> np.ndarray:
        """Map a commanded composition setpoint to manipulated inputs via the nominal gain."""
        return self.u_nom + self._Kdc_inv @ (np.asarray(sp, dtype=float) - self.y_nom)

    def _plant_measure(self, sp: np.ndarray, noisy: bool = True) -> tuple[np.ndarray, float]:
        """Realized compositions + profit when the commanded setpoint is applied to the plant."""
        y = self.plant.steady_state(self._to_inputs(sp), noisy=noisy)
        return y, self.economics.profit(float(y[0]), float(y[1]))

    def _model_profit(self, sp: np.ndarray) -> float:
        """Model-predicted profit (assumes the column reaches the commanded setpoint)."""
        return self.economics.profit(float(sp[0]), float(sp[1]))

    def _box(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.economics.params
        lo = np.array([p.xD_bounds[0], p.xB_bounds[0]])
        hi = np.array([p.xD_bounds[1], min(p.xB_bounds[1], p.xB_max)])
        return lo, hi

    def _result(self, sp: np.ndarray, converged: bool, gap: float | None,
                active: list[str], extra: dict | None = None) -> dict[str, Any]:
        y_real, profit_real = self._plant_measure(sp, noisy=False)
        u = self._to_inputs(sp)
        out = {"setpoints": {"xD": float(sp[0]), "xB": float(sp[1])},
               "realized": {"xD": float(y_real[0]), "xB": float(y_real[1])},
               "inputs": {"R": float(u[0]), "S": float(u[1])},
               "objective": float(profit_real), "converged": bool(converged),
               "active_constraints": active, "model_plant_gap": gap, "type": self.metadata["type"]}
        if extra:
            out.update(extra)
        return out


# ======================================================================================
# Modifier Adaptation (Broyden)
# ======================================================================================
class ModifierAdaptation(_RTOComparatorBase):
    """MA RTO comparator: zeroth/first-order modifiers from measured plant value/gradient gaps."""

    def __init__(self, economics: WoodBerryEconomics | None = None, plant: Any = None,
                 plant_params: WoodBerryParams | None = None, filter_gain: float = 0.5,
                 n_multistart: int = 6, seed: int | None = 0, input_noise_scale: float = 1e-3
                 ) -> None:
        super().__init__(economics, plant, plant_params, seed)
        self.filter_gain = float(filter_gain)
        self.n_multistart = int(n_multistart)
        min_step = 3.0 * float(input_noise_scale)   # conditioning guard (request 2)
        self.grad_profit = BroydenGradientEstimator(2, 1, min_step=min_step)
        self.grad_xB = BroydenGradientEstimator(2, 1, min_step=min_step)
        self.reset()

    @property
    def metadata(self) -> dict[str, Any]:
        return {"setpoint_names": ["xD", "xB"],
                "setpoint_units": ["mole fraction", "mole fraction"], "type": "MA"}

    def reset(self) -> None:
        self.iteration_count = 0
        self.current_sp = self.y_nom.copy()          # start at the nominal economic optimum
        self.eps = {n: 0.0 for n in _NAMES}
        self.lam = {n: np.zeros(2) for n in _NAMES}
        self.grad_profit.reset(); self.grad_xB.reset()
        # Cold-start probe: two deterministic perturbations (one per input) to seed a full-rank
        # Broyden gradient. Without this, starting AT the model optimum gives a zero first step
        # (the lam=0 NLP returns the same point), so the secant -- and the iteration -- never moves.
        self._probe_deltas = [np.array([6e-3, 0.0]), np.array([0.0, 3e-3])]
        self._probe_idx = 0
        self._modifier_started = False
        self.history: list[dict] = []
        self._last: dict[str, Any] = {}

    def solve(self, context: dict | None = None) -> dict[str, Any]:
        self.iterate()
        return self._last

    def get_status(self) -> dict[str, Any]:
        return dict(self._last)

    def run_until_convergence(self, max_iterations: int = 30, tol: float = 1e-4) -> list[dict]:
        small = 0
        for _ in range(max_iterations):
            before = self.current_sp.copy()
            self.iterate()
            small = small + 1 if float(np.linalg.norm(self.current_sp - before)) < tol else 0
            if small >= 2:
                break
        return self.history

    # -- MA core ----------------------------------------------------------------------
    def _model_grad_profit(self, sp: np.ndarray, h: float = 1e-4) -> np.ndarray:
        g = np.zeros(2)
        for i in range(2):
            sp_p = sp.copy(); sp_p[i] += h
            sp_m = sp.copy(); sp_m[i] -= h
            g[i] = (self._model_profit(sp_p) - self._model_profit(sp_m)) / (2 * h)
        return g

    def iterate(self) -> dict[str, Any]:
        k = self.iteration_count + 1
        sp_k = self.current_sp.copy()
        y_p, profit_p = self._plant_measure(sp_k)        # realized profit + xB at this setpoint
        profit_m = self._model_profit(sp_k)

        self.grad_profit.update(sp_k, np.array([profit_p]))
        self.grad_xB.update(sp_k, np.array([y_p[1]]))
        lo, hi = self._box()

        if self._probe_idx < len(self._probe_deltas):
            # PROBE phase: deterministic perturbation to seed the Broyden gradient (no modifiers).
            sp_next = np.clip(sp_k + self._probe_deltas[self._probe_idx], lo, hi)
            self._probe_idx += 1
            phase, converged = "probe", True
        else:
            # MODIFIER phase: eps = plant-model value gap; lam = plant-model gradient gap.
            eps_new = {"profit": profit_p - profit_m, "xB": float(y_p[1] - sp_k[1])}
            lam_new = {"profit": self.grad_profit.B.reshape(2) - self._model_grad_profit(sp_k),
                       "xB": self.grad_xB.B.reshape(2) - np.array([0.0, 1.0])}
            if not self._modifier_started:
                self.eps, self.lam = dict(eps_new), {n: lam_new[n].copy() for n in _NAMES}
                self._modifier_started = True
            else:
                g = self.filter_gain
                self.eps = {n: g * eps_new[n] + (1 - g) * self.eps[n] for n in _NAMES}
                self.lam = {n: g * lam_new[n] + (1 - g) * self.lam[n] for n in _NAMES}
            sol = self._solve_modified(sp_k)
            if sol is not None:
                sp_next = sp_k + self.filter_gain * (sol - sp_k)
                converged = True
            else:
                sp_next, converged = sp_k.copy(), False
            phase = "modifier"

        rec = {"iteration": k, "phase": phase, "setpoint": sp_k.tolist(), "plant_profit": profit_p,
               "model_profit": profit_m, "realized": {"xD": float(y_p[0]), "xB": float(y_p[1])},
               "eps": dict(self.eps), "setpoint_next": sp_next.tolist(), "converged": converged,
               "broyden_skips": self.grad_profit.n_skipped}
        self.history.append(rec)
        self.current_sp = sp_next
        self.iteration_count = k
        active = ["xB_max"] if abs(y_p[1] - self.economics.params.xB_max) < 1e-3 else []
        self._last = self._result(self.current_sp, converged, float(profit_m - profit_p),
                                  active, {"iteration": k})
        return rec

    def _solve_modified(self, sp_k: np.ndarray) -> np.ndarray | None:
        import casadi as ca

        p = self.economics.params
        lo, hi = self._box()
        best = None
        starts = [tuple(sp_k)] + [(float(self._rng.uniform(lo[0], hi[0])),
                                   float(self._rng.uniform(lo[1], hi[1])))
                                  for _ in range(self.n_multistart)]
        for xD0, xB0 in starts:
            opti = ca.Opti(); xD = opti.variable(); xB = opti.variable()
            prof_mod = (self.economics.profit_symbolic(xD, xB) + self.eps["profit"]
                        + self.lam["profit"][0] * (xD - sp_k[0]) + self.lam["profit"][1] * (xB - sp_k[1]))
            opti.minimize(-prof_mod)
            xB_mod = (xB + self.eps["xB"]
                      + self.lam["xB"][0] * (xD - sp_k[0]) + self.lam["xB"][1] * (xB - sp_k[1]))
            opti.subject_to(xB_mod <= p.xB_max)
            opti.subject_to(opti.bounded(p.xD_bounds[0], xD, p.xD_bounds[1]))
            opti.subject_to(opti.bounded(p.xB_bounds[0], xB, p.xB_bounds[1]))
            opti.subject_to(xD - xB >= 0.01)
            if p.D_max is not None:
                opti.subject_to(p.F * (p.z_F - xB) / (xD - xB) <= p.D_max)
            opti.set_initial(xD, min(max(xD0, lo[0]), hi[0]))
            opti.set_initial(xB, min(max(xB0, lo[1]), hi[1]))
            opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes", "tol": 1e-8})
            try:
                sol = opti.solve()
                cand = np.array([float(sol.value(xD)), float(sol.value(xB))])
                val = float(sol.value(prof_mod))
            except Exception:  # noqa: BLE001
                continue
            if best is None or val > best[0]:
                best = (val, cand)
        return None if best is None else best[1]


# ======================================================================================
# MA + GP / Bayesian optimization (del Rio Chanona et al. 2021)
# ======================================================================================
class MAGaussianProcess(_RTOComparatorBase):
    """MA-GP comparator: GP surrogates of the plant-model gap + trust-region Bayesian opt."""

    def __init__(self, economics: WoodBerryEconomics | None = None, plant: Any = None,
                 plant_params: WoodBerryParams | None = None, acquisition: str = "EI",
                 beta_lcb: float = 3.0, eta_1: float = 0.2, eta_2: float = 0.8,
                 gamma_red: float = 0.8, gamma_inc: float = 1.2,
                 initial_trust_radius: float = 0.2, max_trust_radius: float = 1.0,
                 n_initial_samples: int = 6, grid_res: int = 11, seed: int | None = 0) -> None:
        super().__init__(economics, plant, plant_params, seed)
        if acquisition not in ("EI", "LCB"):
            raise ValueError("acquisition must be 'EI' or 'LCB'")
        self.acquisition = acquisition
        self.beta_lcb = float(beta_lcb)
        self.eta_1, self.eta_2 = float(eta_1), float(eta_2)
        self.gamma_red, self.gamma_inc = float(gamma_red), float(gamma_inc)
        self._initial_trust_radius = float(initial_trust_radius)
        self.max_trust_radius = float(max_trust_radius)
        self.n_initial_samples = int(n_initial_samples)
        self.grid_res = int(grid_res)
        self.reset()

    @property
    def metadata(self) -> dict[str, Any]:
        return {"setpoint_names": ["xD", "xB"],
                "setpoint_units": ["mole fraction", "mole fraction"], "type": "MA-GP"}

    def reset(self) -> None:
        self.iteration_count = 0
        self.trust_radius = self._initial_trust_radius
        self._X: list[np.ndarray] = []                 # scaled training inputs
        self._gap: dict[str, list[float]] = {n: [] for n in _NAMES}
        self._gp: dict[str, Any] = {n: None for n in _NAMES}
        self.incumbent_sp: np.ndarray | None = None
        self.incumbent_profit = -np.inf
        self.history: list[dict] = []
        self._last: dict[str, Any] = {}

    # -- scaling over the composition box --------------------------------------------
    def _scale(self, sp: np.ndarray) -> np.ndarray:
        lo, hi = self._box()
        return (np.asarray(sp, dtype=float) - lo) / (hi - lo)

    def _unscale(self, s: np.ndarray) -> np.ndarray:
        lo, hi = self._box()
        return lo + np.asarray(s, dtype=float) * (hi - lo)

    def _fit_gps(self) -> None:
        import warnings as _w

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        X = np.array(self._X)
        for n in _NAMES:
            kernel = (Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e1), nu=2.5)
                      + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1)))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6,
                                          n_restarts_optimizer=1, random_state=0)
            with _w.catch_warnings():
                _w.simplefilter("ignore", category=ConvergenceWarning)
                gp.fit(X, np.array(self._gap[n]))
            self._gp[n] = gp

    def solve(self, context: dict | None = None) -> dict[str, Any]:
        self.iterate()
        return self._last

    def get_status(self) -> dict[str, Any]:
        return dict(self._last)

    def run_until_convergence(self, max_iterations: int = 30, tol: float = 0.02) -> list[dict]:
        for _ in range(max_iterations):
            self.iterate()
            if self.iteration_count > self.n_initial_samples and self.trust_radius < tol:
                break
        return self.history

    def _store(self, sp: np.ndarray) -> tuple[float, float]:
        y_p, profit_p = self._plant_measure(sp)
        self._X.append(self._scale(sp))
        self._gap["profit"].append(profit_p - self._model_profit(sp))
        self._gap["xB"].append(float(y_p[1]) - float(sp[1]))   # model realized xB == sp[1]
        return profit_p, float(y_p[1])

    def iterate(self) -> dict[str, Any]:
        from scipy.stats.qmc import LatinHypercube

        k = self.iteration_count + 1
        p = self.economics.params
        if k <= self.n_initial_samples:                 # global LHS seeding
            s = LatinHypercube(d=2, seed=int(self._rng.integers(1_000_000_000))).random(1)[0]
            sp = self._unscale(s)
            phase = "initial"
        else:
            self._fit_gps()
            sp = self._propose_candidate(p)
            phase = "acquisition"
            if sp is None:
                sp = (self.incumbent_sp.copy() if self.incumbent_sp is not None
                      else self.y_nom.copy())

        profit_p, xB_real = self._store(sp)
        feasible = xB_real <= p.xB_max + 1e-6
        accepted = bool(feasible and profit_p > self.incumbent_profit)
        if accepted:
            self.incumbent_profit, self.incumbent_sp = profit_p, sp.copy()
        if phase == "acquisition":
            self._adapt_trust_region(sp, profit_p, accepted)

        cur = self.incumbent_sp if self.incumbent_sp is not None else sp
        gap = (self._model_profit(cur) - self.incumbent_profit
               if self.incumbent_sp is not None else None)
        rec = {"iteration": k, "phase": phase, "setpoint": sp.tolist(), "plant_profit": profit_p,
               "feasible": feasible, "accepted": accepted, "trust_radius": self.trust_radius,
               "incumbent_profit": self.incumbent_profit}
        self.history.append(rec)
        self.iteration_count = k
        self._last = self._result(np.asarray(cur), self.trust_radius < 0.02, gap,
                                  ([] if feasible else ["xB_max(predicted)"]),
                                  {"iteration": k, "trust_radius": self.trust_radius})
        return rec

    def _propose_candidate(self, p) -> np.ndarray | None:
        center = self._scale(self.incumbent_sp if self.incumbent_sp is not None else self.y_nom)
        lo = np.clip(center - self.trust_radius, 0.0, 1.0)
        hi = np.clip(center + self.trust_radius, 0.0, 1.0)
        g = self.grid_res
        S1, S2 = np.meshgrid(np.linspace(lo[0], hi[0], g), np.linspace(lo[1], hi[1], g))
        Xs = np.column_stack([S1.ravel(), S2.ravel()])
        mu_p, sig_p = self._gp["profit"].predict(Xs, return_std=True)
        mu_b, _ = self._gp["xB"].predict(Xs, return_std=True)
        SP = np.array([self._unscale(s) for s in Xs])
        model_profit = np.array([self._model_profit(sp) for sp in SP])
        corrected_profit = model_profit + mu_p
        pred_xB = SP[:, 1] + mu_b                        # realized xB ~ commanded + gap
        feas = pred_xB <= p.xB_max + 1e-3
        if not feas.any():
            return None
        if self.acquisition == "LCB":
            score = corrected_profit + self.beta_lcb * sig_p
        else:                                            # EI (maximize profit)
            from scipy.stats import norm
            f_inc = self.incumbent_profit if np.isfinite(self.incumbent_profit) else corrected_profit.max()
            imp = corrected_profit - f_inc
            z = np.divide(imp, sig_p, out=np.zeros_like(imp), where=sig_p > 0)
            score = np.where(sig_p > 0, imp * norm.cdf(z) + sig_p * norm.pdf(z), np.maximum(imp, 0.0))
        score = np.where(feas, score, -np.inf)
        return SP[int(np.argmax(score))]

    def _adapt_trust_region(self, sp: np.ndarray, profit_p: float, accepted: bool) -> None:
        pred = self._model_profit(sp) + float(self._gp["profit"].predict(self._scale(sp)[None, :])[0])
        denom = pred - self.incumbent_profit
        rho = (profit_p - self.incumbent_profit) / denom if abs(denom) > 1e-9 else 0.0
        if accepted and rho > self.eta_2:
            self.trust_radius = min(self.trust_radius * self.gamma_inc, self.max_trust_radius)
        elif (not accepted) or rho < self.eta_1:
            self.trust_radius *= self.gamma_red
