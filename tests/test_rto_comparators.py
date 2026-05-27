"""Standalone validation for the MA / MA-GP RTO comparators (Phase 1.5, Gate 3).

Verifies: the steady-state detector and Broyden conditioning guard behave; MA and MA-GP both
correct a load (output) disturbance toward the plant optimum; both respect an active xB
constraint; and -- documenting the recurring Wood-Berry finding -- a pure gain perturbation is
invisible at the (gain-invariant) optimum, so MA reduces to the nominal answer there.

Convergence tests use a noise-free plant for deterministic assertions; the comparators converge
to a neighborhood of the optimum under sensor noise (closing the bulk of the disturbance gap).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from agentic_mpc.interfaces import Optimizer
from agentic_mpc.plants import WoodBerryPlant
from agentic_mpc.rto import (MAGaussianProcess, ModifierAdaptation, SteadyStateDetector,
                             WoodBerryEconomics)
from agentic_mpc.rto.broyden import BroydenGradientEstimator

warnings.filterwarnings("ignore")  # silence GP ConvergenceWarning / CasADi NaN-probe warnings


def _plant(noise: float = 0.0, **disturbance):
    p = WoodBerryPlant(dt=1.0, seed=0, meas_noise_std=noise)
    if disturbance:
        p.set_disturbance(**disturbance)
    return p


# --- utilities -------------------------------------------------------------------------
def test_steady_state_detector_settles_and_flags_motion():
    det = SteadyStateDetector(tau_steps=5, eps=1e-3)
    for _ in range(10):                                   # constant signal -> settles
        settled = det.update(np.array([0.96, 0.005]))
    assert settled is True
    det.reset()
    flagged = [det.update(np.array([0.96 + 0.01 * k, 0.005])) for k in range(10)]
    assert flagged[-1] is False                           # steadily moving -> never settled


def test_steady_state_detector_for_wood_berry_defaults():
    det = SteadyStateDetector.for_wood_berry()
    assert det.tau_steps == 105 and abs(det.eps - 6e-4) < 1e-12   # 5*21 min, 3*2e-4


def test_broyden_conditioning_guard_skips_small_steps():
    est = BroydenGradientEstimator(2, 1, min_step=3e-3)
    est.update(np.array([0.96, 0.005]), np.array([1.0]))
    est.update(np.array([0.9605, 0.005]), np.array([1.01]))   # ||du||=5e-4 < 3e-3 -> skipped
    assert est.n_skipped == 1 and not est.has_estimate
    est.update(np.array([0.97, 0.005]), np.array([1.1]))      # ||du||~1e-2 -> applied
    assert est.has_estimate


# --- (a) MA corrects a load (output) disturbance --------------------------------------
def test_ma_corrects_output_disturbance():
    plant = _plant(noise=0.0, output_bias={"xD": -0.03})       # feed-comp load disturbance
    ma = ModifierAdaptation(plant=plant, plant_params=plant.params, seed=0, input_noise_scale=0.0)
    assert isinstance(ma, Optimizer)
    ma.run_until_convergence(max_iterations=40)
    realized = ma.get_status()["realized"]
    # nominal (uncorrected) would sit at xD=0.93; MA must close most of the 0.03 gap toward 0.96
    assert realized["xD"] > 0.952, realized
    assert realized["xD"] < 0.966, realized
    assert abs(realized["xB"] - 0.005) < 2e-3, realized


# --- (b) MA-GP corrects similarly ------------------------------------------------------
def test_magp_corrects_output_disturbance():
    plant = _plant(noise=0.0, output_bias={"xD": -0.03})
    mg = MAGaussianProcess(plant=plant, plant_params=plant.params, seed=1,
                           n_initial_samples=8, grid_res=15)
    assert isinstance(mg, Optimizer)
    mg.run_until_convergence(max_iterations=35)
    realized = mg.get_status()["realized"]
    assert realized["xD"] > 0.945, realized                    # closes the bulk of the gap
    assert abs(realized["xB"] - 0.005) < 2e-3, realized


# --- (c) both respect an active constraint boundary -----------------------------------
@pytest.mark.parametrize("cls,kw", [(ModifierAdaptation, dict(input_noise_scale=0.0)),
                                    (MAGaussianProcess, dict(n_initial_samples=8, grid_res=15))])
def test_comparators_respect_constraint_boundary(cls, kw):
    econ = WoodBerryEconomics().with_overrides(xB_max=0.003)    # tighter than nominal xB=0.005
    plant = _plant(noise=0.0)
    cmp = cls(economics=econ, plant=plant, plant_params=plant.params, seed=0, **kw)
    cmp.run_until_convergence(max_iterations=35)
    realized = cmp.get_status()["realized"]
    assert realized["xB"] <= 0.003 + 5e-4, realized            # no constraint violation
    assert realized["xD"] > 0.95, realized                     # still high-purity overhead


# --- gain-invariance finding (documented) ---------------------------------------------
def test_pure_gain_mismatch_is_invariant_at_optimum():
    """A +15% R->xD gain perturbation leaves the economic optimum at the nominal compositions
    (it sits where the input deviation is 0), so MA reduces to the nominal answer -- the
    methodological finding that load disturbances, not gain faults, drive Wood-Berry RTO."""
    plant = _plant(noise=0.0, gain_multiplier={("xD", "R"): 1.15})
    ma = ModifierAdaptation(plant=plant, plant_params=plant.params, seed=0, input_noise_scale=0.0)
    ma.run_until_convergence(max_iterations=30)
    realized = ma.get_status()["realized"]
    # xD (dominant) stays at the nominal optimum -> the gain fault did not move it; xB stays
    # within MA's convergence neighborhood. (Contrast the load disturbance, which DOES move xD.)
    assert abs(realized["xD"] - 0.96) < 0.01 and abs(realized["xB"] - 0.005) < 4e-3, realized
