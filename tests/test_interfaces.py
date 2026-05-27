"""Tests for the universal interface ABCs (Step 2 deliverable).

Verifies (a) the ABCs cannot be instantiated directly, (b) a partial subclass that
omits a required method is still abstract, and (c) a fully conforming subclass works
and behaves to the documented contract. These three classes are the agent's contract,
so the tests pin the contract down.
"""
from __future__ import annotations

import numpy as np
import pytest

from agentic_mpc.interfaces import Controller, Plant, SafetyEnvelope


# --------------------------------------------------------------------------------------
# Minimal conforming stubs -- prove each contract is implementable in isolation.
# --------------------------------------------------------------------------------------
class _DummyPlant(Plant):
    @property
    def metadata(self) -> dict:
        return {
            "input_names": ["u1"], "output_names": ["y1"],
            "input_units": ["-"], "output_units": ["-"],
            "dt": 1.0, "time_units": "min",
            "history_window_samples": 30, "history_window_duration": 30.0,
        }

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        return np.asarray(u, dtype=float)

    def get_state(self) -> dict:
        return {"t": 0.0, "y": {"y1": 0.0}, "u": {"u1": 0.0},
                "history": {"t": [], "y": {"y1": []}, "u": {"u1": []}}}

    def reset(self, initial_condition: dict | None = None) -> None:
        pass


class _DummyController(Controller):
    @property
    def metadata(self) -> dict:
        return {"input_names": ["u1"], "output_names": ["y1"],
                "dt": 1.0, "horizon": 10, "control_horizon": 3}

    def compute_control(self, y: np.ndarray, y_sp: np.ndarray,
                        t: float | None = None) -> np.ndarray:
        return np.zeros_like(np.asarray(y_sp, dtype=float))

    def set_targets(self, targets: dict, rationale: str) -> None:
        pass

    def set_constraints(self, constraints: dict, rationale: str) -> None:
        pass

    def get_health(self) -> dict:
        return {"innovation_mean": 0.0, "innovation_std": 0.0,
                "active_constraints": [], "ise_recent": 0.0}


class _DummyEnvelope(SafetyEnvelope):
    def project(self, proposed_action: dict) -> tuple[dict, bool]:
        return proposed_action, False


# A deliberately incomplete subclass (missing get_state, reset) to prove the ABC bites.
class _PartialPlant(Plant):
    @property
    def metadata(self) -> dict:
        return {}

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        return np.asarray(u, dtype=float)


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
def test_abcs_cannot_be_instantiated_directly():
    for cls in (Plant, Controller, SafetyEnvelope):
        with pytest.raises(TypeError):
            cls()  # type: ignore[abstract]


def test_partial_subclass_is_still_abstract():
    with pytest.raises(TypeError):
        _PartialPlant()  # type: ignore[abstract]


def test_conforming_subclasses_instantiate():
    assert isinstance(_DummyPlant(), Plant)
    assert isinstance(_DummyController(), Controller)
    assert isinstance(_DummyEnvelope(), SafetyEnvelope)


def test_plant_contract_shapes():
    plant = _DummyPlant()
    md = plant.metadata
    assert md["input_names"] and md["output_names"]
    assert len(md["input_names"]) == len(md["input_units"])
    assert len(md["output_names"]) == len(md["output_units"])
    # history window is discoverable up front, in both samples and absolute time
    assert md["history_window_duration"] == md["history_window_samples"] * md["dt"]

    y = plant.step(np.array([1.5]), dt=md["dt"])
    assert isinstance(y, np.ndarray) and y.shape == (1,)

    state = plant.get_state()
    assert {"t", "y", "u", "history"} <= set(state)


def test_controller_contract_shapes():
    ctrl = _DummyController()
    md = ctrl.metadata
    assert {"input_names", "output_names", "dt", "horizon", "control_horizon"} <= set(md)

    # t is optional: callable both with and without a clock
    u = ctrl.compute_control(y=np.array([0.1]), y_sp=np.array([0.0]))
    assert isinstance(u, np.ndarray) and u.shape == (1,)
    u_t = ctrl.compute_control(np.array([0.1]), np.array([0.0]), t=42.0)
    assert isinstance(u_t, np.ndarray) and u_t.shape == (1,)

    health = ctrl.get_health()
    assert {"innovation_mean", "innovation_std", "active_constraints", "ise_recent"} <= set(health)


def test_safety_envelope_contract():
    env = _DummyEnvelope()
    action = {"targets": {"xD": 0.96}}
    safe, violated = env.project(action)
    assert isinstance(safe, dict)
    assert isinstance(violated, bool)
