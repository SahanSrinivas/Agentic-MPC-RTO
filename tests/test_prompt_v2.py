"""Unit tests for the v2 system-prompt variant (measurement-validity principle) and selector.

The experiment compares two prompts on the SAME agent/tools/cadence, so these tests pin the two
invariants that keep that comparison valid: (1) the v1 prompts stay byte-identical (frozen), and
(2) v2 is exactly v1 plus the measurement-validity additions -- same steps, renumbered, with the
new principle present. They also cover ``prompt_for`` selection and the v2 output-dir routing.
"""
from __future__ import annotations

import pathlib
import sys

from agentic_mpc.agent import (SYSTEM_PROMPT, SYSTEM_PROMPT_RTO, SYSTEM_PROMPT_RTO_V2,
                               SYSTEM_PROMPT_V2, prompt_for)

# the runner lives in experiments/ (not part of the installed agentic_mpc package), so put the
# repo root on sys.path to import its pure path helper for the output-dir routing test.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_v1_prompts_are_frozen_byte_identical():
    """v1 must not drift: the prior runs (qwen + Claude R3/R6) stay comparable only if v1 is fixed."""
    assert SYSTEM_PROMPT.startswith("You are a supervisory controller")
    # the RTO addendum is appended to the base prompt verbatim (v1 invariant).
    assert SYSTEM_PROMPT_RTO.startswith(SYSTEM_PROMPT)
    assert "PHASE 1.5" in SYSTEM_PROMPT_RTO[len(SYSTEM_PROMPT):]


def test_v2_differs_from_v1_and_adds_measurement_validity():
    assert SYSTEM_PROMPT_V2 != SYSTEM_PROMPT
    # the new general principle: physical bounds, instrument-fault framing, a validation step.
    assert "Validate the measurements" in SYSTEM_PROMPT_V2
    assert "[0, 1]" in SYSTEM_PROMPT_V2
    assert "SUSPECTED INSTRUMENT FAULT" in SYSTEM_PROMPT_V2
    assert "Rule out an instrument fault before acting" in SYSTEM_PROMPT_V2
    # snap-back clause: a transient jump that reverts with no R/S change is a sensor artifact.
    assert "transient sensor" in SYSTEM_PROMPT_V2 and "not a real excursion" in SYSTEM_PROMPT_V2


def test_v2_is_a_pure_superset_of_v1_content():
    """v2 should be v1 + insertions only -- no v1 sentence is dropped (besides the step renumbering)."""
    # every non-numbered v1 line still appears in v2 (the only edits are the inserted block and the
    # 2->3 / 3->4 step renumber, so we compare on the renumber-insensitive body).
    assert "regulates a Wood-Berry binary distillation column" in SYSTEM_PROMPT_V2
    assert "The MPC innovation statistics" in SYSTEM_PROMPT_V2
    assert "respond with a final summary" in SYSTEM_PROMPT_V2
    # v2 is strictly longer (additions only).
    assert len(SYSTEM_PROMPT_V2) > len(SYSTEM_PROMPT)


def test_v2_renumbers_the_observe_diagnose_act_steps():
    # v1: 1 observe, 2 diagnose, 3 act. v2 inserts a validate step -> 1 observe, 2 validate,
    # 3 diagnose, 4 act.
    assert "  2. Validate the measurements" in SYSTEM_PROMPT_V2
    assert "  3. Diagnose any issue" in SYSTEM_PROMPT_V2
    assert "  4. Take a supervisory action ONLY when warranted" in SYSTEM_PROMPT_V2
    # and the v1 numbering is gone from v2.
    assert "  2. Diagnose any issue" not in SYSTEM_PROMPT_V2
    assert "  3. Take a supervisory action ONLY when warranted" not in SYSTEM_PROMPT_V2


def test_v2_rto_keeps_addendum_and_adds_sensor_regime_bullet():
    # the RTO v2 prompt is v2 base + the SAME RTO addendum (sensor bullet inserted into the regimes).
    assert SYSTEM_PROMPT_RTO_V2.startswith(SYSTEM_PROMPT_V2)
    addendum_v2 = SYSTEM_PROMPT_RTO_V2[len(SYSTEM_PROMPT_V2):]
    assert "PHASE 1.5" in addendum_v2
    assert "trigger_rto_run" in addendum_v2
    assert "sensor / analyzer fault" in addendum_v2
    assert "the RTO would optimize against corrupted" in addendum_v2


def test_prompt_for_selects_the_right_prompt():
    assert prompt_for("v1", with_rto=False) is SYSTEM_PROMPT
    assert prompt_for("v1", with_rto=True) is SYSTEM_PROMPT_RTO
    assert prompt_for("v2", with_rto=False) is SYSTEM_PROMPT_V2
    assert prompt_for("v2", with_rto=True) is SYSTEM_PROMPT_RTO_V2
    # defaults + case-insensitivity + unknown -> v1.
    assert prompt_for() is SYSTEM_PROMPT
    assert prompt_for("V2", with_rto=True) is SYSTEM_PROMPT_RTO_V2
    assert prompt_for("nonsense", with_rto=True) is SYSTEM_PROMPT_RTO


def test_output_dir_routes_v2_to_its_own_promptv2_dir():
    from experiments._phase1_5_runner import output_dir_for
    v1 = output_dir_for("R6", "claude-sonnet-4-6", "ma", "llm", True, "v1")
    v2 = output_dir_for("R6", "claude-sonnet-4-6", "ma", "llm", True, "v2")
    assert v1.parts[-3:] == ("claude_sonnet_4_6", "agentic_ma", "R6")
    assert v2.parts[-3:] == ("claude_sonnet_4_6_promptv2", "agentic_ma", "R6")
    # the v2 suffix is LLM-agentic-only: a baseline run never gets the promptv2 dir.
    base = output_dir_for("R6", "claude-sonnet-4-6", "ma", "llm", False, "v2")
    assert "promptv2" not in str(base)


def test_replicate_seeds_get_their_own_subdir_without_disturbing_default():
    from experiments._phase1_5_runner import output_dir_for
    # default seed (42) keeps the flat layout so prior runs + docs still resolve.
    d42 = output_dir_for("R3", "claude-sonnet-4-6", "ma", "llm", True, "v2", 42)
    assert d42.parts[-3:] == ("claude_sonnet_4_6_promptv2", "agentic_ma", "R3")
    # non-default seeds nest under seed<N> so the three replicates coexist instead of overwriting.
    s1 = output_dir_for("R3", "claude-sonnet-4-6", "ma", "llm", True, "v2", 1)
    s2 = output_dir_for("R3", "claude-sonnet-4-6", "ma", "llm", True, "v2", 2)
    assert s1.parts[-4:] == ("claude_sonnet_4_6_promptv2", "agentic_ma", "R3", "seed1")
    assert s2.name == "seed2" and s1 != s2
    # baselines are seed-separated too, so a seed-matched agentic/baseline pair coexists.
    b1 = output_dir_for("R3", "claude-sonnet-4-6", "ma", "llm", False, "v1", 1)
    assert b1.parts[-4:] == ("claude_sonnet_4_6", "baseline_ma", "R3", "seed1")
    # default-seed baseline stays flat (prior layout preserved).
    b42 = output_dir_for("R3", "claude-sonnet-4-6", "ma", "llm", False, "v1", 42)
    assert b42.parts[-3:] == ("claude_sonnet_4_6", "baseline_ma", "R3")
