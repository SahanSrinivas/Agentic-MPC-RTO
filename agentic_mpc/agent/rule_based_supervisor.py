"""Rule-based supervisory baselines for the Phase-1 paper (no LLM).

Paper purpose
-------------
The paper's claim is "the LLM supervisor adds value over a baseline." A reviewer will ask:
"couldn't a 50-line rule book do the same thing?" These classes ARE that rule book, run on the
same scenarios through the same tools, so the comparison is head-to-head. We provide TWO variants
to defeat the "you picked a weak baseline" objection:

  * :class:`RuleBasedSupervisorNaive` -- on a sustained offset, retargets the MPC to the current
    measurement (accepts the drift; the deliberately simple action).
  * :class:`RuleBasedSupervisorSmart` -- on a sustained offset, re-optimizes via the RTO instead.

Both are pure-Python conditionals: no LLM, no Ollama, no GPU -- they run on a laptop and are fully
deterministic. They are drop-in replacements for :class:`~agentic_mpc.agent.supervisor.SupervisoryAgent`:
same constructor shape, the same ``run_cycle(...) -> {"final","actions","iterations"}`` contract,
the same :class:`AgentContext` and tools, and the same ``actions`` log shape (so ``log.json`` and the
downstream analysis / ``n_agent_actions`` counter work unchanged).

Decision rules and threshold provenance
---------------------------------------
Rules are evaluated in order; at most one action per cycle (mirrors the LLM's one-decision cadence).

  Rule 1 (model-plant mismatch -> trigger RTO):
      abs(innovation_mean.xD) > 5e-4  OR  abs(innovation_mean.xB) > 5e-4
      Provenance: the raw one-step-ahead innovation std is ~2.7e-4 at nominal (~= the 2e-4 sensor
      noise; Phase-1 Step-4 validation), and the *rolling innovation mean* sits at ~1e-5 at nominal
      vs ~1.5e-3 under a load disturbance. 5e-4 (~2.5x the noise std, ~50x the nominal mean) fires
      only on a genuinely biased innovation. abs() is REQUIRED: load disturbances bias the
      innovation *negative* (e.g. -1.5e-3 under R2/R7), which a bare `>` would miss.

  Rule 2 (sustained tracking offset; ONLY when settled):
      rto_has_run AND settled_since_command AND
      (abs(y.xD - commanded.xD) > 5e-3  OR  abs(y.xB - commanded.xB) > 1e-3)
      Provenance: xD +/-5e-3 = 0.5% composition (the BoxSafetyEnvelope operating envelope);
      xB +/-1e-3 = 20% relative of the 0.005 nominal. Gated on `settled_since_command` (the
      steady-state detector) so it captures a SUSTAINED offset, not the transient right after an RTO
      move (when the plant hasn't tracked yet). Action differs by variant (see above).

  Rule 3 (stale RTO + economic change -> trigger RTO):
      rto_has_run AND minutes_since_command > 90 AND prices changed since the last cycle
      Provenance: RTO cadence is 60 min, so >90 min = 1.5x overdue. "Economic change" is detected by
      caching the previous cycle's prices and comparing -- without the cache, Rule 3 has no signal.

  Otherwise: hold (no action).

Invocation (CLI)
----------------
Through the scenario runners:
    python experiments/r3_feed_price_spike.py --supervisor rule-based-naive
    python experiments/r3_feed_price_spike.py --supervisor rule-based-smart
Outputs land under experiments/outputs/phase1_5/rule_based_{naive,smart}_<rto>/<scenario>/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_mpc.agent.tools import AgentContext, make_tool_registry
from agentic_mpc.interfaces import Controller, Optimizer, Plant, SafetyEnvelope


@dataclass(frozen=True)
class RuleConfig:
    """Thresholds for the rule-based supervisor (see module docstring for provenance)."""

    innovation_threshold: float = 5e-4   # |innovation mean| trigger (~2.5x sensor-noise std)
    offset_xD: float = 5e-3              # 0.5% composition (operating envelope)
    offset_xB: float = 1e-3              # 20% relative of the 0.005 nominal
    stale_minutes: float = 90.0          # 1.5x the 60-min RTO cadence


class _RuleBasedSupervisorBase:
    """Shared rule logic; variants differ only in the Rule-2 action (:meth:`_rule2`)."""

    VARIANT = "base"

    def __init__(self, plant: Plant, controller: Controller,
                 safety: SafetyEnvelope | None = None, rto: Optimizer | None = None,
                 rto_loop: Any = None, config: RuleConfig | None = None) -> None:
        self.ctx = AgentContext(plant=plant, controller=controller, safety=safety,
                                rto=rto, rto_loop=rto_loop)
        self.registry = make_tool_registry(self.ctx)
        self.config = config if config is not None else RuleConfig()
        self._prev_prices: dict | None = None   # for Rule 3's economic-change detection

    # -- drop-in interface: same contract as SupervisoryAgent.run_cycle --------------
    def run_cycle(self, user_message: str | None = None, max_iterations: int = 1) -> dict[str, Any]:
        """One rule-based supervisory cycle. Returns the same dict shape as the LLM supervisor."""
        actions: list[dict] = []
        state = self._read(actions, "get_process_state")
        health = self._read(actions, "get_mpc_health")
        rto_status = self._read(actions, "get_rto_status")
        econ = self._read(actions, "get_economic_context")
        cfg = self.config

        innov = health.get("innovation_mean", {}) or {}
        iD, iB = abs(innov.get("xD", 0.0) or 0.0), abs(innov.get("xB", 0.0) or 0.0)
        ran = bool(rto_status.get("rto_has_run"))
        cmd = rto_status.get("commanded_setpoints") or {}
        settled = bool(rto_status.get("settled_since_command"))
        mins = rto_status.get("minutes_since_command") or 0.0

        if iD > cfg.innovation_threshold or iB > cfg.innovation_threshold:
            final = self._act(actions, "trigger_rto_run",
                              {"rationale": f"Rule 1: |innovation| (xD={iD:.2e}, xB={iB:.2e}) "
                               f"exceeded {cfg.innovation_threshold:.1e} (model-plant mismatch)."})
        elif ran and settled and self._offset_exceeded(state, cmd):
            final = self._rule2(actions, state, cmd)
        elif ran and mins > cfg.stale_minutes and self._prices_changed(econ):
            final = self._act(actions, "trigger_rto_run",
                              {"rationale": f"Rule 3: RTO stale ({mins:.0f} min > "
                               f"{cfg.stale_minutes:.0f}) and economics changed; re-optimize."})
        else:
            final = f"No action (rule-based {self.VARIANT}): all thresholds clear."

        self._prev_prices = econ.get("prices")
        return {"final": final, "actions": actions, "iterations": 1}

    # -- helpers ----------------------------------------------------------------------
    def _read(self, actions: list, name: str) -> dict:
        res = self.registry[name]()
        actions.append({"iteration": 0, "tool": name, "status": "executed", "args": {}, "result": res})
        return res

    def _act(self, actions: list, name: str, args: dict) -> str:
        res = self.registry[name](**args)
        actions.append({"iteration": 0, "tool": name, "status": "executed", "args": args, "result": res})
        return f"Action [{self.VARIANT}]: {name} -- {args.get('rationale', '')}"

    def _offset_exceeded(self, state: dict, cmd: dict) -> bool:
        if not cmd:
            return False
        y = state.get("y", {})
        ox = abs(y.get("xD", 0.0) - cmd.get("xD", y.get("xD", 0.0)))
        ob = abs(y.get("xB", 0.0) - cmd.get("xB", y.get("xB", 0.0)))
        return ox > self.config.offset_xD or ob > self.config.offset_xB

    def _prices_changed(self, econ: dict) -> bool:
        cur = econ.get("prices")
        return self._prev_prices is not None and cur is not None and cur != self._prev_prices

    def _rule2(self, actions: list, state: dict, cmd: dict) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class RuleBasedSupervisorNaive(_RuleBasedSupervisorBase):
    """Rule 2 action: retarget the MPC to the current measurement (accept the drift)."""

    VARIANT = "naive"

    def _rule2(self, actions: list, state: dict, cmd: dict) -> str:
        y = state["y"]
        return self._act(actions, "update_mpc_target",
                         {"targets": {"xD": float(y["xD"]), "xB": float(y["xB"])},
                          "rationale": "Rule 2 (naive): sustained offset -- retarget MPC to the "
                                       "current measured composition."})


class RuleBasedSupervisorSmart(_RuleBasedSupervisorBase):
    """Rule 2 action: re-optimize via the RTO (do not simply accept the drift)."""

    VARIANT = "smart"

    def _rule2(self, actions: list, state: dict, cmd: dict) -> str:
        return self._act(actions, "trigger_rto_run",
                         {"rationale": "Rule 2 (smart): sustained offset -- re-optimize the "
                                       "operating point via the RTO."})


RULE_BASED_SUPERVISORS = {
    "rule-based-naive": RuleBasedSupervisorNaive,
    "rule-based-smart": RuleBasedSupervisorSmart,
}
