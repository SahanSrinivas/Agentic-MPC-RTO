# MCP Supervisory-Agent Design (Build Spec)

Status: **design / build spec** (no code yet). Target: wrap the *existing* Wood-Berry MPC/RTO
stack behind a Model Context Protocol (MCP) server so an agent can **read diagnostics and propose
bounded supervisory moves** — never touch R/S or the QP. Citations in the prose architecture are
tracked separately and are **MUST-VERIFY before any paper use** (see `docs/` references task).

## 1. Principles (non-negotiable)
1. **The LLM/agent never enters the regulator.** The condensed-QP SLSQP MPC computes R/S every
   minute, unchanged. No LLM picks ΔR/ΔS; no LLM replaces the QP.
2. **MCP is the only interface.** Agents read state and *propose* supervisory moves through tools;
   they cannot call controller/plant objects directly.
3. **Writes are bounded, validated, twin-gated, audited.** Every write passes a fixed validator
   chain (below) and is rate-limited and logged. Mode changes require human ack.
4. **The no-integral MPC is a feature.** Sustained mismatch surfaces as steady offset + biased
   innovation — that telemetry is the supervisory layer's evidence and must not be silently
   absorbed by adding integral action to the *primary* MPC.

## 2. Layering
```
Human / MES / economics  ──goals, prices, limits──►  Agent + LLM (supervisory, 15 min–hours)
                                                          │  (reason over diagnostics + economics)
                                                          ▼
                                            ┌─────────  MCP server  ─────────┐   ← ONLY interface
                                            │ read:  get_mpc_diagnostics      │
                                            │        get_plant_snapshot       │
                                            │        get_economics            │
                                            │        get_rto_status           │
                                            │ twin:  simulate_woodberry       │
                                            │        run_offline_prbs_check   │
                                            │ draft: propose_setpoints        │
                                            │ write: apply_setpoints  (gated) │
                                            │        set_mpc_mode     (gated) │
                                            │        create_incident_report   │
                                            └────────────────┬────────────────┘
                                  validator chain (args → box → rate → twin → audit)
                                                             ▼
                                   Your MPC (unchanged, ~1 min):  Kalman + innovation → condensed QP (SLSQP)
                                                             ▼
                                   Wood-Berry plant/sim:  2×2 FOPDT + delays + noise
```

## 3. Component inventory — what exists vs. what to build
**EXISTS (wrap, don't rebuild):**
- `agentic_mpc.agent.tools` — `get_process_state`, `get_mpc_health`, `update_mpc_target`,
  `trigger_rto_run`, `get_economic_context`, `get_rto_status`, bound via `AgentContext` +
  `make_tool_registry`. These are the capability surface the MCP server re-exports.
- `ClassicalMPC.get_health()` → `{innovation_mean{xD,xB}, innovation_std{xD,xB}, active_constraints,
  ise_recent, ise_recent_by_output, innovation_window_samples}` — Scenario-2 inputs already exist.
- `WoodBerryPlant.get_state()` → `{t, y{xD,xB}, u{R,S}, history{t,y,u}}` (30-sample window).
- `BoxSafetyEnvelope` — clips targets to `xD∈[0.90,0.99]`, `xB∈[0.001,0.05]`; returns
  `(safe_action, was_violated)`. This is the box-bound stage of the write validator.
- `agentic_mpc.agent.validation.validate_and_repair_args` — pydantic tool-arg validation/repair
  (the schema stage of the write validator).
- `RTOMPCLoop` — `run`, `get_rto_status`, `note_external_command`, `request_rto_recompute`; history
  now records `xD_true/xB_true` (true economic state, for the twin/metric).
- PRBS validation protocol (`experiments/phase1_prbs_validation.py`) — basis for the model-drift check.

**TO-BUILD (small, no new control math):**
- `steady_state_offset` field: `y − y_sp` per output (trivial; add to the diagnostics tool).
- Twin snapshot/restore: `WoodBerryPlant.snapshot()/restore(state)` exporting the FO states `_x`,
  the per-input delay buffers, biases, and RNG — needed for an *accurate* forward sim from the live
  point (without it the twin is directional-only; see §6 fidelity note).
- Validator chain wrapper + audit log + rate limiter (new thin module, e.g. `mcp/validator.py`).
- The MCP server itself (e.g. `mcp/server.py`) exporting the tools below.
- Scenario-2 monitor/state-machine (`mcp/diagnostics_fsm.py`).
- `offset_free_disturbance` MPC variant is **not implemented** — `set_mpc_mode` exposes it as a
  feature flag that currently returns `not_available` (documented future work, not silent integral).

## 4. MCP tool contracts (JSON)
Read tools are side-effect-free. `twin` tools run a simulation, no live effect. `write` tools pass
the validator chain (§5).

### 4.1 Read
```jsonc
// get_mpc_diagnostics  (wraps get_mpc_health + adds offset)
{ "name": "get_mpc_diagnostics", "input": {},
  "output": {
    "t": "number",
    "innovation_mean": {"xD": "number", "xB": "number"},
    "innovation_std":  {"xD": "number", "xB": "number"},
    "ise_recent": "number", "ise_recent_by_output": {"xD": "number", "xB": "number"},
    "active_constraints": ["string"],
    "innovation_window_samples": "integer",
    "steady_state_offset": {"xD": "number", "xB": "number"},   // NEW: y - y_sp
    "setpoints": {"xD": "number", "xB": "number"} } }

// get_plant_snapshot  (wraps get_process_state)
{ "name": "get_plant_snapshot", "input": {},
  "output": { "t":"number", "y":{"xD":"number","xB":"number"}, "u":{"R":"number","S":"number"},
              "history": {"t":["number"], "y":{"xD":["number"],"xB":["number"]},
                          "u":{"R":["number"],"S":["number"]}},
              "is_simulation": "boolean" } }

// get_economics  (wraps get_economic_context)
{ "name": "get_economics", "input": {},
  "output": { "prices": {"F":"number","z_F":"number","p_D":"number","p_pen":"number",
                         "p_B":"number","c_S":"number","xB_max":"number","D_max":"number|null"},
              "last_rto_objective":"number", "model_plant_gap":"number|null",
              "current_rto_setpoints": {"xD":"number","xB":"number"} } }

// get_rto_status  (wraps loop.get_rto_status)
{ "name": "get_rto_status", "input": {},
  "output": { "rto_has_run":"boolean", "commanded_setpoints":{"xD":"number","xB":"number"},
              "commanded_at_min":"number", "minutes_since_command":"number",
              "rto_variant":"string", "rto_converged":"boolean", "rto_solve_status":"string",
              "settled_since_command":"boolean", "n_rto_commands":"integer" } }
```

### 4.2 Twin (read-only simulation)
```jsonc
// simulate_woodberry — roll the SAME plant+MPC code forward under a candidate supervisory change
{ "name": "simulate_woodberry",
  "input": { "candidate": { "setpoints": {"xD":"number","xB":"number"},   // optional
                            "mpc_mode": "primary|offset_free_disturbance|hold" }, // optional
             "horizon_min": "integer (default 180)",
             "from": "live_snapshot|nominal (default live_snapshot)" },
  "output": { "predicted_regret": "number",            // integral economic regret on xD_true/xB_true
              "predicted_offset_end": {"xD":"number","xB":"number"},
              "constraint_violations": ["string"], "feasible": "boolean",
              "fidelity": "exact|directional" } }      // exact iff snapshot/restore available

// run_offline_prbs_check — re-identify G vs nominal Wood-Berry, flag drift
{ "name": "run_offline_prbs_check", "input": {"channels": ["R->xD","S->xB","R->xB","S->xD"]},
  "output": { "per_channel": [{"channel":"string","gain_ratio":"number","delay_err_min":"number",
                               "drift_flag":"boolean"}], "overall_drift": "boolean" } }
```

### 4.3 Draft + Write
```jsonc
// propose_setpoints — returns a candidate + twin prediction; DOES NOT apply
{ "name": "propose_setpoints",
  "input": { "targets": {"xD":"number","xB":"number"}, "rationale": "string" },
  "output": { "candidate": {"xD":"number","xB":"number"}, "clipped_by_box": "boolean",
              "predicted_margin_delta": "number", "predicted_offset_end": {"xD":"number","xB":"number"},
              "validator": {"passed":"boolean","reasons":["string"]} } }

// apply_setpoints — commits ONLY after the full validator chain (§5)
{ "name": "apply_setpoints",
  "input": { "targets": {"xD":"number","xB":"number"}, "rationale": "string",
             "ack_token": "string|null" },           // required iff a rule demands human ack
  "output": { "status": "applied|rejected|needs_ack", "applied_targets":{"xD":"number","xB":"number"},
              "clipped_by_box":"boolean", "audit_id":"string", "reasons":["string"] } }

// set_mpc_mode — Scenario-2 mode switch (human ack required)
{ "name": "set_mpc_mode",
  "input": { "mode": "primary|offset_free_disturbance|hold", "rationale":"string", "ack_token":"string" },
  "output": { "status":"applied|rejected|not_available|needs_ack", "mode":"string", "audit_id":"string" } }

// create_incident_report — write a human-facing markdown report (no control effect)
{ "name": "create_incident_report",
  "input": { "title":"string", "summary":"string", "evidence":"object", "recommended_actions":["string"] },
  "output": { "status":"written", "path":"string", "audit_id":"string" } }
```

## 5. Write-path validator chain (fixed order; any stage may reject)
```
apply_setpoints / set_mpc_mode
  1. SCHEMA   : validate_and_repair_args  → reject on un-repairable args
  2. BOX      : BoxSafetyEnvelope.project → clip into xD∈[0.90,0.99], xB∈[0.001,0.05]; record clipped
  3. RATE     : ≤1 setpoint write / agent-cycle; |Δxsp| per write ≤ {xD:0.02, xB:0.005} (anti-thrash)
  4. DIAG-GATE: refuse to apply economic moves while diagnostics say the model is wrong
               (Scenario-2 state ∈ {DIAGNOSE, MITIGATE, ESCALATE}) → status "rejected: model unexplained"
  5. TWIN     : simulate_woodberry(candidate, horizon=180); reject if infeasible or
               predicted_regret worse than hold by > tolerance
  6. ACK      : set_mpc_mode and any out-of-envelope override require a valid human ack_token
  7. AUDIT    : append {audit_id, t, tool, input, validator trace, outcome} to an append-only log
```
This chain *is* the "Validator + twin check" box. It reuses `validate_and_repair_args` (stage 1) and
`BoxSafetyEnvelope` (stage 2) verbatim; stages 3–7 are the new thin wrapper.

## 6. Scenario-2 diagnostic state machine (innovation-driven, allow-listed)
Consumes `get_mpc_diagnostics` on a fast monitor clock (1–5 min). Thresholds reuse the rule-based
provenance: nominal `|innovation_mean|`≈1e-5; trigger band `>5e-4` (~2.5× sensor-noise std, ~50×
nominal). Offset band reuses `BoxSafetyEnvelope`/operating-envelope scales (`xD 5e-3`, `xB 1e-3`).

```
States:  NOMINAL ─▶ WATCH ─▶ DIAGNOSE ─▶ MITIGATE ─▶ (NOMINAL | ESCALATE)
                       └──────────────┘ (clears if signal subsides)

Transitions (triggers):
  NOMINAL → WATCH    : CUSUM/χ² on innovation crosses pre-alarm (|innov_mean| > 5e-4 sustained ≥3 samples)
  WATCH   → DIAGNOSE : sustained ≥ W samples OR ISE > ~10× nominal OR persistent offset > band
  DIAGNOSE→ MITIGATE : a hypothesis passes the corroboration test (below) AND twin agrees
  DIAGNOSE→ ESCALATE : hypothesis is "sensor/analyzer fault" OR not identifiable (see §7) → human
  MITIGATE→ NOMINAL  : post-action twin replay shows signal cleared
  any     → NOMINAL  : signal subsides within WATCH dwell

Allowed actions PER STATE (hard allow-list; nothing else is callable):
  NOMINAL  : { read only }
  WATCH    : { read, run_offline_prbs_check, simulate_woodberry }
  DIAGNOSE : { read, run_offline_prbs_check, simulate_woodberry, classify_mismatch, create_incident_report }
  MITIGATE : { propose_setpoints (guardband relax only), set_mpc_mode→hold,
               request operator check (incident_report) }      // NO economic optimization here
  ESCALATE : { create_incident_report, set_mpc_mode→hold (needs ack) }  // human owns the decision

Corroboration test (the classifier; reuses the gain matrix):
  ρ = Δy_window − K·Δu_window     (K = nominal Wood-Berry gains)
  • residual coupled across BOTH outputs        → "real disturbance"  → MITIGATE/optimize path
  • residual isolated to one analyzer (other at noise floor) → "suspected sensor fault" → ESCALATE/HOLD
  • PRBS check shows gain/delay drift            → "model drift"       → flag retune
```
**Scenario-1 interlock:** the economic agent's `apply_setpoints` is *blocked* (validator stage 4)
whenever the FSM is in DIAGNOSE/MITIGATE/ESCALATE — so the optimizer can't chase a ghost optimum
while the model is known-wrong. This is the concrete "Scenario 2 vetoes Scenario 1" link.

## 7. Identifiability ceiling (honest limits — do not oversell the classifier)
From the S1 work: **an additive load on one output and a sensor bias on the same analyzer are
observationally identical** (same `y`, `u`, innovation). The classifier can only separate faults with
*distinct observable signatures*:
- ✅ separable: physically-implausible reading (∉[0,1]); **coupled** disturbance vs **single-channel**
  analyzer fault; gain/delay drift (via PRBS).
- ❌ NOT separable from innovation alone: single-channel load vs single-channel sensor bias;
  "wrong steady-state map" vs "unmeasured load" in general.
Unidentifiable cases must route to **ESCALATE (human)**, not a confident machine label.

## 8. Build order (milestones)
1. **Read-only MCP** — `get_mpc_diagnostics` (+offset), `get_plant_snapshot`, `get_economics`,
   `get_rto_status`. Usable from any MCP client for ops Q&A. *No write path.*
2. **Twin** — `WoodBerryPlant.snapshot/restore` + `simulate_woodberry` + `run_offline_prbs_check`;
   validator chain stages 1–3,5,7.
3. **Scenario 2 (diagnostics)** — the FSM + `classify_mismatch` + `create_incident_report`; lowest
   business risk and it directly exercises the no-integral observability thesis. **(Our evidence says
   this is where the LLM earns its place — R6 attribution 3/3 vs 0/3.)**
4. **Scenario 1 (economics)** — `propose_setpoints`/`apply_setpoints` with stage-4 diag-gate active.
   **Set expectations: the LLM's value here is NL-goal→bounded-request + tradeoff explanation, NOT a
   regret improvement over a rule (T6/T7/4-arm: LLM tied the rulebook exactly).**

## 9. Non-goals / anti-patterns (rejected)
- LLM choosing ΔR/ΔS per minute; LLM replacing SLSQP.
- Adding integral action to the *primary* MPC to "help" (destroys the observability we designed on).
- MCP tools that write MVs directly (bypass the regulator).
- Any agent action without twin + validator; any `set_mpc_mode` without human ack.

## 10. Open questions
- Twin fidelity: ship `snapshot/restore` (exact) or accept directional twin for v1?
- Where does the economic objective live for `predicted_margin_delta` — reuse `WoodBerryEconomics`
  (yes) and the regret metric (`analysis/control_regret.py`, P_opt=14.1338) for `predicted_regret`.
- Which MCP runtime/transport (stdio vs HTTP) and auth for the write path.
- `offset_free_disturbance` MPC variant: build now (enables a real MITIGATE mode-switch) or keep flagged?

## 11. References (verified 2026-05-29; confirm DOIs at submission)
All web-checked; each is real. Framing caveats noted so we don't mis-cite.
- **InstructMPC** — arXiv:2504.05946, "InstructMPC: A Human-LLM-in-the-Loop Framework for Context-Aware
  Control." Human/LLM supplies *context/disturbance trajectories* (Language-to-Distribution module) to
  the MPC; the MPC keeps the optimization. ✔ Supports "LLM supplies context, not the QP."
- **LLMPC** — arXiv:2501.02486 (G. Maher), "LLMPC: Large Language Model Predictive Control." About LLM
  *planning* viewed through an MPC lens — **not** replacing a regulator. ✔ Supports the anti-pattern
  "don't put the LLM inside the certified QP." (Cite precisely; easy to mis-summarize.)
- **MCP** — Model Context Protocol spec (modelcontextprotocol.io) + Anthropic announcement (Nov 2024).
  Primitives: tools/resources/prompts; JSON-RPC 2.0. ✔ Supports the MCP boundary.
- **LLM Agents + Digital Twins for Fault Handling in Process Plants** — arXiv:2505.02076. Multi-agent
  Monitoring → Action → Validation(twin) → Reprompting. ✔ **Directly supports §6** (Scenario-2 FSM,
  twin-validate, reprompt) — strongest single reference for this design.
- **MCP in Manufacturing** — arXiv:2506.11180, "Beyond Formal Semantics for Capabilities and Skills:
  Model Context Protocol in Manufacturing." ✔ Supports industrial capability exposure via MCP.
- **Innovation-based FDI (seminal)** — Mehra & Peschon (1971), *Automatica* 7(5):637–640, "An
  innovations approach to fault detection and diagnosis in dynamic systems." ✔ Supports innovation as
  the FDI signal (the core of our no-integral observability thesis).
- **Hybrid KF + ML anomaly detection** — Puder et al. (2024), *MDPI Sensors* 24(9):2895. ✔ Real, but
  a **medical/OR** application — cite as a *method analog* (KF-innovation + ML), not as process-control
  evidence.
- **RTO ↔ MPC hierarchy** — real, standard area; **pick a specific citation** before write-up (e.g.
  Darby, Nikolaou, Jones & Nicholson, "RTO: An overview and assessment of current practice," *J. Process
  Control* 21(6), 2011; and Biegler's RTO/optimization works). Marked TO-PICK, not yet pinned.

> Per project rule: these are web-verified (≥1 authoritative source each) but still carry a
> MUST-VERIFY-DOI flag before any paper submission.
