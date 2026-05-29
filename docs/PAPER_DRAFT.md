# Observability-by-Design Supervisory Control for Wood–Berry MPC via MCP

**Working title.** A Model Context Protocol boundary for offset-prone linear MPC: innovation-based
diagnosis and diagnostics-gated economics.
**Artifact.** Repo at the `v0.1.0-paper-closeout` tag, **93 tests passing**, reproducible benchmarks
and figures (see `REPRODUCE.md`).

> Status: DRAFT for internal review. Numbers below are pulled from the committed benchmark JSONs
> (`experiments/outputs/scenario{1,2}/benchmark_results.json`). Citations are web-verified but carry
> a MUST-VERIFY-DOI flag before submission (see §7).

---

## Abstract
Model predictive control (MPC) with integral or output-disturbance correction can absorb sustained
model–plant mismatch, leaving higher supervisory layers blind to load, sensor faults, and model
error. We implement the canonical Wood–Berry 2×2 FOPDT distillation benchmark under a condensed
linear MPC regulator that **deliberately omits integral action**, so persistent mismatch appears as
Kalman innovation bias and steady-state offset rather than being silently corrected. A Model Context
Protocol (MCP) server exposes this closed-loop simulation as a **safe tool surface**: read-only
diagnostics and a single bounded composition-setpoint write clipped to a physical envelope, with no
direct manipulation of reflux or steam and no real-time optimization in scope. A deterministic
diagnostic supervisor (Scenario 2) maps diagnostics to **HOLD, VETO_HOLD, PROPOSE_SETPOINT, or
ESCALATE** (with any setpoint write clipped to the envelope); a lightweight economics layer
(Scenario 1) optimizes a static margin objective **only when diagnostics are green** and is blocked
under load and ambiguity. Benchmarks on five diagnostic and three hierarchical cases, with thesis and
gate figures, demonstrate veto of physically impossible sensor readings, bounded clipped response to
coupled disturbances, honest escalation on single-channel ambiguous mismatch, and correct interlocking
of economics with diagnostics — all while the minute-by-minute SLSQP MPC remains untouched inside the
regulatory boundary.

---

## Contributions
1. **Regulatory MPC with deliberate non–offset-free disturbance rejection** — Kalman innovation and
   steady-state offset are first-class observables, not hidden by integral action.
2. **MCP supervisory boundary** — six stdio tools; one bounded write; envelope projection; no R/S
   commands; economics/RTO explicitly out of server scope.
3. **Deterministic Scenario 2 supervisor** — innovation-based (input-corrected) mismatch detection,
   physical-plausibility veto, latching, and escalation at the observability limit.
4. **Scenario 1-lite hierarchy** — static economics gated by Scenario 2; demonstrates the safe
   interlock without RTO.
5. **Reproducible artifact** — 93 tests, benchmark suite, `thesis_figure.png`, `scenario1_figure.png`.

---

## 1. Introduction
Industrial control hierarchies separate fast regulatory control (MPC/APC) from slower supervisory
optimization (RTO): MPC tracks setpoints and constraints; RTO moves setpoints for economics. Agentic
and LLM-based systems are increasingly proposed for supervisory and fault-handling roles, but direct
LLM access to manipulated variables is unsafe and non-reproducible.

We ask: *Can a certified linear MPC remain the sole regulator while an external agent observes
mismatch and proposes only auditable supervisory moves?* Our answer is a **simulation-backed MCP
server** plus **deterministic supervisors** that consume the same diagnostics an offset-prone MPC
exposes on purpose.

**Scope.** Single benchmark plant (Wood–Berry), simulation only, linear condensed MPC, static
economics (no RTO), LLM hooks for narrative only (not used in decisions).

---

## 2. Methods
### 2.1 Plant and regulatory MPC
Wood–Berry 2×2 binary distillation: manipulated variables reflux R and steam S (lb/min); controlled
outputs overhead xD and bottoms xB (mole fraction). FOPDT transfer matrix with coupled gains
(K = [[12.8, −18.9],[6.6, −19.4]]), time constants 10.9–21 min, transport delays 1–7 min; per-channel
ZOH states + input delays, additive sensor noise (std 2×10⁻⁴). Nominal: xD ≈ 0.96, xB ≈ 0.005 at
R ≈ 1.95, S ≈ 1.71.

Condensed linear MPC: prediction horizon N ≈ 30, control horizon M ≈ 5, solved with `scipy` SLSQP.
Kalman-style observer with innovation e = y_meas − y_pred. Steady-state target map
u_target = K_dc⁻¹(y_sp − y_nom) for setpoint changes. **No integral action** on sustained disturbances
(by design — see §4).

### 2.2 MCP supervisory boundary
A steppable closed-loop sim (plant + MPC) is exposed via six stdio tools: `info`, `reset_sim`,
`advance`, `get_mpc_diagnostics`, `get_plant_snapshot`, `set_target`. Setpoint envelope
xD ∈ [0.90, 0.99], xB ∈ [0.001, 0.05]; out-of-range proposals are **clipped** before reaching the
controller. **No tool commands R or S.** Diagnostics exposed: per-output innovation mean/variance,
recent ISE, active constraints, steady-state offset (y − y_sp).

### 2.3 Scenario 2 — DiagnosticSupervisor
Deterministic policy over the MCP diagnostics. Mismatch trigger: the **per-output Kalman innovation**
(plus active constraints), **not raw offset** — offset is transiently large after any setpoint change
while innovation is not, so gating on innovation avoids false alarms from the supervisor's own moves.
Physical plausibility is window-scanned with a clear-violation tolerance (≫ sensor noise); serious
findings latch. Decisions: **HOLD, VETO_HOLD, PROPOSE_SETPOINT, ESCALATE**; any proposed setpoint is
clipped to the envelope. The LLM `narrative_prompt` hook is **not invoked** in benchmarks.

### 2.4 Scenario 1-lite — Scenario1Agent (gated economics)
Static margin objective **J = w_D·xD − w_B·xB − w_S·S** (config weights, here w_D = 1.0, w_B = 1.0,
w_S = 0.3; S from the K_dc steady-state map). Grid search over an **economic operating band**
(xD ∈ [0.90, 0.975], xB ∈ [0.005, 0.05]) kept strictly inside the hard safety envelope to retain
control headroom from the physical limits. **Gate:** Scenario 2 must report NOMINAL; otherwise no
economic setpoint is proposed.

### 2.5 Validation protocol
PRBS + step tests for the plant/MPC; an MCP stdio client exercising the protocol; unit tests
(**93 total**). Scenario benchmarks: **5 (S2) + 3 (S1)** reproducible, seed-fixed cases.

---

## 3. Results
### 3.1 Scenario 2 — Diagnostic supervision (core)
Under an injected coupled load the no-integral MPC exhibits **sustained innovation bias** and
**steady-state offset** — the designed observability signature. The supervisor responds by case class:

| Case | Trigger | Decision | Interpretation |
|------|---------|----------|----------------|
| Nominal | innovation within noise | **HOLD** | normal tracking |
| Sensor fault (xD reads 1.010) | physically implausible reading | **VETO_HOLD** | analyzer fault; do not chase with setpoints |
| Coupled load | both channel innovations biased | **PROPOSE_SETPOINT** (→ xD 0.956, xB 0.012, clipped) | real disturbance; bounded supervisory action |
| Ambiguous load | single-channel innovation (xD −1.74×10⁻³), xB at floor | **ESCALATE** | load vs in-range sensor bias not separable |
| Out-of-range (xD = 1.2) | envelope violation | **CLIP → 0.99** | safety boundary enforced on the write |

**Figure 1 (`thesis_figure.png`).** Side-by-side: disturbance → innovation + offset rise → **veto**
(sensor) vs **bounded action** (coupled load).

**Key finding.** Escalation on ambiguous single-channel mismatch is **correct behavior**, not a
failure — it encodes the observability limit honestly (an additive single-channel load and an
in-range single-channel sensor bias are indistinguishable from (y, u, innovation) alone) rather than
misclassifying.

### 3.2 Scenario 1-lite — Diagnostics-gated economics
| Case | S2 state | S1 behavior | Result |
|------|----------|-------------|--------|
| Green | NOMINAL | economic setpoint applied every cycle | margin **+0.0147** step, holds |
| Coupled load | REAL_DISTURBANCE after event | blocked | margin drops; economics **does not chase** |
| Ambiguous load | AMBIGUOUS after event | blocked | no economic move under uncertainty |

**Figure 2 (`scenario1_figure.png`).** Green band: economics active; red band: gated by Scenario 2
during the disturbance.

**Key finding.** Hierarchical interlock: optimization runs only when diagnostics permit; the
regulatory MPC and the safety envelope are unchanged throughout.

### 3.3 Combined story
A **three-layer separation**: (1) **Regulatory** — SLSQP MPC, certified, no agent access;
(2) **Diagnostic** — innovation-based FDI + veto/escalate + envelope clip; (3) **Economic (lite)** —
static objective, gated by (2). MCP is the **engineering boundary**, not the scientific claim; the
claim is **observability-by-design** plus **deterministic supervisory policies** validated on a
standard benchmark.

---

## 4. Discussion
**Why innovation, not offset alone.** Offset conflates setpoint tracking with disturbance; the
input-corrected innovation isolates model–measurement mismatch and prevents the supervisor's own
setpoint moves from false-alarming (a fix verified to be necessary for the gated-economics layer).

**Why no LLM in the decision loop.** Reproducibility, auditability, and zero token cost for
benchmarks; the LLM is reserved for an optional narrative/interpretation layer that never changes a
decision.

**Relation to RTO and agent literature.** Analogous to the RTO→MPC setpoint cascade, but scoped to
simulation and static economics; aligns with agent + digital-twin validation patterns (monitor →
act → validate-on-twin → reprompt) without granting manipulated-variable access.

---

## 5. Limitations (honest)
- **Single plant**, linear model, simulation only (no live column, OPC-UA, or noise beyond the model).
- **Small benchmark N** (5 + 3 cases) — sufficient for proof-of-concept, not an exhaustive FDI taxonomy.
- **Ambiguous case** deliberately escalates; no automatic disambiguation without extra sensors/tests.
- **Scenario 1** is a static margin, not RTO; no feed, pricing dynamics, or constraint-active economics.
- **MCP-as-engineering** — a protocol choice for tool standardization; the results do not depend on
  MCP vs REST.
- **No integral-MPC baseline figure** in the artifact yet (recommended future comparison: same
  disturbance, mismatch hidden vs visible).
- **Linear (unclamped) plant** can produce compositions slightly outside [0,1] under aggressive
  control; handled by an economic operating band with headroom and a clear-violation plausibility tol.

---

## 6. Artifact and reproduction
See `REPRODUCE.md`. In brief:
```bash
pip install -e .[mcp]
pytest -q                                   # expect 93 passed
python experiments/scenario2_benchmark.py   # -> outputs/scenario2/{thesis_figure.png, benchmark_results.json}
python experiments/scenario1_benchmark.py   # -> outputs/scenario1/{scenario1_figure.png, benchmark_results.json}
python examples/mcp_smoke_client.py         # exercises the MCP stdio protocol end-to-end
```
Tag: `v0.1.0-paper-closeout`.

---

## 7. References (web-verified 2026-05-29; confirm DOIs before submission)
- InstructMPC: human/LLM-in-the-loop context for MPC — arXiv:2504.05946.
- LLMPC (LLM *planning* viewed as MPC; not a regulator replacement) — arXiv:2501.02486.
- Model Context Protocol — modelcontextprotocol.io spec + Anthropic announcement (Nov 2024).
- LLM agents + digital twins for fault handling in process plants (monitor→act→validate→reprompt) —
  arXiv:2505.02076.
- MCP in manufacturing — arXiv:2506.11180.
- Innovation-based FDI (seminal): Mehra & Peschon (1971), *Automatica* 7(5):637–640.
- (TO-PICK) RTO↔MPC hierarchy review — e.g. Darby et al. (2011), *J. Process Control* 21(6).

---

## Figure captions (for submission)
**Figure 1.** Wood–Berry closed loop under no-integral MPC: an injected disturbance produces
innovation bias and steady offset. Scenario 2 vetoes a physically impossible overhead reading
(VETO_HOLD) while proposing a bounded, envelope-clipped setpoint under coupled load
(PROPOSE_SETPOINT).

**Figure 2.** Scenario 1-lite static economics gated by Scenario 2 diagnostics. Green: margin
improves (+0.0147) when diagnostics are nominal. Red: a coupled disturbance gates economics; margin
falls but the supervisor does not chase it with setpoints.
