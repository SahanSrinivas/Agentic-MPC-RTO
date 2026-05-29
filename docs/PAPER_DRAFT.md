# A Digital Twin for Guardrailed LLM Supervisory Control of Wood–Berry MPC via Model Context Protocol

**Working title.** Digital-twin simulation of a distillation column with MCP as the supervisory
boundary: innovation-based diagnosis, deterministic guardrails, and Claude as an LLM advisor that
must obey the same rules as DCS-style logic.
**Artifact.** Repo on `main` after tag `v0.1.0-paper-closeout` (+ LLM-driven-setpoint increment), **100 tests passing**,
reproducible benchmarks and figures (see `REPRODUCE.md`).

> Status: DRAFT for internal review. Numbers below are pulled from committed benchmark JSONs
> (`experiments/outputs/scenario{1,2}/benchmark_results.json`,
> `experiments/outputs/comparison/comparison_table.json`). Citations are web-verified but carry a
> MUST-VERIFY-DOI flag before submission (see §8).

---

## Abstract

Model predictive control (MPC) with integral or output-disturbance correction can absorb sustained
model–plant mismatch, leaving higher supervisory layers blind to load, sensor faults, and model
error. We build a **digital twin** of the canonical Wood–Berry 2×2 FOPDT distillation column under a
condensed linear MPC regulator that **deliberately omits integral action**, so persistent mismatch
appears as Kalman innovation bias and steady-state offset rather than being silently corrected. A
Model Context Protocol (MCP) server exposes this closed-loop **simulation** as a safe tool surface:
read-only diagnostics and a single bounded composition-setpoint write clipped to a physical envelope,
with no direct manipulation of reflux or steam and no real-time optimization (RTO) in scope. A
deterministic diagnostic supervisor maps diagnostics to **HOLD, VETO_HOLD, PROPOSE_SETPOINT, or
ESCALATE** (with any setpoint write clipped to the envelope); a lightweight economics layer optimizes
a static margin objective **only when diagnostics are green**. A **three-arm comparison** on the same
plant, MPC, and fault cases contrasts MPC-only operation, MCP plus deterministic rules, and
**LLM (Claude) plus MCP with guardrails** — the rules remain validator and source of truth. On four
diagnostic cases Claude **matched the rules 4/4 (100%)** with **zero overrides**; MPC-only operation
**chased a biased xD analyzer** and drove true overhead composition from **0.960 → 0.917** while both
supervisory arms vetoed. The claim is not that the LLM beats rules, but that a digital twin with MCP
and guardrailed LLM advice can mirror auditable DCS-style supervisory logic while keeping the
minute-by-minute SLSQP MPC untouched inside the regulatory boundary.

---

## Contributions

1. **Digital twin of Wood–Berry + condensed MPC** — steppable closed-loop sim with deliberate
   non–offset-free disturbance rejection; innovation and offset as first-class observables.
2. **MCP supervisory boundary on the twin** — six stdio tools; one bounded write; envelope projection;
   no R/S commands; RTO explicitly out of scope.
3. **Deterministic diagnostic supervisor** — innovation-based mismatch detection, physical-plausibility
   veto, latching, and escalation at the observability limit (portable to DCS logic).
4. **Scenario 1-lite hierarchy** — static economics gated by diagnostics; interlock without RTO.
5. **Three-arm comparison** — MPC-only vs MCP+rules vs LLM+MCP+guardrails; rules as validator;
   **100% LLM–rules agreement**, **0 overrides** on benchmark cases; quantified MPC-only harm on
   sensor fault.
6. **Reproducible artifact** — 100 tests, thesis + gate figures, cached LLM responses for re-run
   without token cost.

---

## 1. Introduction

Industrial control hierarchies separate fast regulatory control (MPC/APC) from slower supervisory
optimization (RTO): MPC tracks setpoints and constraints; RTO moves setpoints for economics. We do
**not** implement RTO here — Scenario 1-lite is a static margin **proxy** for the economic layer
that would sit above diagnostics in a full hierarchy. Agentic and LLM-based systems are increasingly
proposed for supervisory and fault-handling roles, but direct LLM access to manipulated variables is
unsafe and non-reproducible.

We ask: *Can a certified linear MPC remain the sole regulator while a digital twin exposes mismatch
signals and external advisors — rule-based or LLM — propose only auditable supervisory moves?* Our
answer is a **simulation-backed MCP server** (the twin’s API) plus **deterministic supervisors** that
consume the same diagnostics an offset-prone MPC exposes on purpose, with an optional **guardrailed
LLM arm** that must agree with or be overridden by those rules.

**Scope.** Single benchmark plant (Wood–Berry), **simulation only** (digital twin, not live DCS),
linear condensed MPC, static economics (no RTO). LLM used in the three-arm study (Claude via
Anthropic API); deterministic rules always have final say.

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
(by design — see §4 Discussion).

### 2.2 Digital twin and MCP supervisory boundary

The **digital twin** is a steppable closed-loop simulation (plant + MPC). MCP exposes it via six stdio
tools: `info`, `reset_sim`, `advance`, `get_mpc_diagnostics`, `get_plant_snapshot`, `set_target`.
Setpoint envelope xD ∈ [0.90, 0.99], xB ∈ [0.001, 0.05]; out-of-range proposals are **clipped**
before reaching the controller. **No tool commands R or S.** The same supervisory policy could be
implemented in DCS calc blocks on live tags; MCP standardizes the twin interface for agents and LLMs.

Diagnostics exposed: per-output innovation mean/variance, recent ISE, active constraints, steady-state
offset (y − y_sp).

### 2.3 DiagnosticSupervisor (deterministic rules)

Deterministic policy over the twin diagnostics. Mismatch trigger: the **per-output Kalman innovation**
(plus active constraints), **not raw offset** — offset is transiently large after any setpoint change
while innovation is not, so gating on innovation avoids false alarms from the supervisor's own moves.
Physical plausibility is window-scanned with a clear-violation tolerance (≫ sensor noise); serious
findings latch. Supervisor decisions: **HOLD, VETO_HOLD, PROPOSE_SETPOINT, ESCALATE** only.
**CLIP** is envelope enforcement on any `set_target` write, not a fifth supervisor decision. The
LLM `narrative_prompt` hook supplies optional operator text; benchmarks use rules as authority unless
noted in §3.4.

### 2.4 Scenario 1-lite — gated economics

Static margin objective **J = w_D·xD − w_B·xB − w_S·S** (w_D = 1.0, w_B = 1.0, w_S = 0.3; S from
K_dc steady-state map). Grid search over an economic operating band (xD ∈ [0.90, 0.975],
xB ∈ [0.005, 0.05]) inside the hard safety envelope. **Gate:** diagnostics must report NOMINAL.

### 2.5 Three-arm comparison protocol

Same plant, same MPC, same seed (1), same four diagnostic episodes (inject at t = 100 min, decision
snapshot at t = 130 min) plus one out-of-range clip test:

| Arm | Supervisory layer | Setpoint authority |
|-----|-------------------|-------------------|
| **1 — MPC only** | None | Fixed initial setpoints; MPC tracks; no `set_target` |
| **2 — MCP + rules** | `DiagnosticSupervisor` on twin diagnostics | Rules propose; envelope clips |
| **3 — LLM + MCP + guardrails** | Claude reads same diagnostics JSON; proposes one of four actions | **Rules validate**; on disagreement, **rules win** (`reconcile()`) |

Arm 3 uses a measurement-validity system prompt (physically impossible compositions → VETO; coupled
innovations → PROPOSE; single-channel ambiguity → ESCALATE). LLM responses are cached to
`experiments/outputs/comparison/llm_responses.json` for reproducibility without re-spending tokens.

### 2.6 Validation protocol

PRBS + step tests for plant/MPC; MCP stdio smoke client; unit tests (**95 total**). Scenario
benchmarks: **5 (diagnostic) + 3 (economics gate) + 4 (three-arm diagnostic arms)** reproducible cases.

---

## 3. Results

### 3.1 Diagnostic supervision (Scenario 2 benchmark)

Under injected coupled load the no-integral MPC exhibits **sustained innovation bias** and
**steady-state offset** — the designed observability signature. The supervisor responds by case class:

| Case | Trigger | Supervisor decision | Envelope (if write) | Interpretation |
|------|---------|---------------------|---------------------|----------------|
| Nominal | innovation within noise | **HOLD** | — | normal tracking |
| Sensor fault (xD reads >1) | physically implausible reading | **VETO_HOLD** | — | analyzer fault; do not chase |
| Coupled load | both innovations biased | **PROPOSE_SETPOINT** (→ xD 0.956, xB 0.012) | clipped | real disturbance; bounded action |
| Ambiguous load | single-channel innovation (xD −1.74×10⁻³) | **ESCALATE** | — | load vs bias not separable |
| Out-of-range request (xD = 1.2) | — | — | **CLIP → 0.99** | safety on write, not a supervisor action |

**Figure 1 (`thesis_figure.png`).** Disturbance → innovation + offset rise → veto (sensor) vs bounded
action (coupled load).

### 3.2 Diagnostics-gated economics (Scenario 1-lite)

| Case | Diagnostic state | Economics behavior | Result |
|------|------------------|-------------------|--------|
| Green | NOMINAL | setpoint applied | margin **+0.0147** step, holds |
| Coupled load | REAL_DISTURBANCE | blocked | margin drops; no chase |
| Ambiguous load | AMBIGUOUS | blocked | no move under uncertainty |

**Figure 2 (`scenario1_figure.png`).** Green: economics active; red: gated by diagnostics during disturbance.

### 3.3 Three-arm comparison — MPC-only vs rules vs guardrailed LLM

**Table 1.** Supervisory policy comparison (digital twin, seed = 1, t_decide = 130 min after injection
at t = 100 min). True xD is the **plant state** (not the biased measurement).

| Case | MPC only (no supervisor) | MCP + rules | LLM + MCP (Claude proposal) | Rules = final | Match |
|------|--------------------------|-------------|----------------------------|---------------|-------|
| Nominal | tracks; true xD **0.960** | HOLD | HOLD | HOLD | ✓ |
| Sensor fault (+0.05 xD bias) | **chases bad reading**; true xD **0.917** | VETO_HOLD | VETO_HOLD | VETO_HOLD | ✓ |
| Coupled load | true xD **0.957**; no supervisory move | PROPOSE_SETPOINT | PROPOSE_SETPOINT | PROPOSE_SETPOINT | ✓ |
| Ambiguous load | true xD **0.956**; no escalate | ESCALATE | ESCALATE | ESCALATE | ✓ |
| Out-of-range (xD = 1.2) | N/A | CLIP → **0.99** | CLIP → **0.99** | CLIP → **0.99** | ✓ |

**LLM–rules agreement: 4/4 (100%). Overrides: 0.** The validator (`DiagnosticSupervisor`) is
source of truth; any future LLM disagreement would be logged and overridden.

**Headline findings.**

1. **MPC-only exposes mismatch but can misbehave** — on sensor fault, innovation mean xD =
   **+2.51×10⁻³** and the regulator trusts the corrupted measurement, driving **true** xD from
   **0.960 → 0.917** (reflux cuts to chase the bogus high reading).
2. **Rules fix it deterministically** — window-scanned plausibility detects xD excursion above 1.0 →
   VETO_HOLD.
3. **Guardrailed LLM matches rules** — Claude cited the xD range **[0.9646, 1.0097]** exceeding 1.0
   for veto; coupled and ambiguous cases matched without override. Claim: **LLM obeys guardrails**,
   not “LLM beats rules.”

Run: `python experiments/three_arm_comparison.py` (cached LLM arm); `--no-llm` for arms 1–2 only.

### 3.4 Combined story

A **digital twin** with three layers above the regulatory MPC: (1) **Diagnostic** — innovation-based
FDI, veto, escalate; (2) **Economic (lite)** — static margin, gated by (1); (3) **Optional LLM advisor**
— same diagnostics and actions as (1), validated by (1). MCP is the **engineering API** on the twin;
the scientific claims are **observability-by-design**, **deterministic supervisory policy**, and
**guardrailed LLM alignment** — portable to DCS setpoint supervision without MCP in production.

---

## 4. Discussion

**Digital twin vs plant.** All results are from simulation. The twin validates policy before DCS
implementation (innovation tags, plausibility checks, setpoint clamps on MPC composition SPs).

**Why innovation, not offset alone.** Offset conflates setpoint tracking with disturbance; innovation
isolates model–measurement mismatch and prevents false alarms after the supervisor’s own setpoint moves.

**LLM role.** Claude proposes; rules decide. This mirrors recommended agent architecture (monitor →
propose → validate → apply). Narrative richness without sacrificing auditability when the validator
wins.

**MCP vs DCS.** MCP standardizes agent access to the twin; a production deployment would map the same
logic to DCS/APC tags and alarms — MCP is not required on site.

**Relation to RTO.** RTO would sit **above** the diagnostic layer in a full hierarchy; we demonstrate
the diagnostic and gating slot only, with static economics as a stand-in.

---

## 5. Limitations

- **Simulation only** — digital twin, not live column or OPC-UA.
- **Small benchmark N** — proof-of-concept, not exhaustive FDI taxonomy.
- **Ambiguous case** escalates by design; disambiguation needs extra sensors or tests.
- **Scenario 1** is not RTO.
- **Single LLM model/run** — Claude Sonnet 4.6, temperature 0.1, four cached calls; broader model
  sweep is future work.
- **No integral-MPC baseline figure** yet (recommended: same disturbance, mismatch hidden vs visible).
- **Linear (unclamped) plant** can briefly exceed [0,1]; plausibility tolerance and economic band
  provide headroom.

---

## 6. Conclusions

We presented a **digital twin** of the Wood–Berry column with an offset-prone linear MPC that exposes
sustained mismatch as innovation bias and steady offset, a **deterministic diagnostic supervisor**
(veto on impossible readings, bounded propose on coupled load, escalate on ambiguity), and
**diagnostics-gated static economics**. An MCP server exposes the twin for safe agent interaction; a
**three-arm study** showed MPC-only operation chases a sensor fault (true xD **0.960 → 0.917**) while
rules and guardrailed Claude both veto (**100% agreement, zero overrides**). The supervisory policy is
portable to DCS logic; MCP and LLM are enablers for simulation and guardrailed advisory, not
replacements for the regulatory MPC.

---

## 7. Artifact and reproduction

See `REPRODUCE.md`. In brief:

```bash
pip install -e .[mcp]
pytest -q                                              # expect 100 passed
python experiments/scenario2_benchmark.py
python experiments/scenario1_benchmark.py
python experiments/three_arm_comparison.py             # LLM arm: ANTHROPIC_API_KEY set
python experiments/three_arm_comparison.py --no-llm    # arms 1-2 only, no tokens
python examples/mcp_smoke_client.py
```

Tags: `v0.1.0-paper-closeout` (S1/S2 closeout); three-arm comparison at `e44dc3e`.

---

## 8. References (web-verified 2026-05-29; confirm DOIs before submission)

- Wood & Berry (1973) — Wood–Berry distillation benchmark (TO-PICK full citation).
- Qin & Badgwell (2003) — MPC survey (TO-PICK).
- Darby et al. (2011) — RTO↔MPC hierarchy review, *J. Process Control* 21(6).
- Mehra & Peschon (1971) — innovation-based FDI, *Automatica* 7(5):637–640.
- LLM agents + digital twins for fault handling — arXiv:2505.02076.
- MCP in manufacturing — arXiv:2506.11180.
- Model Context Protocol — modelcontextprotocol.io specification.

---

## Figure captions (for submission)

**Figure 1.** Digital twin, Wood–Berry closed loop under no-integral MPC: injected disturbance
produces innovation bias and steady offset. Diagnostic supervisor vetoes a physically impossible
overhead reading (VETO_HOLD) while proposing a bounded, envelope-clipped setpoint under coupled load
(PROPOSE_SETPOINT).

**Figure 2.** Static economics gated by diagnostic state. Green: margin improves (+0.0147) when
diagnostics are nominal. Red: coupled disturbance gates economics; margin falls but no setpoint chase.

**Table 1.** Three-arm supervisory comparison (§3.3): MPC-only misbehavior on sensor fault vs
rule-based and guardrailed LLM supervisory arms (100% LLM–rules agreement).

**Graphical abstract (suggested).** Left: innovation/offset time series under fault; right: twin
hierarchy (MPC → diagnostics → rules/LLM → clipped setpoints); caption: “Digital twin exposes mismatch;
guardrailed LLM matches deterministic rules.”
