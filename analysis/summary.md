# Phase 1.5 — Executive Summary

**Experiment.** Four supervisors (LLM-agentic `qwen3:30b`, scheduled-RTO baseline, rule-based-naive,
rule-based-smart) × two RTO variants (MA, MA-GP) × seven plant-event scenarios (R1–R7) on the
Wood-Berry column. 56 runs, seed 42, 240-min scenarios, RTO+agent cadence 60 min, MPC 1 min. Full
tables in `RESULTS.md`; CSVs in `tables/`.

## Headline findings

1. **The LLM agent added no measurable value over the scheduled-RTO baseline.** Across all 7
   scenarios and both RTO variants, the agentic and baseline runs reach **bit-identical** final
   operating points (Δ realized xD = 0.000000). The agent took a consequential action **zero** times
   — 0 actions in the 7 MA scenarios; 2 single `trigger_rto_run`s in MA-GP (R1, R7), both on the
   final cycle (t=240) with no effect. With `qwen3:30b` the agent *observes and reasons* but does not
   *act* — the same pattern seen earlier at 4B, now confirmed at 30B. **The Phase-1 value claim is
   not supported by this data and should not be made until the agent is induced to act.**

2. **Rule-based supervisors act but don't change outcomes — and sometimes over-react.** Rule 1
   (|innovation| > 5e-4) fires on R1, R2, R6, R7 under MA, adding up to +3 RTO solves per scenario,
   yet the endpoints match baseline to ≤0.001 xD. On **R1** (benign slow drift) and **R6** (sensor
   fault) the rule books fire on signals the agent correctly ignores — so the agent shows better
   *restraint*, though via universal passivity rather than demonstrated diagnosis. `rule_naive` and
   `rule_smart` are identical everywhere (their distinguishing Rule 2 never fired).

3. **R6 (sensor gross error) is the most important result: it breaks the RTO, and nothing recovers
   it.** A +0.05 analyzer bias on xD (t=100–140) drives the MA RTO from converged to **infeasible at
   t=120**, and it stays infeasible through t=240 — *after* the bias clears — because the corrupted
   modifiers persist. No supervisor caught or corrected it. This is the clearest motivation for
   genuine supervisory diagnosis (and the planned data-reconciliation layer).

4. **R5 infeasibility is handled correctly by all 8 configurations** — RTO returns `converged=False`,
   the loop holds the last valid setpoint, no crash, no inappropriate action. The earlier R5 fix
   holds up across the full matrix.

5. **MA adapts; MA-GP does not (within scenario length).** Classical MA tracks the shifted optimum on
   the economic scenarios (R3 xD→0.92, R4 xD→0.99); MA-GP holds ≈0.96 in every scenario and reports
   `converged=False` throughout, because its trust-region BO needs ~6 seeding samples but a scenario
   provides only ~5–6 RTO solves — it never leaves seeding. The MA-vs-MA-GP gap therefore reflects
   non-adaptation, not GP quality; a fair GP comparison needs longer scenarios / faster RTO cadence.

## What this means for the paper

The honest Phase-1 story is **not** "the LLM agent improves control." It is: (i) a fully wired,
reproducible autonomous-supervisor testbed over MPC+RTO on a benchmark column; (ii) a clean negative/
restraint result — the agent does not act and so neither helps nor harms, while a thresholded rule
book acts but inconsequentially and sometimes over-reacts; and (iii) two concrete, figure-worthy
findings: a **sensor-fault RTO failure mode (R6)** that no current supervisor handles, and an
**MA-GP adaptation limitation** at realistic RTO cadences. The constructive next step is to make the
agent act reliably (baseline-relative health signals, sharper action directives, earlier cadence)
and re-run — and to lengthen scenarios so MA-GP is testable — before any value claim.

## Recommended figures
1. R6 sensor-fault: measured-vs-true xD + RTO `converged` flag flipping at t=120 and not recovering.
2. R3/R4: MA vs MA-GP xD trajectories (adapts vs holds).
3. Supervisor activity vs Δ-outcome bars (rule books act, agent holds, neither moves the endpoint).

## Caveats
LLM stochastic and prone to mis-reading small magnitudes (an MA-GP R1 rationale wrongly claimed
"infeasible"); most supervisor endpoint differences are at/below the 2e-4 noise floor; `baseline_ma-gp`
was regenerated locally (deterministic) after a transient pod crash at R3.
