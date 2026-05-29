# Reproducing the artifact

Tag: `v0.1.0-paper-closeout`. All steps are deterministic and run on a laptop (no GPU, no LLM, no
network). The companion write-up is `docs/PAPER_DRAFT.md`; the architecture spec is
`docs/MCP_SUPERVISOR_DESIGN.md`.

## 0. Environment
```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows; use bin/activate on POSIX
pip install -e .[mcp]                                 # installs deps + the mcp SDK (Milestone-1 server)
```
Python ≥ 3.10 (developed on 3.12).

## 1. Tests
```bash
pytest -q          # expect: 100 passed
```

## 2. Scenario 2 — diagnostic supervision (the core result + thesis figure)
```bash
python experiments/scenario2_benchmark.py
```
Writes `experiments/outputs/scenario2/`:
- `thesis_figure.png` — disturbance → innovation + offset → veto (sensor) / bounded action (coupled).
- `benchmark_results.json` — 5 cases. Expected: nominal→HOLD, sensor_fault→VETO_HOLD,
  coupled_load→PROPOSE_SETPOINT, ambiguous_load→ESCALATE, out_of_range_request→CLIP (xD→0.99).
  `all_ok: true`.

## 3. Scenario 1-lite — diagnostics-gated economics (gate figure)
```bash
python experiments/scenario1_benchmark.py
```
Writes `experiments/outputs/scenario1/`:
- `scenario1_figure.png` — margin vs time, green = economics active, red = gated by Scenario 2.
- `benchmark_results.json` — green: not gated, margin gain ≈ **+0.0147**; coupled_load & ambiguous_load:
  gated after the disturbance. `all_ok: true`.

## 4. Three-arm comparison (MPC-only / MCP+rules / LLM+MCP+guardrails)
```bash
python experiments/three_arm_comparison.py              # LLM arm: needs ANTHROPIC_API_KEY + network
python experiments/three_arm_comparison.py --no-llm   # arms 1-2 only, deterministic, no tokens
```
Writes `experiments/outputs/comparison/`:
- `comparison_table.json` — Table 1 in `docs/PAPER_DRAFT.md` §3.3.
- `llm_responses.json` — cached Claude proposals (re-used on re-run unless `--refresh`).

Expected: LLM–rules match **4/4**, overrides **0**; sensor_fault true xD **0.960 → 0.917** under MPC-only.

## 5. MCP server (Milestone 1, MPC-only)
Run the stdio server (it idle-waits for an MCP client — Ctrl-C to stop):
```bash
python -m agentic_mpc.mcp_server      # or: agentic-mpc-mcp
```
Exercise it over the real protocol without a GUI:
```bash
python examples/mcp_smoke_client.py   # lists tools, resets R7, advances, reads diagnostics, clips a setpoint
```
Register with a client (Claude Code / Desktop / Cursor) by pointing it at that command as a stdio
server; `.mcp.json` is a local example (gitignored — contains a machine-specific interpreter path).

## 6. Optional: re-run the upstream phase-1.5 experiments
The R2/R7 control-win and rule-based attribution tables (T6/T7) and the v2-prompt diagnosis sweep are
in `analysis/` and `experiments/`; see `analysis/control_regret.py` and `docs/RUNPOD.md`. Those LLM
runs require the Anthropic backend (`.env`) and cost tokens; the Scenario-1/2 artifact above does not.

## Determinism notes
- All sims are seeded (`seed=1` in the Scenario benchmarks); the supervisors are pure-Python rules.
- The plant is a linear FOPDT model: compositions can drift slightly outside [0,1] under aggressive
  control; the economic operating band keeps headroom and the plausibility check uses a
  clear-violation tolerance (≫ the 2×10⁻⁴ sensor-noise std).
