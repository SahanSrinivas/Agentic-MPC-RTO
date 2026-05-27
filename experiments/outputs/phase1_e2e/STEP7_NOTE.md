# Step 7 (end-to-end supervisory scenario) — status

**Step 7 was validated via wiring tests and short diagnostic runs; full closed-loop run
deferred to RunPod execution with a stronger model.**

What is established:
- The full stack (WoodBerry plant + classical MPC + LLM supervisory agent) is wired through
  the universal interfaces, and the agent→`set_targets`→MPC→plant propagation path is
  unit-tested (`tests/test_agent_tools.py`).
- Live single/short cycles on local `qwen3:4b` confirm the tool-calling round-trip works
  (the agent calls `get_process_state` + `get_mpc_health` and reasons over them).
- **Diagnostic finding:** with the limited Phase-1 tool set, `qwen3:4b` reliably *observes*
  the degradation but does not *act* (issues no `update_mpc_target`) even when the documented
  thresholds are exceeded; it correctly holds when the process is healthy (no false positives).
  This is a model-scale / tool-set-design finding.

Why deferred: a full 300-min, every-10-min closed-loop run is ~60–120 min of local
`qwen3:4b` compute and, per the finding above, shows observation rather than action. The
publishable Step-7 numbers (and the contrast against a stronger model that acts) will be
produced on RunPod with `qwen3:30b`, alongside the Phase-1.5 R1–R7 scenarios.

Artifacts present: `phase1_e2e.png` (from a short diagnostic run); `experiments/
phase1_end_to_end.py` is runnable (`--`-free; configurable horizon / trigger window).
