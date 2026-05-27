# Agentic-MPC — Phase 1 Vertical Slice (Wood-Berry)

An LLM supervisory agent sitting above a classical Model Predictive Controller that
controls a simulated **Wood-Berry binary distillation column**. The agent observes a
real running closed loop and makes supervisory decisions (retuning setpoints) that
propagate to the MPC and back to the plant.

This is **Phase 1** of a comparative case study (classical MPC, RNN-MPC,
classical+agent, RNN+agent across two processes). Phase 1 builds the first real
vertical slice for one process (Wood-Berry) with a classical MPC and the agent.

## The contract: universal interfaces

Everything talks through the abstract base classes in
[`agentic_mpc/interfaces.py`](agentic_mpc/interfaces.py):

- **`Plant`** — `step(u, dt) -> y`, `get_state()`, `reset()`, `metadata`.
- **`Controller`** — `compute_control(y, y_sp) -> u`, `set_targets()`,
  `set_constraints()`, `get_health()`, `metadata`.
- **`SafetyEnvelope`** — `project(proposed_action) -> (safe_action, was_violated)`.

The agent only ever talks to these ABCs — never a concrete plant/controller. That is
the "one algorithm, multiple processes" claim made concrete: swapping the plant or
controller requires no change to the agent.

## Layout

```
agentic_mpc/
  interfaces.py        # ABCs: Plant, Controller, SafetyEnvelope, Optimizer
  metrics.py           # ISE / IAE / settling-time helpers
  plants/wood_berry.py # 2x2 FOPDT Wood-Berry column
  controllers/classical_mpc.py  # condensed-QP linear MPC
  safety.py            # concrete BoxSafetyEnvelope
  agent/               # supervisor, tools, validation, llm_config
  rto/                 # Phase 1.5: economics, WoodBerryRTO (nominal NLP),
                       #   modifier_adaptation (MA + MA-GP), broyden, rto_mpc_loop
  scenarios.py         # Phase 1.5: 7 disturbance scenarios (R1-R7)
experiments/           # phase1_prbs_validation.py, phase1_end_to_end.py,
                       #   r1..r7 scenario runners, run_all_phase_1_5.sh
tests/                 # interfaces, plant, MPC, agent, RTO, comparators, scenarios
```

## Setup

Uses the existing virtual environment at `C:\Pegasus-Sample\venv_name`
(Python 3.12). Install the package editable:

```powershell
C:\Pegasus-Sample\venv_name\Scripts\python.exe -m pip install -e .
```

Dependencies: numpy, scipy, openai, pydantic, matplotlib, casadi, scikit-learn (+ pytest).
The MPC uses scipy SLSQP (matching the prior KIRA MPC); the Phase-1.5 RTO uses CasADi/IPOPT
(shared comparator solver path) and scikit-learn (MA-GP surrogate).

## LLM backend

The agent uses a local **Ollama** server (model `qwen3:4b`, base URL
`http://localhost:11434/v1`) for the paper, swappable to Claude Sonnet in production
via a single config change in `agentic_mpc/agent/llm_config.py`.

## Tests

```powershell
C:\Pegasus-Sample\venv_name\Scripts\python.exe -m pytest tests/
```

## Phase 1.5 — RTO layer

An economic real-time-optimization (RTO) layer sits above the MPC and picks the
economically-optimal composition setpoints. Three comparators share one CasADi/IPOPT solver
path: nominal NLP (`WoodBerryRTO`), modifier adaptation (`ModifierAdaptation`), and MA-GP
(`MAGaussianProcess`). `RTOMPCLoop` drives the RTO (slow) over the MPC (fast) with explicit
timers and a clean agent handoff log. Run the scenarios (R1–R7):

```powershell
# one scenario (agentic, MA RTO)
C:\Pegasus-Sample\venv_name\Scripts\python.exe experiments/r3_feed_price_spike.py --model qwen3:4b
# baseline (no LLM), MA-GP RTO
... experiments/r7_load_disturbance.py --no-agent --rto ma-gp
# all seven for a model
bash experiments/run_all_phase_1_5.sh qwen3:30b
```

## Status

- [x] Steps 1–6 — interfaces, Wood-Berry plant, classical MPC, PRBS validation, agent refactor
- [x] Step 7 — end-to-end wiring + diagnostics (full closed-loop run deferred to RunPod / qwen3:30b)
- [x] Phase 1.5 — RTO layer (nominal / MA / MA-GP), 7 scenarios, RTO-MPC loop, RTO agent tools

Full LLM scenario runs (R1–R7 with a stronger model) are executed on RunPod; local `qwen3:4b`
is used for wiring verification (it observes but under-reacts with the limited tool set).
