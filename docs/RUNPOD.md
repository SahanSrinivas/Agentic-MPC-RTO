# Phase 1.5 — Project Guide & RunPod Reproduction

This document is meant to be read once, start to finish, by anyone opening the repo — a
reviewer, a co-author, or future-you in three months. It explains **what** we are building,
**why**, and **how to reproduce** the Phase-1.5 experiment matrix on a RunPod GPU pod. Plain
English where possible; technical detail where it matters. Every value below reflects what the
code in this repository actually does (verified against `agentic_mpc/` and `experiments/`).

---

## 1. What this project is

We are building an **autonomous LLM-based supervisory agent** that sits *above* the classical
control stack (MPC + RTO) of a benchmark distillation column and makes closed-loop supervisory
decisions on its own. The benchmark plant is the **Wood-Berry** 2×2 methanol–water column — a
standard process-control testbed with strong loop interaction and transport delays. Industry
today is adjacent but more cautious: AspenTech's AVA, Honeywell's Experion Operations
Assistant, and Yokogawa's FKDPP (reinforcement-learning control) are, at the *supervisory*
layer, **advisory-only** — they suggest, a human decides. We are testing something a step
beyond: an agent with bounded *authority* to act in the loop (retune setpoints, trigger a
re-optimization), with a deterministic safety projector as the backstop.

The **Phase-1 paper goal** is narrow and defensible: show that the agent adds *measurable*
value over strong non-LLM baselines across **seven representative plant-event scenarios**
(R1–R7), and characterize *where* it helps and where a simpler rule book or the RTO alone
suffices. We are not claiming the LLM is always better — we are measuring, with rigor, the
cases where autonomous supervisory reasoning pays off (we expect: compound or ambiguous events,
especially the sensor-fault case) versus where lower layers already handle the problem.

---

## 2. The control architecture and data flow

### The 5-layer mental model

Industrial process control is layered, each layer running slower and "thinking bigger" than the
one below:

| Layer | Decides | Cadence | In this repo? |
|---|---|---|---|
| **Planning / scheduling (LP)** | production targets, feed rates | hours–days | context only (not implemented) |
| **RTO** (real-time optimization) | economically-optimal setpoints (xD, xB) | ~tens of min | **yes** — `agentic_mpc/rto/` |
| **MPC** (model predictive control) | manipulated moves (R, S) to track setpoints | ~1 min | **yes** — `controllers/classical_mpc.py` |
| **PID / regulatory** | valve positions | ~seconds | abstracted into the plant |
| **Field** (sensors, actuators, plant) | physical reality | continuous | `plants/wood_berry.py` (FOPDT + noise) |

Each layer exists because the problem decomposes by timescale: economics change slowly (RTO),
disturbances need fast rejection (MPC/PID), and the plant just responds. Our Phase-1 code
implements **RTO → MPC → plant**; the PID/field layers are folded into the first-order-plus-
deadtime (FOPDT) plant model, and the planning layer is context above the RTO.

The **supervisory agent** sits above all of this. Its *authority* is deliberately bounded to two
actions — `trigger_rto_run` (force an economic re-optimization) and `update_mpc_target` (nudge
the MPC's setpoints) — and **every action is clipped by a deterministic safety projector**
(`BoxSafetyEnvelope`: xD ∈ [0.90, 0.99], xB ∈ [0.001, 0.05]) before it reaches the MPC. The agent
cannot drive the plant outside a physically sane envelope no matter what it decides.

### The closed-loop cycle, step by step

1. **RTO solves the economic optimization** (a CasADi/IPOPT nonlinear program: a bilinear
   material-balance profit minus an ln-separation-factor reboiler-duty cost) to produce target
   compositions **(xD\*, xB\*)**. It runs on a **60-minute** cadence. (Why 60, same as the agent?
   So the agent never reads *stale* RTO data when it cycles — RTO and supervisor are phase-aligned.)
2. **RTO hands the setpoints to the MPC** via `controller.set_targets({"xD": xD*, "xB": xB*}, rationale)`.
3. **MPC runs every 1 minute**, solving a condensed quadratic program (scipy SLSQP) to drive the
   plant toward those setpoints using **R** (reflux) and **S** (steam), respecting per-step move
   limits |Δu| ≤ 0.5 and input bounds R, S ∈ [0.5, 3.0]. State is tracked by a steady-state
   Kalman filter, whose **innovation** (measured − one-step-ahead prediction) is the model-plant
   mismatch signal the supervisor reads.
4. **The plant responds** with FOPDT dynamics: time constants τ ≈ 10.9–21.0 min, transport delays
   1–7 min per channel, plus zero-mean Gaussian sensor noise σ = 2e-4 on each output.
5. **Modifier adaptation (MA)** closes the model-plant gap the nominal RTO can't: each RTO cycle
   it measures the (quasi-steady) plant economics, computes the zeroth-order gap **ε** (plant −
   model value) and the first-order gap **λ** (plant − model gradient, via a Broyden estimate),
   filters them, and adds them to the RTO model. A **steady-state detector** (settled when
   ‖y(t) − y(t−τ)‖ < 3σ over τ ≈ 105 min) reports settling status to the supervisor and gates the
   rule-based "sustained offset" rule, so the supervisor can reason about *"the RTO commanded a
   target N minutes ago — has the plant settled?"*
6. **The cycle repeats** with corrected modifiers, so the RTO converges toward the *true* plant
   optimum despite a structurally mismatched model.

The **agent runs every 60 simulated minutes**. It reads the loop through four tools —
`get_process_state`, `get_mpc_health`, `get_economic_context`, `get_rto_status` — and may act
through `trigger_rto_run` or `update_mpc_target`. Every action passes through the safety
projector before reaching the MPC.

---

## 3. The seven scenarios (R1–R7) — what each tests and why

All scenarios share one operating envelope, so any difference in outcome is attributable to the
injected event, not to different initial conditions. The shared baseline is documented first.

### Shared baseline (every scenario starts here)

| Parameter | Value | Notes |
|---|---|---|
| Feed flow F | 1.0 (normalized) | constant unless a scenario changes it |
| Feed composition z_F | 0.5 (50% light) | constant unless noted |
| Nominal setpoint xD\* | 0.96 | distillate purity at the unperturbed economic optimum |
| Nominal setpoint xB\* | 0.005 | bottoms impurity at the unperturbed economic optimum |
| Distillate price p_D | 29.3513 | calibrated so the optimum is *interior* at xD = 0.96 |
| Bottoms credit p_B | 0.3 | below distillate, so recovery is worth something |
| Steam cost c_S | 0.1306 | calibrated baseline |
| Off-spec penalty p_pen | 2.0 | penalty on distillate impurity |
| Max bottoms impurity xB_max | 0.05 | initial spec; tightened only in R5 |
| Distillate demand cap D_max | unconstrained | a cap only in R4 |
| MPC period | 1 min | QP each tick; Δu and bounds respected |
| RTO period (scheduled) | 60 min | MA modifiers updated each solve |
| Agent / supervisor period | 60 min | when the supervisor reads state and decides |
| Plant FOPDT τ | 10.9–21.0 min | xD channel faster, xB slower |
| Plant FOPDT delay | 1–7 min | per channel |
| Sensor noise σ | 2e-4 | Gaussian, zero-mean, each output |
| Run length | 240 min | 4 hours of plant time |

The supervisors are **blind to the event schedule** — they must infer events from observations.
Events fire at **t = 100 min** for the abrupt scenarios (R2–R6) so plots align; R1 is a ramp and
R7 is a sustained step.

### Scenario-by-scenario detail

**R1 — Slow feed drift** *(model-mismatch, slow).* A linear xD output bias ramps from 0 to
**−0.03** between **t = 50 and t = 250 min**. (The negative sign is the load-disturbance
convention — the drift makes xD read lower than the model expects.) Physically: gradual feed-
composition drift or slow upstream change the RTO model doesn't know about; MA's modifier slowly
absorbs the gap. **Tests** MA's ability to track slow degradation autonomously and the
supervisor's *restraint*. **Expected:** the supervisor mostly holds; MA handles it; setpoints
drift slightly as modifiers update.

**R2 — Efficiency loss** *(model-mismatch, abrupt).* At **t = 100 min**, two simultaneous steps:
the R→xD plant gain drops **10%** (the equipment fault) and an xD output bias steps to **−0.015**
(the load term that carries the RTO-detectable signal). Physically: a fouled tray or undersized
reflux pump. **Tests** abrupt-vs-gradual discrimination; innovation grows fast and the bias is
**negative** — which is exactly why the rule-based supervisor must use `abs()` on the innovation.
**Expected:** detect within 1–2 cycles; possibly trigger an early RTO or nudge the target.

**R3 — Steam-cost spike** *(economic-shift).* At **t = 100 min**, steam cost c_S is multiplied by
**2.0** (0.1306 → 0.2612) and held. Physically: an energy-market event upstream; the optimum
should move toward lower purity (less reboiler duty). **Tests** awareness of *economic* context
versus *process* state — the plant dynamics don't change, only the objective. **Expected:** detect
via `get_economic_context`, trigger an early RTO; the new optimum sits at a lower xD (~0.93–0.94).

**R4 — Demand shift** *(economic-shift, constraint-driven).* At **t = 100 min**, a hard
distillate-demand cap **D ≤ D_max** (≈ 0.50, just below the unconstrained optimum's throughput) is
imposed. Physically: a downstream sales cap; less throughput frees reflux for higher purity.
**Tests** constraint-driven re-optimization. **Expected:** the RTO shifts toward *higher* xD
(~0.98–0.99) with the demand constraint active.

**R5 — Spec tightening** *(RTO infeasibility).* At **t = 100 min**, xB_max is ratcheted from 0.05
down to **0.0008**. This value is **deliberately sub-achievable**: the linear-model operating
envelope bottoms out near xB ≈ 0.0011, so 0.0008 is *genuinely infeasible* — the RTO returns no
solution. (We use 0.0008, not 0.003: 0.003 is feasible and the RTO would simply pin xB to that
boundary, which would *not* test infeasibility handling, R5's whole purpose.) Physically: a
customer/regulatory spec the current regime cannot meet. **Tests** graceful failure-mode handling.
**Expected:** the supervisor diagnoses infeasibility (RTO status `converged: False`) and reports
rather than commanding an out-of-spec target; a naive rule book may escalate awkwardly.

**R6 — Analyzer gross error** *(sensor anomaly).* A fixed **+0.05** bias is added to the xD
*analyzer reading* over **t = 100 → 140 min** (a 40-minute window), then cleared. The true plant
composition is unchanged — only the sensor lies. Physically: a composition-analyzer fault (GC
drift, contaminated sample line). **Tests** the agent's hardest case: distinguishing a *sensor*
fault from a *real* process shift by cross-referencing multiple tools — the strongest case for LLM
reasoning over a single-signal rule. **Expected:** a good supervisor notices the innovation/state
pattern is inconsistent with a real shift (e.g., R barely moves) and refrains from over-reacting;
rule-based supervisors will likely fire Rule 1 on the innovation and over-react.

**R7 — Load disturbance** *(MPC tracking degradation).* A sustained **−0.03** xD output (load)
disturbance steps in at **t = 100 min** and persists. Physically: an unmodeled load the MPC must
reject; tracking error and innovation grow. **Tests** layer interaction — the MPC sees rising
innovation, the RTO a shifted optimum, and the supervisor must decide whether to retarget,
re-optimize, or hold. (Note: this is a *step* load disturbance — a sustained, not oscillatory,
event. A sinusoidal feed-composition variation is a possible Phase-2 enhancement; the validated
Phase-1 mechanism is the step, which already exercises the MPC-tracking-degradation claim.)
**Expected:** detect the biased innovation / rising ISE; MA or a target nudge restores tracking.

---

## 4. The supervisor variants we compare

Four supervisor "kinds" run on the same seven scenarios:

- **Agentic (LLM-based)** — `qwen3:30b` reads state via the four tools, reasons, and decides via
  tool calls. (Why `qwen3:30b`? It is a Mixture-of-Experts model: ~30B parameters but only a
  fraction active per token, so it is far faster to serve on one GPU than a dense 32B while still
  reasoning well over tool outputs. Temperature **0.1** — low, for consistent, near-deterministic
  supervisory decisions.)
- **Rule-based naive** — pure-Python `if/else` with cited thresholds. Rule 1: |innovation| > 5e-4
  → `trigger_rto_run`. Rule 2 (sustained offset, gated on the steady-state detector) →
  `update_mpc_target` to the **current measured** composition. Rule 3 (RTO stale > 90 min AND
  prices changed) → `trigger_rto_run`. "Naive" because Rule 2 *accepts* the drift (retargets to
  where the plant is) rather than re-optimizing.
- **Rule-based smart** — same Rules 1 and 3, but Rule 2 calls **`trigger_rto_run`** instead of
  accepting the drift. A more sophisticated rule book.
- **Baseline (no supervisor)** — scheduled RTO + MPC only, no supervisory intervention. The
  "what the RTO does on its own" reference.

Each supervisor runs against two **RTO variants**:

- **MA** — classical modifier adaptation with Broyden gradient estimation.
- **MA-GP** — modifier adaptation with a Gaussian-Process surrogate of the plant-model gap
  (Matérn-5/2 kernel + white noise, trust-region Bayesian optimization, Expected-Improvement
  acquisition; del Rio Chanona et al. 2021).

**Full matrix: 4 supervisors × 2 RTO variants × 7 scenarios = 56 runs.**

---

## 5. What we're comparing and why

The paper's claims rest on four comparisons:

1. **Agentic vs Baseline** (all scenarios): does the LLM agent add measurable value over
   scheduled-RTO-only?
2. **Agentic vs Rule-based** (all scenarios): can an LLM outperform a thoughtful rule book — and
   *where* specifically?
3. **MA vs MA-GP** (within each supervisor): does the GP surrogate help, and where?
4. **Rule-based-naive vs Rule-based-smart**: how much does rule sophistication matter?

Per-scenario expectations going in:

- **R1, R2 (slow drift, fouling):** MA should absorb most of it; all supervisors should be quiet.
  Tests appropriate *restraint* — acting when nothing is wrong is a failure mode too.
- **R3, R4 (economic):** all supervisors should respond; the differences are in *timing* and
  *depth* of response. Tests event-driven supervisory action.
- **R5 (infeasibility):** the agent should diagnose-and-handle gracefully; rule-based may fail or
  escalate awkwardly. Tests failure-mode handling.
- **R6 (sensor):** the agent must distinguish a bad sensor from a real shift by cross-reading
  multiple tools — the LLM's strongest case; rule-based likely over-reacts.
- **R7 (MPC degradation):** all supervisors should detect controller stress. Tests layer
  interaction.

---

## 6. Pod setup recap

These steps assume a fresh RunPod pod (A100 PCIe class) with a `/workspace` network volume.

```bash
# 1. zstd (Ollama model layers are zstd-compressed)
apt-get install -y zstd

# 2. Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 3. Ollama env (point at the network volume; smaller context to fit VRAM; keep the model warm)
export OLLAMA_MODELS=/workspace/ollama_data     # models live on /workspace, not the container disk
export OLLAMA_CONTEXT_LENGTH=8192               # plenty for our tool I/O; saves VRAM
export OLLAMA_KEEP_ALIVE=30m                    # don't unload between agent cycles
export OLLAMA_FLASH_ATTENTION=true

# 4. start the server (backgrounded, logged)
nohup ollama serve > /workspace/ollama.log 2>&1 &

# 5. pull the model
ollama pull qwen3:30b

# 6. verify
ollama list
curl -s http://localhost:11434/api/tags | head -c 300

# 7. clone the repo onto the network volume
cd /workspace && git clone git@github.com:SahanSrinivas/Agentic-MPC-RTO.git
cd Agentic-MPC-RTO

# 8. Python venv on /root (NOT /workspace -- a venv is thousands of small files; /workspace is a
#    MooseFS network FS with a quota and slow small-file I/O, which makes installs crawl and can
#    blow the quota. /root is fast local container disk with ~17 GB free.)
python3 -m venv /root/venv && source /root/venv/bin/activate

# 9. pip cache also on /root (same reason)
export PIP_CACHE_DIR=/root/.pip-cache

# 10. install + verify
pip install --upgrade pip && pip install -e . && pip install pytest
pytest tests/ -q          # expect: 60 passed
```

Make the environment permanent so a reconnect "just works":

```bash
cat >> ~/.bashrc <<'EOF'
source /root/venv/bin/activate
export OLLAMA_MODELS=/workspace/ollama_data
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_FLASH_ATTENTION=true
export PIP_CACHE_DIR=/root/.pip-cache
export PATH=/workspace/.npm-global/bin:$PATH
EOF
```

**Disk hierarchy on the pod:**

| Mount | Size | Use |
|---|---|---|
| `/workspace` | MooseFS network, ~20 GB quota | model + repo + outputs (persists across pod restarts) |
| `/root` | container disk, ~17 GB free | Python venv + pip cache (fast, ephemeral) |
| `/dev/shm` | high-speed scratch, ~117 GB | unused but available |

---

## 7. Clean up stale pre-fix outputs

An earlier batch ran *before* the output-directory fix, when every configuration wrote to the same
`phase1_5/qwen3_30b/<scenario>/` path and overwrote each other. Remove those stale artifacts on the
pod before the re-run so the tree is uniform:

```bash
cd /workspace/Agentic-MPC-RTO
rm -rf experiments/outputs/phase1_5/qwen3_30b/R1
rm -rf experiments/outputs/phase1_5/qwen3_30b/R2
rm -rf experiments/outputs/phase1_5/qwen3_30b/R3
rm -rf experiments/outputs/phase1_5/qwen3_30b/R4
rm -f  experiments/outputs/batch_*.log
rm -f  experiments/outputs/batch_master.log
```

The fix writes config-separated paths — `phase1_5/<model>/<supervisor>_<rto>/<scenario>/` for LLM
and baseline, `phase1_5/rule_based_<variant>_<rto>/<scenario>/` for rule-based — so configurations
no longer collide.

---

## 8. Full run matrix

CLI flags verified against `experiments/_phase1_5_runner.py`:
`--model`, `--rto {nominal,ma,ma-gp}`, `--supervisor {llm,rule-based-naive,rule-based-smart}`,
`--no-agent`, `--t-end`, `--rto-interval`, `--agent-interval`, `--seed` (default 42).
`run_all_phase_1_5.sh <MODEL> [extra flags...]` passes the extra flags through to all seven scripts.

| # | Command | Supervisor | RTO | Output path | Est. |
|---|---|---|---|---|---|
| 1 | `bash experiments/run_all_phase_1_5.sh qwen3:30b --rto ma` | agentic | MA | `phase1_5/qwen3_30b/agentic_ma/<scenario>/` | ~85 min |
| 2 | `bash experiments/run_all_phase_1_5.sh qwen3:30b --rto ma-gp` | agentic | MA-GP | `phase1_5/qwen3_30b/agentic_ma-gp/<scenario>/` | ~85 min |
| 3 | `bash experiments/run_all_phase_1_5.sh qwen3:30b --no-agent --rto ma` | baseline | MA | `phase1_5/qwen3_30b/baseline_ma/<scenario>/` | ~10 min |
| 4 | `bash experiments/run_all_phase_1_5.sh qwen3:30b --no-agent --rto ma-gp` | baseline | MA-GP | `phase1_5/qwen3_30b/baseline_ma-gp/<scenario>/` | ~10 min |
| 5 | `bash experiments/run_all_phase_1_5.sh qwen3:30b --supervisor rule-based-naive --rto ma` | rule-naive | MA | `phase1_5/rule_based_naive_ma/<scenario>/` | ~5 min |
| 6 | `bash experiments/run_all_phase_1_5.sh qwen3:30b --supervisor rule-based-naive --rto ma-gp` | rule-naive | MA-GP | `phase1_5/rule_based_naive_ma-gp/<scenario>/` | ~5 min |
| 7 | `bash experiments/run_all_phase_1_5.sh qwen3:30b --supervisor rule-based-smart --rto ma` | rule-smart | MA | `phase1_5/rule_based_smart_ma/<scenario>/` | ~5 min |
| 8 | `bash experiments/run_all_phase_1_5.sh qwen3:30b --supervisor rule-based-smart --rto ma-gp` | rule-smart | MA-GP | `phase1_5/rule_based_smart_ma-gp/<scenario>/` | ~5 min |

Notes: for the rule-based rows the `qwen3:30b` argument is ignored (no LLM is called), and the
output path carries no model prefix. The two **agentic** runs dominate wall-clock; baselines and
rule-based runs are fast because they make no LLM calls.

**Total wall-clock ≈ 3.5 hours** (agentic rows dominate). **Total compute ≈ $5** on an A100 PCIe
at ~$1.39/hr.

---

## 9. Launching the batch in tmux

Run inside `tmux` so the batch survives an SSH disconnect.

```bash
tmux new -s phase1_5_v2
cd /workspace/Agentic-MPC-RTO
source /root/venv/bin/activate

echo "=== START $(date) ===" | tee experiments/outputs/batch_master.log

for cmd in \
  "bash experiments/run_all_phase_1_5.sh qwen3:30b --rto ma" \
  "bash experiments/run_all_phase_1_5.sh qwen3:30b --rto ma-gp" \
  "bash experiments/run_all_phase_1_5.sh qwen3:30b --no-agent --rto ma" \
  "bash experiments/run_all_phase_1_5.sh qwen3:30b --no-agent --rto ma-gp" \
  "bash experiments/run_all_phase_1_5.sh qwen3:30b --supervisor rule-based-naive --rto ma" \
  "bash experiments/run_all_phase_1_5.sh qwen3:30b --supervisor rule-based-naive --rto ma-gp" \
  "bash experiments/run_all_phase_1_5.sh qwen3:30b --supervisor rule-based-smart --rto ma" \
  "bash experiments/run_all_phase_1_5.sh qwen3:30b --supervisor rule-based-smart --rto ma-gp"; do
    echo "--- $cmd ---" | tee -a experiments/outputs/batch_master.log
    eval "$cmd" 2>&1 | tee -a experiments/outputs/batch_master.log
done

echo "=== DONE $(date) ===" | tee -a experiments/outputs/batch_master.log
```

Detach with **Ctrl-B** then **D**. Reattach later with `tmux attach -t phase1_5_v2`.

---

## 10. Recovery if something fails mid-batch

Which scenarios completed for a config (each completed scenario has its own directory):

```bash
ls experiments/outputs/phase1_5/qwen3_30b/agentic_ma/
```

Resume a single failed scenario:

```bash
python experiments/r3_feed_price_spike.py --model qwen3:30b --rto ma
```

Resume an entire interrupted config (re-runs all seven; completed ones are overwritten — runs are
seeded, so this is deterministic):

```bash
bash experiments/run_all_phase_1_5.sh qwen3:30b --rto ma
```

---

## 11. After the batch — paper-prep next steps

```bash
# 1. commit all outputs
git add experiments/outputs/
git commit -m "Full Phase 1.5 batch: 4 supervisors x 2 RTO x 7 scenarios"
git push origin main
```

2. **Stop the pod** in the RunPod UI — the compute charge stops; the network volume keeps
   `qwen3:30b` and the repo for next time.
3. **On the laptop:** `git pull origin main`, analyze the results, and draft the comparison figures
   (Agentic vs Baseline vs Rule-based × MA/MA-GP, per scenario).

---

### Reproducibility note

All runs take `--seed` (default **42**), which propagates to numpy, the plant sensor noise, the RTO
multi-start, and the LLM request seed. Rule-based and baseline runs are fully deterministic; LLM
runs are as deterministic as `qwen3:30b` + seeded sampling allow. The single backend switch (Ollama
↔ Claude) lives in `agentic_mpc/agent/llm_config.py` — nothing else reads the model config.
