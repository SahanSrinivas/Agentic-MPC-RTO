#!/usr/bin/env bash
# Batch-run all Phase-1.5 scenarios R1-R7 for a given model.
#
#   ./experiments/run_all_phase_1_5.sh [MODEL] [extra args passed to each script...]
#
# Examples:
#   ./experiments/run_all_phase_1_5.sh qwen3:30b                          # LLM agentic, MA RTO
#   ./experiments/run_all_phase_1_5.sh qwen3:30b --rto ma-gp              # LLM agentic, MA-GP RTO
#   ./experiments/run_all_phase_1_5.sh qwen3:30b --no-agent               # baselines only (no LLM)
#   ./experiments/run_all_phase_1_5.sh _ --supervisor rule-based-naive    # rule-based (no LLM; model arg ignored)
#   ./experiments/run_all_phase_1_5.sh _ --supervisor rule-based-smart    # rule-based smart variant
#
# Outputs are config-separated (no overwrites):
#   phase1_5/<model>/agentic_<rto>/<scenario>      (LLM agentic)
#   phase1_5/<model>/baseline_<rto>/<scenario>     (--no-agent)
#   phase1_5/rule_based_{naive,smart}_<rto>/<scenario>   (--supervisor rule-based-*; no model)
set -euo pipefail

MODEL="${1:-qwen3:4b}"
shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"

SCENARIOS=(
  r1_slow_feed_drift
  r2_efficiency_loss
  r3_feed_price_spike
  r4_demand_shift
  r5_spec_tightening
  r6_analyzer_gross_error
  r7_load_disturbance
)

echo "Phase 1.5 batch | model=${MODEL} | extra args: $*"
for s in "${SCENARIOS[@]}"; do
  echo ">>> ${s}"
  "${PY}" "${HERE}/${s}.py" --model "${MODEL}" "$@"
done
echo "Phase 1.5 batch complete -> experiments/outputs/phase1_5/${MODEL//:/_}/"
