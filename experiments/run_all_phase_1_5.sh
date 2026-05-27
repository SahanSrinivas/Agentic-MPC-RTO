#!/usr/bin/env bash
# Batch-run all Phase-1.5 scenarios R1-R7 for a given model.
#
#   ./experiments/run_all_phase_1_5.sh [MODEL] [extra args passed to each script...]
#
# Examples:
#   ./experiments/run_all_phase_1_5.sh qwen3:30b                 # agentic, MA RTO (defaults)
#   ./experiments/run_all_phase_1_5.sh qwen3:4b --rto ma-gp      # MA-GP RTO
#   ./experiments/run_all_phase_1_5.sh qwen3:4b --no-agent       # baselines only (no LLM)
#
# Outputs land under experiments/outputs/phase1_5/<model>/<scenario>/ (model-name-suffixed).
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
