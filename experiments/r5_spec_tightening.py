"""Phase 1.5 scenario R5 -- bottoms-spec tightening below achievable (RTO infeasibility).

Run:  python experiments/r5_spec_tightening.py --model qwen3:4b [--rto ma|ma-gp|nominal] [--no-agent]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _phase1_5_runner import main

if __name__ == "__main__":
    main("R5")
