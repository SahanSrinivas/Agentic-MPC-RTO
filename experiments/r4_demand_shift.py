"""Phase 1.5 scenario R4 -- product demand shift / distillate-demand cap (economic-shift).

Run:  python experiments/r4_demand_shift.py --model qwen3:4b [--rto ma|ma-gp|nominal] [--no-agent]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _phase1_5_runner import main

if __name__ == "__main__":
    main("R4")
