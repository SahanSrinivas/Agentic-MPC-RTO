"""Phase 1.5 scenario R7 -- load disturbance stressing MPC tracking (mpc-tracking degradation).

Run:  python experiments/r7_load_disturbance.py --model qwen3:4b [--rto ma|ma-gp|nominal] [--no-agent]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _phase1_5_runner import main

if __name__ == "__main__":
    main("R7")
