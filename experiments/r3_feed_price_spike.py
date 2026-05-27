"""Phase 1.5 scenario R3 -- steam/utility price spike (economic-shift).

Run:  python experiments/r3_feed_price_spike.py --model qwen3:4b [--rto ma|ma-gp|nominal] [--no-agent]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _phase1_5_runner import main

if __name__ == "__main__":
    main("R3")
