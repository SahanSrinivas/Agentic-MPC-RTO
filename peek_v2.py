"""Compare R6 cycle-2 (post-sensor-jump) reasoning across v1/v2 prompts and seeds 1-3.

Run from the repo root:  python peek_v2.py
"""
import json
import sys

try:                                            # logs contain emoji/unicode; avoid cp1252 crash
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAULT_WORDS = ["sensor", "analyzer", "instrument", "implausible", "physically", "not a real"]

for seed in [1, 2, 3]:
    for ver, base in [("v2", "claude_sonnet_4_6_promptv2"), ("v1", "claude_sonnet_4_6")]:
        path = f"experiments/outputs/phase1_5/{base}/agentic_ma/R6/seed{seed}/log.json"
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except FileNotFoundError:
            continue
        decisions = d.get("agent_decisions", [])
        if len(decisions) < 2:
            print(f"=== R6 {ver} seed{seed} | <2 cycles ({len(decisions)}); skipping ===\n")
            continue
        c2 = decisions[1]                       # cycle 2, the one right after the t=101 jump
        final = c2.get("final", "")
        acts = [a.get("tool") for a in c2.get("actions", [])
                if a.get("tool") in ("trigger_rto_run", "update_mpc_target")]
        text = final.lower()
        named = [w for w in FAULT_WORDS if w in text]
        print(f"=== R6 {ver} seed{seed} | total actions={d.get('n_agent_actions')} | cycle2 acts={acts} ===")
        print(f"    fault-language hits: {named or 'NONE'}")
        print("    cycle2 final (first 600 chars):")
        print("   ", final[:600].replace(chr(10), " "))
        print()
