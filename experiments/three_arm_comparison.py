"""Three-arm supervisory comparison on Wood-Berry (same plant, same MPC, same 5 cases):

  Arm 1  MPC only                  -- regulatory MPC, no supervisor, no set_target.
  Arm 2  MCP + rules               -- DiagnosticSupervisor over the MCP diagnostics (deterministic).
  Arm 3  LLM + MCP + guardrails     -- Claude reads the SAME diagnostics and PROPOSES one of
                                       {HOLD, VETO_HOLD, PROPOSE_SETPOINT, ESCALATE}; the deterministic
                                       DiagnosticSupervisor is the VALIDATOR / source of truth. If the
                                       LLM disagrees on action, the canonical decision wins. On
                                       PROPOSE_SETPOINT with action agreement, the LLM's (xD_sp, xB_sp)
                                       drive set_target when inside the safety envelope; otherwise rules'
                                       proposed_targets are applied (and always clipped by the envelope).

The MPC is identical in all arms. The claim is NOT "LLM beats rules" -- it is "the LLM obeys the
guardrails": with the measurement-validity principle in its prompt it should MATCH the rules, and
where it doesn't, the validator wins. Reports one table + a match rate.

Arms 1-2 are free/deterministic. Arm 3 calls Claude (Anthropic) -- one call per case, cached to
outputs/comparison/llm_responses.json (re-used unless --refresh). Run:
    AGENTIC_MPC_BACKEND=anthropic python experiments/three_arm_comparison.py [--no-llm] [--refresh]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import warnings

from agentic_mpc.agent.llm_config import LLM_CONFIG, make_client
from agentic_mpc.mcp_sandbox import MPCSandbox
from agentic_mpc.scenario2_agent import DiagnosticSupervisor
from agentic_mpc.supervisory_apply import apply_supervisory_setpoint

warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).parent / "outputs" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

T_EVENT, T_DECIDE, SEED = 100, 130, 1          # inject at 100; capture the decision snapshot at 130
INJECT = {
    "nominal":        lambda sb: None,
    "sensor_fault":   lambda sb: sb.plant.set_sensor_bias({"xD": 0.05}),
    "coupled_load":   lambda sb: sb.plant.set_disturbance(output_bias={"xD": -0.02, "xB": 0.012}),
    "ambiguous_load": lambda sb: sb.plant.set_disturbance(output_bias={"xD": -0.03}),
}

LLM_SYSTEM = (
    "You are a diagnostic supervisor sitting ABOVE a Wood-Berry distillation MPC. You see only "
    "diagnostics; you CANNOT command reflux R or steam S. Each cycle choose EXACTLY ONE action:\n"
    "- HOLD: tracking is normal (innovation within noise); do nothing.\n"
    "- VETO_HOLD: a reading is physically impossible (a composition must lie in [0,1]) -> an "
    "analyzer/sensor gross error; hold and do NOT chase it with setpoints.\n"
    "- PROPOSE_SETPOINT: a real process disturbance shows in BOTH compositions (both innovations "
    "biased); recommend a bounded new (xD,xB) setpoint.\n"
    "- ESCALATE: sustained mismatch isolated to ONE composition while the other and the inputs are "
    "quiet -> a real load and an in-range sensor bias are indistinguishable from telemetry alone; "
    "request a corroborating check rather than guessing.\n"
    "Nominal |innovation| ~1e-5; a value above ~5e-4 indicates mismatch. Respond ONLY with JSON: "
    '{"action": "HOLD|VETO_HOLD|PROPOSE_SETPOINT|ESCALATE", "xD_sp": <float, only for '
    'PROPOSE_SETPOINT>, "xB_sp": <float, only for PROPOSE_SETPOINT>, "rationale": "<one sentence>"}.'
)


def _user_msg(diag, snap, t) -> str:
    xd = snap["history"]["y"]["xD"]; xb = snap["history"]["y"]["xB"]
    im, off = diag["innovation_mean"], diag["steady_state_offset"]
    return (f"Diagnostics at t={t} min:\n"
            f"- innovation_mean: xD={im['xD']:+.2e}, xB={im['xB']:+.2e}\n"
            f"- steady_state_offset (y - setpoint): xD={off['xD']:+.4f}, xB={off['xB']:+.4f}\n"
            f"- recent measured xD range: [{min(xd):.4f}, {max(xd):.4f}]; "
            f"xB range: [{min(xb):.4f}, {max(xb):.4f}]\n"
            f"- current y: xD={snap['y']['xD']:.4f}, xB={snap['y']['xB']:.4f}; "
            f"inputs R={snap['u']['R']:.3f}, S={snap['u']['S']:.3f}\n"
            f"- active_constraints: {diag['active_constraints']}\n"
            f"Choose exactly one action as JSON.")


def _parse(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"action": "PARSE_ERROR", "rationale": text[:120]}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "PARSE_ERROR", "rationale": text[:120]}
    d["action"] = str(d.get("action", "PARSE_ERROR")).strip().upper()
    return d


def reconcile(llm_action: str, canonical_action: str) -> dict:
    """The guardrail: rules win on action class; see supervisory_apply.resolve_setpoint_targets for SPs."""
    from agentic_mpc.supervisory_apply import reconcile as _reconcile
    return _reconcile(llm_action, canonical_action)


def _apply_rules_setpoint(sb, canonical) -> dict | None:
    if canonical.action != "PROPOSE_SETPOINT" or canonical.proposed_targets is None:
        return None
    return apply_supervisory_setpoint(sb, canonical.action, canonical, llm_proposal=None)


def _apply_llm_setpoint(sb, canonical, llm_proposal, rec) -> dict | None:
    return apply_supervisory_setpoint(sb, rec["final_action"], canonical, llm_proposal)


def llm_propose(client, model, diag, snap, t) -> dict:
    resp = client.chat.completions.create(
        model=model, temperature=0.1, max_tokens=400,
        messages=[{"role": "system", "content": LLM_SYSTEM},
                  {"role": "user", "content": _user_msg(diag, snap, t)}])
    return _parse(resp.choices[0].message.content or "")


def snapshot_at_decision(name):
    """Run the shared MPC-only trajectory to the decision point; return (sandbox, diag, snap)."""
    sb = MPCSandbox(seed=SEED)
    injected = False
    t = 0
    while t < T_DECIDE:
        sb.advance(5); t += 5
        if not injected and t >= T_EVENT:
            INJECT[name](sb); injected = True
    return sb, sb.get_mpc_diagnostics(), sb.get_plant_snapshot()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="skip the Claude arm (arms 1-2 only)")
    ap.add_argument("--refresh", action="store_true", help="ignore cached LLM responses")
    args = ap.parse_args()

    cache_path = OUT / "llm_responses.json"
    cache = json.loads(cache_path.read_text()) if (cache_path.exists() and not args.refresh) else {}
    client = None
    if not args.no_llm:
        client = make_client(LLM_CONFIG)

    rows = []
    for name in INJECT:
        sb_mpc, diag, snap = snapshot_at_decision(name)
        canonical = DiagnosticSupervisor().assess(diag, snap)        # the VALIDATOR / source of truth
        true_xD = float(sb_mpc.plant.last_true_output()[0])          # MPC-only: real state at decision

        sb_rules, _, _ = snapshot_at_decision(name)
        rules_apply = _apply_rules_setpoint(sb_rules, canonical)

        row = {"case": name,
               "mpc_only": {"innov_xD": diag["innovation_mean"]["xD"],
                            "offset_xD": diag["steady_state_offset"]["xD"],
                            "measured_xD": snap["y"]["xD"], "true_xD": true_xD,
                            "supervisor": "none"},
               "rules_action": canonical.action, "rules_state": canonical.state,
               "rules_setpoint": (rules_apply["applied_targets"] if rules_apply else None),
               "rules_target_source": (rules_apply["target_source"] if rules_apply else None)}
        if not args.no_llm:
            sb_llm, _, _ = snapshot_at_decision(name)
            if name in cache and not args.refresh:
                prop = cache[name]
            else:
                prop = llm_propose(client, LLM_CONFIG.model, diag, snap, T_DECIDE)
                cache[name] = prop
            llm_action = prop.get("action", "ERROR")
            rec = reconcile(llm_action, canonical.action)    # validator wins on action class
            llm_apply = _apply_llm_setpoint(sb_llm, canonical, prop, rec)
            row["llm_action"] = llm_action
            row["llm_rationale"] = prop.get("rationale", "")
            row["match"] = rec["match"]
            row["final_action"] = rec["final_action"]
            row["llm_overridden"] = rec["overridden"]
            row["llm_setpoint"] = (llm_apply["applied_targets"] if llm_apply else None)
            row["llm_target_source"] = (llm_apply["target_source"] if llm_apply else None)
            row["llm_proposed_sp"] = (
                {"xD": prop.get("xD_sp"), "xB": prop.get("xB_sp")}
                if prop.get("xD_sp") is not None and prop.get("xB_sp") is not None else None)
        rows.append(row)
    if not args.no_llm:
        cache_path.write_text(json.dumps(cache, indent=2))

    # out-of-range CLIP row (server-side, identical for every arm)
    sb = MPCSandbox(seed=SEED); sb.advance(30)
    clip = sb.set_target(xD=1.2, xB=0.004, rationale="comparison: out-of-range")
    oor = {"clipped": clip["clipped_by_safety"], "applied_xD": clip["applied_targets"]["xD"]}

    _print_and_write(rows, oor, no_llm=args.no_llm)


def _print_and_write(rows, oor, no_llm) -> None:
    hdr = f"{'case':<15}{'MPC-only (true xD)':<22}{'MCP+rules':<20}"
    if not no_llm:
        hdr += f"{'LLM proposal':<18}{'match?':<8}"
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    matches = total = 0
    for r in rows:
        m = r["mpc_only"]
        line = (f"{r['case']:<15}{('tracks; true xD=%.3f' % m['true_xD']):<22}"
                f"{r['rules_action']:<20}")
        if not no_llm:
            total += 1; matches += int(r["match"])
            line += f"{r['llm_action']:<18}{('OK' if r['match'] else 'OVERRIDE'):<8}"
        print(line)
    print(f"{'out_of_range':<15}{'N/A (clip test)':<22}{'CLIP->%.2f' % oor['applied_xD']:<20}"
          + ("" if no_llm else f"{'CLIP->%.2f' % oor['applied_xD']:<18}{'OK':<8}"))
    print("=" * len(hdr))
    if not no_llm:
        print(f"LLM-vs-rules match rate: {matches}/{total}  ({100*matches/max(total,1):.0f}%)  "
              f"| overrides: {total - matches}  (validator always wins)")
    payload = {"rows": rows, "out_of_range": oor,
               "match_rate": (None if no_llm else f"{matches}/{total}")}
    (OUT / "comparison_table.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {OUT / 'comparison_table.json'}")


if __name__ == "__main__":
    main()
