"""Synchronous-round acceptance for the evaluator daemons (sol56 reconstruction, env SYNC_ROUNDS=1).

Within round r every submission is scored immediately (the agent still gets its visible scores) and compared with the
champion FROZEN at the start of round r. A submission is *eligible* iff
    visible > base_visible + eps_vis   AND   hidden check passes against the frozen base.
Nothing is accepted during the round. When the orchestrator writes RUN/round_close_<r> (all agents of the round have
finished their episode and the queue is empty), the daemon ranks the eligible submissions by
    (visible DESC, hidden DESC, agent id ASC, per-agent submission index ASC)
and the first one becomes the champion of round r+1 (none eligible -> champion unchanged). Arrival order / wall-clock
are never used. The daemon then writes RUN/round_done_<r>.json with the decision."""
import json, time
from pathlib import Path


def init(st, base_visible, base_hidden):
    st.setdefault("round", 1); st.setdefault("eligible", [])
    st.setdefault("base_visible", base_visible); st.setdefault("base_hidden", base_hidden)
    st.setdefault("agent_idx", {})


def next_index(st, agent):
    st["agent_idx"][agent] = st["agent_idx"].get(agent, 0) + 1
    return st["agent_idx"][agent]


def record(st, agent, idx, visible, hidden_scalar, eligible, ref):
    if eligible:
        st["eligible"].append(dict(round=st["round"], agent=agent, idx=idx, visible=visible, hidden=hidden_scalar, ref=ref))


def rank_key(e):
    return (-e["visible"], -e["hidden"], e["agent"], e["idx"])


def pick(eligible):
    return sorted(eligible, key=rank_key)[0] if eligible else None


def maybe_close(run: Path, st, apply_winner, queue_empty: bool):
    """apply_winner(entry) installs the winner as the shared best and returns (new_base_visible, new_base_hidden)."""
    r = st["round"]; flag = run / f"round_close_{r}"
    if not (queue_empty and flag.exists()): return False
    cands = [e for e in st["eligible"] if e["round"] == r]; w = pick(cands)
    if w is not None:
        bv, bh = apply_winner(w); st["base_visible"], st["base_hidden"] = bv, bh
    json.dump(dict(round=r, n_eligible=len(cands), winner=w, ranking=sorted(cands, key=rank_key), t=time.time()),
              open(run / f"round_done_{r}.json", "w"), indent=1)
    st["round"] = r + 1
    return True
