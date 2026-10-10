"""Protocol v3: per-run opaque row handles. Agent-visible views are keyed by fresh random handles (new for every run dir
and every host-side scoring call), shuffled, and carry no task/document/group/series identity; the handle -> task map
stays host-private. anonymize(views_by_tid) -> (views_by_handle, handle_to_tid)."""
import random, secrets

IDENTITY_KEYS = ("tid", "task_id", "group_id", "entity_name", "series", "variable")


def anonymize(views):
    tids = list(views); random.SystemRandom().shuffle(tids)
    out, hmap = {}, {}
    for t in tids:
        h = "r_" + secrets.token_hex(8)
        v = {k: val for k, val in views[t].items() if k not in IDENTITY_KEYS}
        v["documents"] = [dict(d, document_id="d_" + secrets.token_hex(6)) for d in v.get("documents", [])]
        out[h] = v; hmap[h] = t
    return out, hmap
