"""Protocol v5: per-run opaque row handles over the compact view store. A new store is written for every run dir (and
every host-side scoring call) with fresh random row handles (r_...) in shuffled order and fresh document ids (d_...);
views carry no task/document/group/series identity; the handle -> task map stays host-private.
anonymize(pack_store, task_ids, dst_store) -> handle_to_tid"""
import random, secrets
import viewstore

IDENTITY_KEYS = ("tid", "task_id", "group_id", "entity_name", "series", "variable")


def anonymize(pack_store, task_ids, dst):
    s = pack_store if isinstance(pack_store, viewstore.Store) else viewstore.Store(pack_store)
    tids = list(task_ids); random.SystemRandom().shuffle(tids)
    key_map = {"r_" + secrets.token_hex(8): t for t in tids}
    refs = {r for t in tids for r in s.meta[t]["document_refs"]}
    doc_map = {r: "d_" + secrets.token_hex(6) for r in refs}
    viewstore.copy_subset(s, dst, key_map, doc_map, drop_keys=IDENTITY_KEYS)
    return key_map
