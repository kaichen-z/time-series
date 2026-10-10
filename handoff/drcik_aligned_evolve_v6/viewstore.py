"""Compact view store (protocol v5). A store directory holds:
  views_meta.json      {key: view without method_forecasts/documents, plus "_fc": {method: [offset, length]}, "document_refs": [...]}
  forecasts.f32        all frozen forecasts as little-endian float32, concatenated
  documents.json       {document_id: {"content": str, "events": [...]}}
materialize(key) returns the ordinary view dict modules expect (method_forecasts as lists, documents inlined as
{document_id, content, events}). Pure python + numpy (memory-mapped), so it also works inside the agent sandbox.
usage: s = Store(dir); for key in s.keys(): view = s.view(key)"""
import json, sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # read-only fallback for agents whose python has no numpy (save/copy_subset need numpy)
    np = None


class Store:
    def __init__(self, d):
        d = Path(d); self.meta = json.load(open(d / "views_meta.json")); self.docs = json.load(open(d / "documents.json"))
        if np is None:
            import array
            self.fc = array.array("f"); self.fc.frombytes((d / "forecasts.f32").read_bytes())
            if sys.byteorder != "little": self.fc.byteswap()
        else:
            self.fc = np.memmap(d / "forecasts.f32", dtype="<f4", mode="r") if (d / "forecasts.f32").stat().st_size else np.zeros(0, "<f4")

    def keys(self): return list(self.meta)

    def view(self, key):
        m = self.meta[key]; v = {k: val for k, val in m.items() if k not in ("_fc", "document_refs")}
        v["method_forecasts"] = {name: [float(x) for x in self.fc[o:o + n]] for name, (o, n) in m["_fc"].items()}
        v["documents"] = [dict(document_id=r, **self.docs[r]) for r in m["document_refs"]]
        return v


def save(d, views, docs):
    """views: {key: view with method_forecasts (lists) and document_refs}; docs: {doc_id: {...}} (only referenced ones kept)."""
    d = Path(d); d.mkdir(parents=True, exist_ok=True); meta, chunks, off, used = {}, [], 0, set()
    for key, v in views.items():
        fcm = {}
        for name, fc in v["method_forecasts"].items():
            a = np.asarray(fc, dtype="<f4"); chunks.append(a); fcm[name] = [off, len(a)]; off += len(a)
        meta[key] = {k: val for k, val in v.items() if k not in ("method_forecasts", "documents")}; meta[key]["_fc"] = fcm
        used.update(v["document_refs"])
    (np.concatenate(chunks) if chunks else np.zeros(0, "<f4")).astype("<f4").tofile(d / "forecasts.f32")
    json.dump(meta, open(d / "views_meta.json", "w")); json.dump({k: docs[k] for k in sorted(used)}, open(d / "documents.json", "w"))


def copy_subset(src, dst, key_map, doc_map, drop_keys=()):
    """Write a new store at dst with views src[old] under new keys (key_map: new -> old), document ids renamed by doc_map
    (old -> new), meta keys in drop_keys removed. Forecast floats are copied at the array level (no float lists)."""
    s = src if isinstance(src, Store) else Store(src); dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    meta, parts, off, docs = {}, [], 0, {}
    for new, old in key_map.items():
        m = s.meta[old]; fcm = {}
        for name, (o, n) in m["_fc"].items():
            parts.append(np.asarray(s.fc[o:o + n])); fcm[name] = [off, n]; off += n
        nm = {k: v for k, v in m.items() if k not in drop_keys and k not in ("_fc", "document_refs")}
        nm["_fc"] = fcm; nm["document_refs"] = [doc_map[r] for r in m["document_refs"]]; meta[new] = nm
        for r in m["document_refs"]: docs[doc_map[r]] = s.docs[r]
    (np.concatenate(parts) if parts else np.zeros(0, "<f4")).astype("<f4").tofile(dst / "forecasts.f32")
    json.dump(meta, open(dst / "views_meta.json", "w")); json.dump(docs, open(dst / "documents.json", "w"))
