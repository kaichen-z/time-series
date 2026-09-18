"""Skill memory: a persistent library of the best evolved controllers (Voyager /
AlphaEvolve program-database style). Each evolution run adds its champion; the next run
loads them as extra seeds, so the system accumulates useful, auditable rules across runs
instead of restarting from fixed seeds every time.

Controllers are stored as JSON (each instruction = {type, fields}), so the library is
human-readable and auditable, and round-trips exactly back to runnable controllers.
"""
from __future__ import annotations

import json
from pathlib import Path

from .controller import Controller, INSTRUCTIONS

_BY_NAME = {cls.__name__: cls for cls in INSTRUCTIONS}


def _instr_to_payload(step) -> dict:
    fields = {k: (list(v) if isinstance(v, tuple) else v) for k, v in step.__dict__.items()}
    return {"type": type(step).__name__, "fields": fields}


def _instr_from_payload(p: dict):
    cls = _BY_NAME[p["type"]]
    fields = {k: (tuple(v) if isinstance(v, list) else v) for k, v in p.get("fields", {}).items()}
    return cls(**fields)


def controller_to_payload(c: Controller) -> dict:
    return {"name": c.name, "steps": [_instr_to_payload(s) for s in c.steps]}


def controller_from_payload(p: dict) -> Controller:
    return Controller(steps=tuple(_instr_from_payload(s) for s in p.get("steps", ())),
                      name=p.get("name", "loaded"))


class SkillMemory:
    """A capacity-bounded, score-ranked, deduplicated library of controllers on disk."""

    def __init__(self, path, capacity: int = 20):
        self.path = Path(path)
        self.capacity = capacity
        self.entries = self._load()

    def _load(self) -> list:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text())
            return [{"score": float(e["score"]),
                     "controller": controller_from_payload(e["controller"]),
                     "meta": e.get("meta", {})} for e in raw]
        except Exception:
            return []

    def add(self, controller: Controller, score: float, meta: dict | None = None) -> None:
        sig = controller_to_payload(controller)
        for e in self.entries:                       # dedupe by exact structure
            if controller_to_payload(e["controller"]) == sig:
                if score > e["score"]:
                    e["score"], e["meta"] = float(score), meta or e["meta"]
                self._trim()
                return
        self.entries.append({"score": float(score), "controller": controller, "meta": meta or {}})
        self._trim()

    def _trim(self) -> None:
        self.entries.sort(key=lambda e: e["score"], reverse=True)
        del self.entries[self.capacity:]

    def seeds(self, k: int | None = None) -> list:
        return [e["controller"] for e in self.entries[:(k if k is not None else len(self.entries))]]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"score": e["score"], "controller": controller_to_payload(e["controller"]),
                    "meta": e["meta"]} for e in self.entries]
        self.path.write_text(json.dumps(payload, indent=1))
