"""Bound forecast amplitude when retrieved context is noisy rather than actionable."""


def retrieve(view):
    if not (view.get("documents") or []):
        return []
    horizon = int(view["H"])
    return [{
        "start": 0,
        "end": horizon,
        "multiplier": 0.9,
        "confidence": 0.6,
        "source": "document_uncertainty",
        "kind": "bounded_mean_shrink",
    }]
