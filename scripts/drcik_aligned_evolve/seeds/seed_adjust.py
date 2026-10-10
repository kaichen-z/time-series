"""Seed Decision module (L4 start point): apply retrieved corrections to the base forecast conservatively.
Each correction multiplies the base on [start, end); overlapping corrections combine multiplicatively; the total
change per step is bounded to +-10% of the base. adjust(view) -> list of H floats (view has base_forecast, corrections)."""


def adjust(view):
    base = list(view["base_forecast"]); H = view["H"]; mult = [1.0] * H
    for c in view.get("corrections") or []:
        for i in range(max(0, int(c["start"])), min(H, int(c["end"]))):
            mult[i] *= float(c["multiplier"])
    return [b * max(0.9, min(1.1, m)) for b, m in zip(base, mult)]
