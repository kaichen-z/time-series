"""A small (mu+lambda) evolutionary search over Level-0 DSL policies.

The genome IS a `dsl.Policy` (rules-as-data), so every candidate is auditable by
construction (`policy_to_text`) and safe by construction (the kernel bounds it). This
module only provides variation (mutate/crossover) + selection; the *fitness* is passed
in by the caller (so the same engine works for any dataset / stratified objective).

Nothing here can violate the invariant kernel: mutation only edits predicate/action
fields; grounding + bounding are enforced inside `apply_policy`, not here.
"""
from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable, Sequence

from .dsl import KERNEL_MAX_FRAC, Action, Policy, Predicate, Rule, available_regimes

_DIRECTION_SETS = ((), ("increase",), ("decrease",), ("increase", "decrease"))
_OPS = ("scale", "shift", "none")
_MAGS = ("frac_capped", "fixed", "history_calibrated")
_PRED_FLAGS = (
    "require_numeric_eligible", "require_entity_match", "require_target_match",
    "require_window", "require_magnitude", "require_overlap_anomaly",
)


def _clamp_cap(c: float) -> float:
    return max(0.0, min(KERNEL_MAX_FRAC, c))


def random_predicate(rng: random.Random) -> Predicate:
    return Predicate(
        directions=rng.choice(_DIRECTION_SETS),
        require_numeric_eligible=rng.random() < 0.3,
        require_entity_match=rng.random() < 0.3,
        require_target_match=rng.random() < 0.3,
        require_window=rng.random() < 0.8,
        require_magnitude=rng.random() < 0.5,
        require_overlap_anomaly=rng.random() < 0.2,
    )


def random_action(rng: random.Random) -> Action:
    mag = rng.choice(_MAGS)
    return Action(
        op=rng.choice(("scale", "scale", "shift")),  # bias toward scale
        magnitude=mag,
        cap=_clamp_cap(rng.uniform(0.05, KERNEL_MAX_FRAC)),
        fixed=rng.uniform(0.0, 0.5),
        regime=rng.choice(available_regimes()) if mag == "history_calibrated" else "",
    )


def random_rule(rng: random.Random) -> Rule:
    return Rule(random_predicate(rng), random_action(rng))


def _mutate_predicate(p: Predicate, rng: random.Random) -> Predicate:
    if rng.random() < 0.4:
        return replace(p, directions=rng.choice(_DIRECTION_SETS))
    flag = rng.choice(_PRED_FLAGS)
    return replace(p, **{flag: not getattr(p, flag)})


def _mutate_action(a: Action, rng: random.Random) -> Action:
    r = rng.random()
    if r < 0.35:
        return replace(a, cap=_clamp_cap(a.cap + rng.uniform(-0.15, 0.15)))
    if r < 0.6:
        mag = rng.choice(_MAGS)
        return replace(a, magnitude=mag,
                       regime=(rng.choice(available_regimes()) if mag == "history_calibrated"
                               else a.regime))
    if r < 0.8 and a.magnitude == "history_calibrated":
        return replace(a, regime=rng.choice(available_regimes()))
    if r < 0.9:
        return replace(a, op=rng.choice(_OPS))
    return replace(a, fixed=max(0.0, a.fixed + rng.uniform(-0.1, 0.1)))


def mutate(policy: Policy, rng: random.Random) -> Policy:
    """Return a new Policy with one structural or field-level edit."""
    rules = list(policy.rules)
    r = rng.random()
    if not rules or r < 0.15:                      # add a rule
        rules.insert(rng.randint(0, len(rules)), random_rule(rng))
    elif r < 0.30 and len(rules) > 1:              # remove a rule
        del rules[rng.randrange(len(rules))]
    elif r < 0.42 and len(rules) > 1:              # swap order
        i, j = rng.sample(range(len(rules)), 2)
        rules[i], rules[j] = rules[j], rules[i]
    else:                                          # edit a random rule
        i = rng.randrange(len(rules))
        rule = rules[i]
        if rng.random() < 0.5:
            rule = Rule(_mutate_predicate(rule.predicate, rng), rule.action)
        else:
            rule = Rule(rule.predicate, _mutate_action(rule.action, rng))
        rules[i] = rule
    return Policy(rules=tuple(rules), name="evolved", max_frac=policy.max_frac)


def crossover(a: Policy, b: Policy, rng: random.Random) -> Policy:
    """One-point crossover on the two rule lists."""
    ra, rb = list(a.rules), list(b.rules)
    if not ra or not rb:
        return Policy(rules=tuple(ra + rb) or (), name="evolved")
    ca = ra[: rng.randint(0, len(ra))]
    cb = rb[rng.randint(0, len(rb)):]
    child = (ca + cb)[:6] or [random_rule(rng)]    # cap length; never empty
    return Policy(rules=tuple(child), name="evolved", max_frac=a.max_frac)


def run_evolution(
    seeds: Sequence[Policy],
    fitness: Callable[[Policy], float],
    *,
    generations: int = 25,
    pop_size: int = 40,
    elite: int = 8,
    seed: int = 20260918,
):
    """(mu+lambda) evolution. Returns (best_policy, best_fitness, history).

    `fitness` maps a Policy to a scalar (higher is better). Ties in scoring are broken
    toward fewer rules (parsimony) so auditable, simple champions win.
    """
    rng = random.Random(seed)
    pop = list(seeds)
    while len(pop) < pop_size:
        base = rng.choice(seeds) if seeds and rng.random() < 0.5 else Policy(name="rand")
        pop.append(mutate(base, rng) if seeds else Policy(rules=(random_rule(rng),)))

    def key(p):
        return (fitness(p), -len(p.rules))

    history = []
    best = max(pop, key=key)
    for gen in range(generations):
        ranked = sorted(pop, key=key, reverse=True)
        parents = ranked[:elite]
        gbest = ranked[0]
        if key(gbest) > key(best):
            best = gbest
        history.append((gen, fitness(gbest), len(gbest.rules)))
        children = list(parents)
        while len(children) < pop_size:
            if rng.random() < 0.5 and len(parents) >= 2:
                a, b = rng.sample(parents, 2)
                child = crossover(a, b, rng)
            else:
                child = rng.choice(parents)
            children.append(mutate(child, rng))
        pop = children
    return best, fitness(best), history
