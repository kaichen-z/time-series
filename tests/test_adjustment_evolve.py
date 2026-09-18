from __future__ import annotations

import random

from evolving_loop.adjustment.dsl import (
    GROUNDED_EVENT, IDENTITY, KERNEL_MAX_FRAC, Policy, apply_policy,
)
from evolving_loop.adjustment.evolve import (
    crossover, mutate, random_rule, run_evolution,
)


def test_mutate_returns_legal_bounded_policy():
    rng = random.Random(0)
    p = GROUNDED_EVENT
    for _ in range(200):
        p = mutate(p, rng)
        assert isinstance(p, Policy)
        for r in p.rules:
            assert 0.0 <= r.action.cap <= KERNEL_MAX_FRAC   # cap stays in kernel range
    # a mutated policy still applies within the kernel envelope on arbitrary input
    base = [10.0, 20.0, 30.0]
    ts = ("2026-01-01T00:00:00", "2026-01-02T00:00:00", "2026-01-03T00:00:00")
    out, _ = apply_policy(p, base, (), ts)
    for b, o in zip(base, out):
        assert abs(o - b) <= 0.5 * abs(b) + 1e-9


def test_crossover_never_empty():
    rng = random.Random(1)
    a = Policy(rules=(random_rule(rng), random_rule(rng)))
    b = Policy(rules=(random_rule(rng),))
    for _ in range(50):
        c = crossover(a, b, rng)
        assert len(c.rules) >= 1 and len(c.rules) <= 6


def test_evolution_optimizes_a_toy_objective():
    # reward a policy whose first rule's cap is near 0.4
    def fit(p):
        if not p.rules:
            return -1.0
        return -abs(p.rules[0].action.cap - 0.4)
    best, score, hist = run_evolution([GROUNDED_EVENT], fit, generations=40, pop_size=30)
    assert abs(best.rules[0].action.cap - 0.4) < 0.05     # search converged near target
    assert hist[-1][1] >= hist[0][1]                       # non-decreasing best fitness


def test_evolution_is_deterministic_given_seed():
    fit = lambda p: -len(p.rules)
    b1, s1, _ = run_evolution([GROUNDED_EVENT], fit, generations=10, pop_size=20, seed=7)
    b2, s2, _ = run_evolution([GROUNDED_EVENT], fit, generations=10, pop_size=20, seed=7)
    assert s1 == s2 and b1.rules == b2.rules
