from __future__ import annotations

from evolving_loop.adjustment.fitness import cvar_downside_fitness, worst_case


def test_empty_and_identity_are_zero():
    assert cvar_downside_fitness([]) == 0.0
    assert cvar_downside_fitness([0.0, 0.0, 0.0]) == 0.0


def test_pure_improvement_equals_mean_no_penalty():
    # no regressions -> downside tail is non-negative -> fitness == mean improvement
    d = [0.1, 0.2, 0.0, 0.3]
    assert abs(cvar_downside_fitness(d) - (sum(d) / len(d))) < 1e-9


def test_regression_is_penalized_below_its_mean():
    d = [0.3, 0.3, 0.3, -0.3]           # one bad task
    m = sum(d) / len(d)
    f = cvar_downside_fitness(d)
    assert f < m                         # the downside tail drags it below the plain mean


def test_ordering_improve_zeroreg_beats_identity_beats_regressor():
    improve_clean = [0.0] * 7 + [0.1, 0.15, 0.2]     # gains, no regression
    identity = [0.0] * 10
    regressor = [0.2, 0.2, 0.1] + [-0.15] * 7         # a few wins, many small regressions
    a = cvar_downside_fitness(improve_clean)
    b = cvar_downside_fitness(identity)
    c = cvar_downside_fitness(regressor)
    assert a > b >= c or (a > b and b > c)
    assert a > 0 and b == 0.0 and c < 0.0


def test_beta_makes_it_more_risk_averse():
    d = [0.3, 0.3, -0.2]
    f_low = cvar_downside_fitness(d, beta=1.0)
    f_high = cvar_downside_fitness(d, beta=4.0)
    assert f_high < f_low                 # larger beta punishes the downside tail more


def test_worst_case_reports_biggest_regression():
    assert worst_case([0.2, -0.4, 0.1]) == -0.4
    assert worst_case([]) == 0.0
