from __future__ import annotations

from pathlib import Path

import pytest

from numerical_agent.evolution.history import History, Operation, parse_history

REPO = Path(__file__).resolve().parent.parent / "runs" / "method_evolution" / "v002"


def log(*commits: str) -> str:
    """Commits newest first, the order git itself prints them in."""
    return "\n\n".join(commits) + "\n"


def generation(number: int, *lines: str) -> str:
    return f"generation {number}: {len(lines)} operations\n" + "\n".join(lines)


def test_one_generation_parses_every_operation_it_carried() -> None:
    text = log(generation(2, "- delete beta: too slow", "- add gamma: covers the gap"))

    operations = parse_history(text).operations

    assert [(o.generation, o.op, o.name, o.reason) for o in operations] == [
        (2, "delete", "beta", "too slow"),
        (2, "add", "gamma", "covers the gap"),
    ]


def test_a_reason_containing_a_colon_survives_whole() -> None:
    text = log(generation(3, "- rewrite alpha: Rewrite to be selective: only accept breaks."))

    assert parse_history(text).operations[0].reason == (
        "Rewrite to be selective: only accept breaks."
    )


def test_the_seed_commit_contributes_no_operations() -> None:
    text = log("seed 5 composed forecasting methods\n- alpha\n- beta")

    assert parse_history(text).operations == ()
    assert not parse_history(text)


def test_bookkeeping_for_a_name_consumed_earlier_is_not_an_operation() -> None:
    text = log(generation(
        2,
        "- delete beta: already removed earlier in this batch; superseded",
        "- add gamma: real work",
    ))

    operations = parse_history(text).operations

    assert [o.name for o in operations] == ["gamma"]


def test_generations_are_ordered_oldest_first_although_git_prints_them_newest_first() -> None:
    text = log(
        generation(3, "- add gamma: third"),
        generation(2, "- add beta: second"),
    )

    assert [o.generation for o in parse_history(text).operations] == [2, 3]


def test_a_merge_records_the_name_it_produced_and_the_names_it_consumed() -> None:
    text = log(generation(4, "- merge alpha, beta -> gamma: one is enough"))

    operation = parse_history(text).operations[0]

    assert operation.op == "merge"
    assert operation.name == "gamma"
    assert operation.sources == ("alpha", "beta")
    # Neither source survives under its own name, so the merge removed both.
    assert operation.removes == ("alpha", "beta")


def test_a_merge_into_one_of_its_own_sources_does_not_remove_that_source() -> None:
    text = log(generation(4, "- merge alpha, beta -> alpha: fold beta in"))

    operation = parse_history(text).operations[0]

    assert operation.name == "alpha"
    assert operation.removes == ("beta",)


def test_for_method_returns_one_methods_operations_oldest_first() -> None:
    text = log(
        generation(3, "- rewrite alpha: tightened"),
        generation(2, "- add alpha: new", "- add beta: unrelated"),
    )

    history = parse_history(text)

    assert [(o.generation, o.op) for o in history.for_method("alpha")] == [(2, "add"), (3, "rewrite")]


def test_a_method_deleted_and_never_re_added_is_reported_as_removed() -> None:
    text = log(generation(2, "- delete beta: measurably worse"))

    assert [o.name for o in parse_history(text).removed()] == ["beta"]


def test_a_method_deleted_then_re_added_is_no_longer_reported_as_removed() -> None:
    text = log(
        generation(3, "- add beta: worth another try"),
        generation(2, "- delete beta: measurably worse"),
    )

    history = parse_history(text)

    assert history.removed() == ()
    # The deletion is not lost, it just belongs to the living method's own history.
    assert [o.op for o in history.for_method("beta")] == ["delete", "add"]


def test_a_method_still_in_the_module_is_never_reported_as_removed() -> None:
    text = log(
        generation(4, "- add gamma: brand new"),
        generation(2, "- delete beta: measurably worse"),
    )

    assert [o.name for o in parse_history(text).removed()] == ["beta"]


def test_a_method_deleted_twice_is_reported_twice() -> None:
    text = log(
        generation(5, "- delete beta: failed again"),
        generation(4, "- add beta: different configuration"),
        generation(2, "- delete beta: failed"),
    )

    removed = parse_history(text).removed()

    assert [(o.generation, o.name) for o in removed] == [(5, "beta"), (2, "beta")]


def test_removed_is_newest_first_so_the_freshest_evidence_reads_first() -> None:
    text = log(
        generation(4, "- delete gamma: later"),
        generation(2, "- delete beta: earlier"),
    )

    assert [o.name for o in parse_history(text).removed()] == ["gamma", "beta"]


def test_an_empty_log_is_an_empty_history() -> None:
    assert parse_history("").operations == ()
    assert not History()


@pytest.mark.skipif(not REPO.exists(), reason="the v002 evolution repository is not present")
def test_the_real_v002_run_parses_into_its_recorded_operations() -> None:
    from numerical_agent.evolution import run_git

    history = parse_history(run_git(REPO, "log", "--format=%s%n%b"))

    assert len(history.operations) == 47
    counts = {op: sum(1 for o in history.operations if o.op == op) for op in ("add", "delete", "rewrite")}
    assert counts == {"add": 16, "delete": 14, "rewrite": 17}
    # Added, deleted, re-added, deleted, re-added: the churn this whole feature exists to stop.
    assert [o.op for o in history.for_method("state_space_forecast")] == [
        "add", "delete", "add", "delete", "add",
    ]


def test_removed_takes_the_modules_own_names_because_seed_methods_have_no_operation() -> None:
    """A seed method is in the module but was never added by an operation."""
    text = log(
        generation(3, "- delete gamma: worse than the seed"),
        generation(2, "- add gamma: worth a try"),
    )
    history = parse_history(text)

    # Replaying operations alone cannot know that the seed method alpha still exists.
    assert "alpha" not in history.live()
    assert [o.name for o in history.removed(live=["alpha"])] == ["gamma"]


def test_describe_past_generations_is_empty_before_anything_has_happened() -> None:
    from numerical_agent.evolution.prompts import describe_past_generations

    assert describe_past_generations(parse_history(""), ["alpha"]) == ""


def test_describe_past_generations_omits_the_heading_when_no_live_method_has_a_past() -> None:
    """A seed-only module with one unrelated deletion still has something to say; nothing at
    all to say means no heading."""
    from numerical_agent.evolution.prompts import describe_past_generations

    text = log(generation(2, "- rewrite alpha: tightened"))
    # alpha is not live, so its history is irrelevant and nothing was removed either.
    assert describe_past_generations(parse_history(text), ["beta"]) == ""


def test_describe_past_generations_shows_a_live_methods_own_operations() -> None:
    from numerical_agent.evolution.prompts import describe_past_generations

    text = log(generation(2, "- rewrite alpha: fixed the unpacking bug"))

    rendered = describe_past_generations(parse_history(text), ["alpha"])

    assert "How the current methods got here" in rendered
    assert "gen 2 rewrite -- fixed the unpacking bug" in rendered


def test_describe_past_generations_shows_prior_burials_under_a_method_that_is_alive_again() -> None:
    """The evidence that matters most: a live method that already failed twice."""
    from numerical_agent.evolution.prompts import describe_past_generations

    text = log(
        generation(4, "- add beta: third attempt"),
        generation(3, "- delete beta: failed again"),
        generation(2, "- add beta: first attempt"),
    )

    rendered = describe_past_generations(parse_history(text), ["beta"])

    assert "gen 3 delete -- failed again" in rendered
    assert "Tried and removed" not in rendered


def test_describe_past_generations_carries_every_removal_with_its_reason_verbatim() -> None:
    from numerical_agent.evolution.prompts import describe_past_generations

    reasons = [f"removal number {i} with its own distinct rationale" for i in range(30)]
    text = log(*[generation(i + 2, f"- delete m{i}: {reason}") for i, reason in enumerate(reasons)])

    rendered = describe_past_generations(parse_history(text), ["alpha"])

    for index, reason in enumerate(reasons):
        assert f"m{index} (removed in generation {index + 2}) -- {reason}" in rendered


def test_the_evolve_prompt_puts_the_history_before_the_measurements() -> None:
    from numerical_agent.evolution.prompts import build_improve_request

    text = log(generation(2, "- delete beta: measurably worse"))

    rendered = build_improve_request(
        module_source="def alpha(history, horizon, frequency):\n    return []\n",
        reports=[{"method": "alpha"}],
        generation=3,
        task_count=80,
        history=parse_history(text),
        live=["alpha"],
    )

    assert rendered.index("already established") < rendered.index("# Measured results")
    assert "beta" in rendered


def test_the_evolve_prompt_is_unchanged_when_there_is_no_history() -> None:
    from numerical_agent.evolution.prompts import build_improve_request

    rendered = build_improve_request(
        module_source="def alpha(history, horizon, frequency):\n    return []\n",
        reports=[{"method": "alpha"}],
        generation=1,
        task_count=80,
        history=parse_history(""),
        live=["alpha"],
    )

    assert "already established" not in rendered
    assert rendered.startswith("# Measured results")


def test_the_system_prompt_says_a_removed_method_cannot_come_back() -> None:
    """With no `add`, a delete is final -- the prompt must not imply it can be undone."""
    from numerical_agent.evolution.prompts import IMPROVE_METHODS_PROMPT

    assert "gone for good" in IMPROVE_METHODS_PROMPT
    assert '"op": "add"' not in IMPROVE_METHODS_PROMPT


def test_an_empty_module_is_told_that_no_operation_can_fill_it() -> None:
    """With no `add`, an empty module cannot be rebuilt from the prompt -- say so plainly."""
    from numerical_agent.evolution.prompts import build_improve_request

    rendered = build_improve_request(module_source="x", reports=[], generation=1, task_count=80)

    assert "The module is empty" in rendered
    assert "has to be seeded" in rendered


def test_the_system_prompt_weighs_the_three_metrics_equally() -> None:
    """No metric may be named the primary one -- that framing is what selected for flatness."""
    from numerical_agent.evolution.prompts import IMPROVE_METHODS_PROMPT

    assert "three metrics of equal weight" in IMPROVE_METHODS_PROMPT
    assert "primary signal" not in IMPROVE_METHODS_PROMPT
    assert "mean_shape_correlation" in IMPROVE_METHODS_PROMPT


def test_compressing_the_prompt_did_not_drop_a_rule() -> None:
    """The prompt is written tight; every constraint it used to state must still be stated."""
    from numerical_agent.evolution.prompts import IMPROVE_METHODS_PROMPT

    required = [
        "There are no fallbacks.",            # no silent recovery
        "Never catch a broad exception",      # ... nor a disguised one
        "entire function",                    # code fields are never abbreviated
        "apply in sequence",                  # operations are ordered and consume names
        "states a reason",                    # every operation is justified
        "inside the function body",           # module-level imports are discarded
        "three metrics of equal weight",      # no metric outranks the others
        "mean_shape_correlation",
        "mean_smae",
        "mean_srmse",
    ]
    missing = [rule for rule in required if rule not in IMPROVE_METHODS_PROMPT]

    assert not missing, missing


def test_the_prompt_stays_within_budget() -> None:
    """A prompt that creeps back up costs a call every generation; catch it here."""
    from numerical_agent.evolution.prompts import IMPROVE_METHODS_PROMPT

    assert len(IMPROVE_METHODS_PROMPT) < 15_500


def test_the_report_fields_the_prompt_documents_are_the_ones_actually_sent() -> None:
    """A documented field that no longer exists teaches the model to look for nothing."""
    from numerical_agent.evolution.execution import MethodReport
    from numerical_agent.evolution.prompts import IMPROVE_METHODS_PROMPT
    import dataclasses

    documented = {"mean_smae", "mean_srmse", "mean_shape_correlation",
                  "mean_variance_ratio", "mean_change_smae", "coverage",
                  "not_applicable", "crashed", "invalid", "smae_by_series_type",
                  "sample_failures"}
    actual = {f.name for f in dataclasses.fields(MethodReport)}

    assert documented <= actual, documented - actual
    assert all(name in IMPROVE_METHODS_PROMPT for name in documented)
