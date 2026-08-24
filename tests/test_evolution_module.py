"""Tests for numerical_agent/evolution/module: parsing, applying operations, and the seed-import allow-list."""
from __future__ import annotations

import pytest
import json
from pathlib import Path
from numerical_agent.evolution.module import (
    EVOLUTION_IMPORTS,
    MODULE_HEADER,
    MethodModule,
    ModuleError,
    SKILLS_MODULE,
    apply_operations,
    parse_method,
    parse_module,
    read_module,
    write_module,
)
from common.sandbox import ALLOWED_IMPORTS, UnsafeCodeError, check_code
from numerical_agent.evolution.seed import EXCLUDED_CATEGORIES, batches, seed_definitions


def method_source(name: str, body: str = "    return [float(history[-1])] * horizon") -> str:
    return (
        f"def {name}(history, horizon, frequency):\n"
        f'    """Use when nothing better applies."""\n'
        f"{body}\n"
    )


def module_text(*names: str) -> str:
    return MODULE_HEADER + "\n\n" + "\n\n".join(method_source(n) for n in names)


def test_parse_module_reads_every_method() -> None:
    module = parse_module(module_text("naive_last", "naive_mean"))

    assert module.names() == ("naive_last", "naive_mean")
    assert module.get("naive_last").docstring == "Use when nothing better applies."


def test_render_and_parse_round_trip_preserves_methods() -> None:
    module = parse_module(module_text("alpha", "beta", "gamma"))

    assert parse_module(module.render()).names() == module.names()


def test_parse_module_rejects_a_wrong_signature() -> None:
    bad = MODULE_HEADER + '\n\ndef alpha(history, horizon):\n    """Doc."""\n    return []\n'

    with pytest.raises(ModuleError, match="must take exactly"):
        parse_module(bad)


def test_parse_module_rejects_a_missing_docstring() -> None:
    bad = MODULE_HEADER + "\n\ndef alpha(history, horizon, frequency):\n    return []\n"

    with pytest.raises(ModuleError, match="docstring"):
        parse_module(bad)


def test_parse_module_rejects_duplicate_names() -> None:
    with pytest.raises(ModuleError, match="duplicate"):
        parse_module(module_text("alpha", "alpha"))


def test_parse_module_rejects_a_syntax_error() -> None:
    with pytest.raises(ModuleError, match="does not parse"):
        parse_module(MODULE_HEADER + "\n\ndef alpha(:\n")


def test_delete_removes_one_method() -> None:
    module = parse_module(module_text("alpha", "beta"))

    updated, summaries = apply_operations(
        module, [{"op": "delete", "name": "alpha", "reason": "dominated everywhere"}]
    )

    assert updated.names() == ("beta",)
    assert summaries == ("delete alpha: dominated everywhere",)


def test_delete_refuses_to_empty_the_module() -> None:
    module = parse_module(module_text("only"))

    with pytest.raises(ModuleError, match="last remaining"):
        apply_operations(module, [{"op": "delete", "name": "only", "reason": "bad"}])


def test_rewrite_replaces_source_in_place() -> None:
    module = parse_module(module_text("alpha", "beta"))
    new = (
        "def alpha(history, horizon, frequency):\n"
        '    """Use for strictly positive series."""\n'
        "    return [1.0] * horizon\n"
    )

    updated, _ = apply_operations(
        module, [{"op": "rewrite", "name": "alpha", "code": new, "reason": "fixed indexing"}]
    )

    # Order is preserved so diffs stay readable across generations.
    assert updated.names() == ("alpha", "beta")
    assert updated.get("alpha").docstring == "Use for strictly positive series."


def test_rewrite_rejects_a_renamed_function() -> None:
    module = parse_module(module_text("alpha"))

    with pytest.raises(ModuleError, match="expected 'alpha'"):
        apply_operations(
            module,
            [{"op": "rewrite", "name": "alpha", "code": method_source("other"), "reason": "x"}],
        )


def test_add_appends_a_new_method() -> None:
    module = parse_module(module_text("alpha"))

    updated, _ = apply_operations(
        module, [{"op": "add", "code": method_source("gamma"), "reason": "covers a gap"}]
    )

    assert updated.names() == ("alpha", "gamma")


def test_add_rejects_an_existing_name() -> None:
    module = parse_module(module_text("alpha"))

    with pytest.raises(ModuleError, match="already exists"):
        apply_operations(
            module, [{"op": "add", "code": method_source("alpha"), "reason": "dup"}]
        )


def test_merge_consolidates_several_methods_into_one() -> None:
    module = parse_module(module_text("alpha", "beta", "gamma"))

    updated, summaries = apply_operations(
        module,
        [
            {
                "op": "merge",
                "names": ["alpha", "beta"],
                "into": "alpha",
                "code": method_source("alpha"),
                "reason": "identical forecasts",
            }
        ],
    )

    assert set(updated.names()) == {"gamma", "alpha"}
    assert summaries[0].startswith("merge alpha, beta -> alpha:")


def test_merge_requires_at_least_two_methods() -> None:
    module = parse_module(module_text("alpha", "beta"))

    with pytest.raises(ModuleError, match="at least two"):
        apply_operations(
            module,
            [{"op": "merge", "names": ["alpha"], "into": "alpha", "code": method_source("alpha"), "reason": "x"}],
        )


def test_every_operation_requires_a_reason() -> None:
    module = parse_module(module_text("alpha", "beta"))

    with pytest.raises(ModuleError, match="reason"):
        apply_operations(module, [{"op": "delete", "name": "alpha"}])


def test_unknown_operation_is_rejected() -> None:
    module = parse_module(module_text("alpha"))

    with pytest.raises(ModuleError, match="unsupported op"):
        apply_operations(module, [{"op": "obliterate", "name": "alpha", "reason": "x"}])


def test_a_failing_operation_leaves_the_module_untouched() -> None:
    module = parse_module(module_text("alpha", "beta"))

    with pytest.raises(ModuleError, match="operation 2"):
        apply_operations(
            module,
            [
                {"op": "delete", "name": "alpha", "reason": "ok"},
                {"op": "delete", "name": "missing", "reason": "bad"},
            ],
        )

    # The caller's module object is unchanged: nothing is applied in place.
    assert module.names() == ("alpha", "beta")


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    module = parse_module(module_text("alpha", "beta"))
    destination = tmp_path / "methods.py"

    write_module(destination, module)

    text = destination.read_text(encoding="utf-8")
    assert read_module(destination).names() == ("alpha", "beta")
    # NotApplicable is imported from the skill library, not redefined, so a skill raising it
    # is the very class the runner catches.
    assert f"from {SKILLS_MODULE} import NotApplicable" in text
    assert f"import {SKILLS_MODULE} as P" in text


def test_write_rejects_a_disallowed_import(tmp_path: Path) -> None:
    module = parse_module(
        module_text("alpha")
        + "\n\ndef beta(history, horizon, frequency):\n"
        '    """Doc."""\n'
        "    import os\n"
        "    return [0.0] * horizon\n"
    )

    with pytest.raises(Exception, match="disallowed import"):
        write_module(tmp_path / "methods.py", module)


def test_parse_method_rejects_more_than_one_function() -> None:
    with pytest.raises(ModuleError, match="exactly one function"):
        parse_method(method_source("alpha") + "\n" + method_source("beta"))


def test_written_module_is_importable_and_callable(tmp_path: Path) -> None:
    import importlib.util

    module = parse_module(module_text("naive_last"))
    destination = write_module(tmp_path / "methods.py", module)

    spec = importlib.util.spec_from_file_location("evolved_methods", destination)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)

    assert loaded.naive_last([1.0, 2.0, 5.0], 3, "1 day") == [5.0, 5.0, 5.0]
    assert issubclass(loaded.NotApplicable, Exception)
    assert loaded.P.infer_period([1.0, 2.0] * 20) == 2


def test_notapplicable_from_an_except_handler_is_rejected() -> None:
    laundered = (
        "def alpha(history, horizon, frequency):\n"
        '    """Use for seasonal series."""\n'
        "    try:\n"
        "        value = compute(history)\n"
        "    except Exception:\n"
        '        raise NotApplicable("decomposition failed")\n'
        "    return [value] * horizon\n"
    )

    # A bug disguised as inapplicability is exactly what the metrics must never absorb.
    with pytest.raises(ModuleError, match="except handler"):
        parse_method(laundered)


def test_returning_a_forecast_from_a_broad_handler_is_rejected() -> None:
    fallback = (
        "def alpha(history, horizon, frequency):\n"
        '    """Use for seasonal series."""\n'
        "    try:\n"
        "        return model(history, horizon)\n"
        "    except Exception:\n"
        "        return [float(history[-1])] * horizon\n"
    )

    with pytest.raises(ModuleError, match="no fallbacks"):
        parse_method(fallback)


def test_a_narrow_handler_that_does_not_silence_is_allowed() -> None:
    fine = (
        "def alpha(history, horizon, frequency):\n"
        '    """Use for positive series."""\n'
        "    if len(history) < 3:\n"
        '        raise NotApplicable("needs 3 points")\n'
        "    try:\n"
        "        scale = 1.0 / history[-1]\n"
        "    except ZeroDivisionError:\n"
        "        scale = 1.0\n"
        "    return [float(history[-1]) * scale] * horizon\n"
    )

    assert parse_method(fine).name == "alpha"


def test_delete_after_a_merge_consumed_the_method_is_a_no_op() -> None:
    """The model merges three naives, then redundantly deletes one it already merged."""
    module = parse_module(module_text("naive_last", "naive_mean", "naive_drift", "other"))

    updated, summaries = apply_operations(
        module,
        [
            {
                "op": "merge",
                "names": ["naive_last", "naive_mean", "naive_drift"],
                "into": "naive_last",
                "code": method_source("naive_last"),
                "reason": "similar MASE; consolidate",
            },
            {"op": "delete", "name": "naive_mean", "reason": "merged into naive_last"},
        ],
    )

    assert set(updated.names()) == {"other", "naive_last"}
    assert "already removed earlier in this batch" in summaries[1]


def test_merge_naming_a_method_an_earlier_delete_removed_still_merges_the_rest() -> None:
    """The model deletes theta_classic, then merges it again alongside theta_optimized."""
    module = parse_module(module_text("theta_classic", "theta_optimized", "other"))

    updated, summaries = apply_operations(
        module,
        [
            {"op": "delete", "name": "theta_classic", "reason": "identical to theta_optimized"},
            {
                "op": "merge",
                "names": ["theta_optimized", "theta_classic"],
                "into": "theta_optimized",
                "code": method_source("theta_optimized"),
                "reason": "identical MASE; redundant",
            },
        ],
    )

    assert set(updated.names()) == {"other", "theta_optimized"}
    assert summaries[1].startswith("merge theta_optimized, theta_classic -> theta_optimized:")


def test_a_method_that_never_existed_is_still_rejected() -> None:
    module = parse_module(module_text("alpha", "beta"))

    with pytest.raises(ModuleError, match="unknown method"):
        apply_operations(
            module,
            [{
                "op": "merge", "names": ["alpha", "hallucinated"], "into": "alpha",
                "code": method_source("alpha"), "reason": "x",
            }],
        )


def test_placeholder_code_says_what_is_actually_wrong() -> None:
    """The v002 run lost four generations to `"code": "..."`; the error must name the cause."""
    with pytest.raises(ModuleError, match="complete function source"):
        parse_method("...")
    with pytest.raises(ModuleError, match="complete function source"):
        parse_method("x = 1")

CATALOG = "numerical_agent/datasets/forecast_method_dataset_v001.json"


def test_the_shared_allow_list_still_rejects_heavy_libraries() -> None:
    # evolving_loop's published-results code depends on this staying narrow.
    with pytest.raises(UnsafeCodeError, match="torch"):
        check_code("import torch\n")
    assert "torch" not in ALLOWED_IMPORTS


def test_the_evolution_allow_list_permits_them() -> None:
    check_code("import torch\nimport sklearn\nimport xgboost\n", EVOLUTION_IMPORTS)

    assert ALLOWED_IMPORTS < EVOLUTION_IMPORTS


def test_the_wider_list_still_blocks_dangerous_modules() -> None:
    with pytest.raises(UnsafeCodeError, match="os"):
        check_code("import os\n", EVOLUTION_IMPORTS)


def test_seed_selection_splits_the_catalog_as_expected() -> None:
    seeds, excluded = seed_definitions(CATALOG)

    assert len(seeds) == 93
    assert len(excluded) == 18
    assert len(seeds) + len(excluded) == 111


def test_excluded_methods_carry_a_recorded_reason() -> None:
    _, excluded = seed_definitions(CATALOG)

    assert {entry["category"] for entry in excluded} == set(EXCLUDED_CATEGORIES)
    assert all(entry["reason"] for entry in excluded)


def test_no_seed_belongs_to_an_excluded_category() -> None:
    seeds, _ = seed_definitions(CATALOG)

    assert not {s["category"] for s in seeds} & set(EXCLUDED_CATEGORIES)


def test_every_seed_name_is_a_usable_function_name() -> None:
    seeds, _ = seed_definitions(CATALOG)
    names = [str(s["name"]) for s in seeds]

    assert all(name.isidentifier() for name in names)
    assert len(set(names)) == len(names)


def test_every_seed_carries_the_text_the_prompt_needs() -> None:
    seeds, _ = seed_definitions(CATALOG)

    assert all(s["description"] for s in seeds)
    assert all(isinstance(s["assumptions"], list) for s in seeds)


def test_batches_cover_every_definition_exactly_once() -> None:
    seeds, _ = seed_definitions(CATALOG)

    grouped = batches(seeds, 10)

    assert sum(len(batch) for batch in grouped) == len(seeds)
    flattened = [str(item["name"]) for batch in grouped for item in batch]
    assert sorted(flattened) == sorted(str(s["name"]) for s in seeds)


def test_batches_keep_a_category_together() -> None:
    seeds, _ = seed_definitions(CATALOG)

    grouped = batches(seeds, 10)

    # Sorted by category, so a category spans at most ceil(n/size)+1 batches.
    baseline_batches = {
        index
        for index, batch in enumerate(grouped)
        for item in batch
        if item["category"] == "baseline"
    }
    assert len(baseline_batches) <= 2
