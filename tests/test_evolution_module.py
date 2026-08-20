from __future__ import annotations

from pathlib import Path

import pytest

from numerical_agent.evolution.module import (
    MODULE_HEADER,
    MethodModule,
    ModuleError,
    apply_operations,
    parse_method,
    parse_module,
    read_module,
    write_module,
)


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

    assert read_module(destination).names() == ("alpha", "beta")
    assert "class NotApplicable" in destination.read_text(encoding="utf-8")


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
