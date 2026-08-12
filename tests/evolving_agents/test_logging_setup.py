"""The file gets everything, the terminal gets a readable subset."""

from __future__ import annotations

import logging

from evolving_agents.logging_setup import TRACE_LOGGER, configure, log_exception


def _emit_both() -> None:
    """Log one ordinary progress line and one verbose trace line."""
    logging.getLogger("evolving_agents.evolve.loop").info("generation 0: evaluating")
    logging.getLogger(TRACE_LOGGER).info("LLM_CALL draw=0")


def test_trace_goes_to_the_file_but_not_the_terminal(tmp_path, capsys) -> None:
    path = configure(tmp_path / "run.log", console_level="INFO")
    _emit_both()
    logging.shutdown()

    body = path.read_text(encoding="utf-8")
    assert "generation 0: evaluating" in body
    assert "LLM_CALL draw=0" in body  # the file has everything

    console = capsys.readouterr().err
    assert "generation 0: evaluating" in console
    assert "LLM_CALL draw=0" not in console  # the terminal stays readable


def test_trace_console_opts_back_in(tmp_path, capsys) -> None:
    configure(tmp_path / "run.log", trace_to_console=True)
    _emit_both()
    assert "LLM_CALL draw=0" in capsys.readouterr().err


def test_console_level_can_silence_progress(tmp_path, capsys) -> None:
    path = configure(tmp_path / "run.log", console_level="WARNING")
    _emit_both()
    logging.getLogger("evolving_agents").warning("something worth seeing")
    logging.shutdown()

    console = capsys.readouterr().err
    assert "generation 0: evaluating" not in console
    assert "something worth seeing" in console
    assert "generation 0: evaluating" in path.read_text(encoding="utf-8")  # still recorded


def test_progress_goes_to_stderr_so_stdout_stays_machine_readable(tmp_path, capsys) -> None:
    configure(tmp_path / "run.log")
    logging.getLogger("evolving_agents.evolve.loop").info("progress")
    captured = capsys.readouterr()
    assert "progress" in captured.err
    assert captured.out == ""


def test_uncaught_traceback_reaches_the_log_file(tmp_path) -> None:
    path = configure(tmp_path / "run.log")
    try:
        raise RuntimeError("model failed to load")
    except RuntimeError as exc:
        log_exception(exc)
    logging.shutdown()

    body = path.read_text(encoding="utf-8")
    assert "model failed to load" in body
    assert "Traceback" in body


def test_reconfiguring_does_not_duplicate_handlers(tmp_path) -> None:
    configure(tmp_path / "a.log")
    configure(tmp_path / "b.log")
    logging.getLogger("evolving_agents.evolve.loop").info("only once")
    logging.shutdown()

    assert "only once" not in (tmp_path / "a.log").read_text(encoding="utf-8")
    assert (tmp_path / "b.log").read_text(encoding="utf-8").count("only once") == 1


def test_log_directory_is_created(tmp_path) -> None:
    path = configure(tmp_path / "nested" / "deeper" / "run.log")
    assert path.parent.is_dir()
