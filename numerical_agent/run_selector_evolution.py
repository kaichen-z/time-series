"""Build history-only hindcasts, evolve the Numerical Selector, and freeze it on Dev."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import multiprocessing
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from common.data import load_tasks
from common.llm import CodexCLIClient, CodexCLIConfig
from common.payload import read_json_object, write_json

from .evolution.execution import NOT_APPLICABLE, SUCCESS, Outcome, Task, load_methods
from .evolution.module import MethodModule, read_module
from .evolution.numerical_selector import (
    DecisionPolicy,
    HindcastConfig,
    diagnose_candidate,
)
from .evolution.portfolio import (
    CombinedPolicy,
    PolicyPortfolio,
    TSFMPolicy,
    _run_tsfm,
    _signal,
    read_policy_file,
)
from .evolution.screening import materialize_active_dictionary, profile_task
from .evolution.screening_evolution import parse_screening_source
from .evolution.selector_evolution import (
    DecisionCase,
    decision_policy_hash,
    evaluate_decision,
    evolve_selector_train_then_dev,
    render_decision_source,
)
from .main import _add_tsfm_runtime_options, _runtime_registry
from .run_task_conditioned_screening import _training_outcomes, load_frozen_partitions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--screening-dir", required=True)
    parser.add_argument("--split-file", default="splits/drcik_public_80_20_99_v1.json")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--outcome-cache-dir", required=True)
    parser.add_argument("--policy-outcome-cache-dir", required=True)
    parser.add_argument("--hindcast-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=80)
    parser.add_argument("--dev-limit", type=int, default=20)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--train-validation-folds", type=int, default=4)
    parser.add_argument("--codex-model", default="gpt-5.6-luna")
    parser.add_argument(
        "--codex-reasoning-effort", choices=("none", "low", "medium", "high"), default="low"
    )
    parser.add_argument("--codex-timeout", type=int, default=900)
    parser.add_argument("--codex-cache-dir", default=None)
    _add_tsf_runtime_options_compat(parser)
    return parser


def _add_tsf_runtime_options_compat(parser: argparse.ArgumentParser) -> None:
    _add_tsfm_runtime_options(parser)


class ForecastStore:
    """Content-addressed history-only forecast cache shared across selector generations."""

    def __init__(
        self,
        root: str | Path,
        module_path: Path,
        skills_path: Path | None,
        module: MethodModule,
        portfolio: PolicyPortfolio,
        runtimes,
        screening_hash: str,
        statistical_time_budget_s: float = 20.0,
        statistical_failure_limit: int = 2,
    ) -> None:
        if statistical_time_budget_s <= 0:
            raise ValueError("statistical_time_budget_s must be positive")
        if statistical_failure_limit < 1:
            raise ValueError("statistical_failure_limit must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.not_applicable = _HistoryOnlyNotApplicable
        self.statistical_names = frozenset(module.names())
        self._statistical = _IsolatedStatisticalRuntime(
            module_path,
            skills_path,
            time_budget_s=statistical_time_budget_s,
            failure_limit=statistical_failure_limit,
        )
        self.module = module
        self.portfolio = portfolio
        self.runtimes = runtimes
        self.screening_hash = screening_hash
        self.tsfm = {policy.name: policy for policy in portfolio.tsfm}
        self.combined = {policy.name: policy for policy in portfolio.combined}
        self.identity_hash = hashlib.sha256(
            (module_path.read_text(encoding="utf-8") + repr(portfolio)).encode("utf-8")
        ).hexdigest()
        self.hits = 0
        self.misses = 0

    def forecast(
        self, name: str, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        key = self._key(name, history, horizon, frequency)
        path = self.root / f"{key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("key") == key and payload.get("status") == SUCCESS:
                values = tuple(float(value) for value in payload["forecast"])
                if len(values) == horizon and all(map(math.isfinite, values)):
                    self.hits += 1
                    return values
            if payload.get("key") == key and payload.get("status") == NOT_APPLICABLE:
                self.hits += 1
                raise self.not_applicable(str(payload.get("detail", "not applicable")))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass
        self.misses += 1
        try:
            values = self._execute(name, history, horizon, frequency)
        except self.not_applicable as error:
            self._write(path, {"key": key, "status": NOT_APPLICABLE, "detail": str(error)[:200]})
            raise
        self._write(path, {"key": key, "status": SUCCESS, "forecast": list(values)})
        return values

    def _execute(
        self, name: str, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        if name in self.statistical_names:
            return self._statistical.forecast(name, history, horizon, frequency)
        if policy := self.tsfm.get(name):
            dummy = Task("history-only", history, horizon, frequency, (0.0,) * horizon)
            outcome = _run_tsfm(policy, dummy, self.runtimes)
            if outcome.status == NOT_APPLICABLE:
                raise self.not_applicable(outcome.detail)
            if outcome.status != SUCCESS:
                raise RuntimeError(outcome.detail or outcome.status)
            return _valid_forecast(outcome.forecast, horizon)
        if policy := self.combined.get(name):
            return self._combined(policy, history, horizon, frequency)
        raise KeyError(f"unknown numerical candidate {name}")

    def close(self) -> None:
        self._statistical.close()

    def _combined(
        self,
        policy: CombinedPolicy,
        history: tuple[float, ...],
        horizon: int,
        frequency: str,
    ) -> tuple[float, ...]:
        left = self.forecast(policy.tsfm_parent, history, horizon, frequency)
        right = self.forecast(policy.statistical_parent, history, horizon, frequency)
        if policy.mode == "blend":
            return tuple(
                policy.weight * a + (1.0 - policy.weight) * b
                for a, b in zip(left, right, strict=True)
            )
        dummy = Task("history-only", history, horizon, frequency, (0.0,) * horizon)
        choose_left = _signal(policy.signal, dummy) >= policy.threshold
        if policy.tsfm_when == "below":
            choose_left = not choose_left
        return left if choose_left else right

    def _key(self, name, history, horizon, frequency) -> str:
        payload = json.dumps({
            "schema": 2,
            "identity": self.identity_hash,
            "name": name,
            "history": history,
            "horizon": horizon,
            "frequency": frequency,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _write(path: Path, payload: Mapping[str, object]) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, prefix=f".{path.stem}.", delete=False
            ) as handle:
                temporary = handle.name
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)


class _HistoryOnlyNotApplicable(Exception):
    """Transport-safe applicability signal from the statistical worker."""


class _IsolatedStatisticalRuntime:
    """Execute generated statistical methods behind a restartable hard boundary."""

    def __init__(
        self,
        module_path: str | Path,
        skills_path: str | Path | None,
        *,
        time_budget_s: float,
        failure_limit: int,
        startup_timeout_s: float = 10.0,
    ) -> None:
        self.module_path = str(Path(module_path).resolve())
        self.skills_path = str(Path(skills_path).resolve()) if skills_path else None
        self.time_budget_s = time_budget_s
        self.failure_limit = failure_limit
        self.startup_timeout_s = startup_timeout_s
        self.context = multiprocessing.get_context("spawn")
        self.connection = None
        self.worker = None
        self.hard_failures: dict[str, int] = {}

    def forecast(
        self, name: str, history: Sequence[float], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        failures = self.hard_failures.get(name, 0)
        if failures >= self.failure_limit:
            raise RuntimeError(
                f"hard failure circuit breaker after {failures} failures for {name}"
            )
        self._ensure_worker()
        try:
            self.connection.send((name, tuple(history), horizon, frequency))
        except (BrokenPipeError, EOFError, OSError) as error:
            self._hard_failure(name)
            raise RuntimeError(f"statistical worker crashed: {type(error).__name__}") from None
        if not self.connection.poll(self.time_budget_s):
            self._hard_failure(name)
            raise RuntimeError(f"hard timeout after {self.time_budget_s:g}s for {name}")
        try:
            status, payload = self.connection.recv()
        except (EOFError, OSError):
            self._hard_failure(name)
            raise RuntimeError(f"statistical worker crashed while running {name}") from None
        if status == SUCCESS:
            return _valid_forecast(payload, horizon)
        if status == NOT_APPLICABLE:
            raise _HistoryOnlyNotApplicable(str(payload))
        raise RuntimeError(str(payload))

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.connection.close()
        _stop_statistical_worker(self.worker)
        self.connection = None
        self.worker = None

    def _ensure_worker(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self.close()
        parent, child = self.context.Pipe()
        worker = self.context.Process(
            target=_statistical_worker,
            args=(self.module_path, self.skills_path, child),
            daemon=True,
        )
        worker.start()
        child.close()
        if not parent.poll(self.startup_timeout_s):
            parent.close()
            _stop_statistical_worker(worker)
            raise RuntimeError(
                f"statistical worker startup timeout after {self.startup_timeout_s:g}s"
            )
        try:
            status, detail = parent.recv()
        except (EOFError, OSError):
            parent.close()
            _stop_statistical_worker(worker)
            raise RuntimeError("statistical worker crashed during startup") from None
        if status != "ready":
            parent.close()
            _stop_statistical_worker(worker)
            raise RuntimeError(str(detail))
        self.connection = parent
        self.worker = worker

    def _hard_failure(self, name: str) -> None:
        self.hard_failures[name] = self.hard_failures.get(name, 0) + 1
        self.close()


def _statistical_worker(
    module_path: str, skills_path: str | None, connection: object
) -> None:
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stderr(sink):
        try:
            loaded, functions = load_methods(module_path, skills_path=skills_path)
            not_applicable = loaded.NotApplicable
            connection.send(("ready", ""))  # type: ignore[attr-defined]
            while True:
                request = connection.recv()  # type: ignore[attr-defined]
                if request is None:
                    return
                name, history, horizon, frequency = request
                try:
                    raw = functions[name](list(history), horizon, frequency)
                    connection.send((SUCCESS, _valid_forecast(raw, horizon)))  # type: ignore[attr-defined]
                except not_applicable as error:
                    connection.send((NOT_APPLICABLE, str(error)[:200]))  # type: ignore[attr-defined]
                except BaseException as error:
                    connection.send(  # type: ignore[attr-defined]
                        ("failed", f"{type(error).__name__}: {error}"[:200])
                    )
        except BaseException as error:
            try:
                connection.send(("startup_failed", f"{type(error).__name__}: {error}"[:200]))  # type: ignore[attr-defined]
            except BaseException:
                pass


def _stop_statistical_worker(worker: multiprocessing.Process | None) -> None:
    if worker is None:
        return
    if worker.is_alive():
        worker.terminate()
    worker.join(timeout=1.0)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=1.0)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    repo = Path(args.repo).resolve()
    screening_dir = Path(args.screening_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    screening_path = screening_dir / "frozen_screening_policy.py"
    screening_manifest = read_json_object(screening_dir / "screening_manifest.json")
    actual_screening_hash = _sha256(screening_path)
    if screening_manifest.get("frozen_screening_policy_sha256") != actual_screening_hash:
        raise ValueError("frozen screening policy hash does not match its manifest")
    screening = parse_screening_source(screening_path.read_text(encoding="utf-8"))
    train, dev = load_frozen_partitions(
        args.split_file, args.tasks_file,
        train_limit=args.train_limit, dev_limit=args.dev_limit,
    )
    entity_by_task = {
        source.task_id: source.entity_name
        for source in load_tasks(args.tasks_file)
    }
    module = read_module(repo / "methods.py")
    portfolio = read_policy_file(repo / "policies.py")
    final_outcomes, final_cache = _training_outcomes(
        args, repo, module, portfolio, train + dev
    )
    final_by_key = {(row.method, row.task_id): row for row in final_outcomes}
    runtimes = _runtime_registry(args)
    try:
        store = ForecastStore(
            args.hindcast_cache_dir,
            repo / "methods.py",
            repo / "skills.py" if (repo / "skills.py").is_file() else None,
            module,
            portfolio,
            runtimes,
            actual_screening_hash,
        )
        try:
            config = HindcastConfig(folds=args.folds)
            cases = tuple(
                _build_case(
                    task,
                    screening,
                    actual_screening_hash,
                    final_by_key,
                    store,
                    config,
                    group_id=entity_by_task.get(task.task_id, task.task_id),
                )
                for task in train + dev
            )
        finally:
            store.close()
    finally:
        runtimes.close()

    train_cases = cases[: len(train)]
    dev_cases = cases[len(train):]
    _write_cases(output / "train_decision_cases.jsonl", train_cases)
    _write_cases(output / "dev_decision_cases.jsonl", dev_cases)
    parent = DecisionPolicy()
    agent = CodexCLIClient(CodexCLIConfig(
        model=args.codex_model,
        reasoning_effort=args.codex_reasoning_effort,
        timeout_seconds=args.codex_timeout,
        cache_dir=args.codex_cache_dir or output / "agent-cache",
    ))
    evolution = evolve_selector_train_then_dev(
        parent,
        train_cases,
        dev_cases,
        agent,
        generations=args.generations,
        available_hindcast_folds=args.folds,
        train_validation_folds=args.train_validation_folds,
        screening_policy_hash=actual_screening_hash,
        transcript_dir=output / "transcripts",
    )
    generations = []
    for result in evolution.generations:
        generation = result.generation
        source = render_decision_source(result.child, screening_policy_hash=actual_screening_hash)
        (output / f"generation_{generation:03d}_child_decision_policy.py").write_text(
            source, encoding="utf-8"
        )
        proposal_source = render_decision_source(
            result.proposal, screening_policy_hash=actual_screening_hash
        )
        (output / f"generation_{generation:03d}_proposal_decision_policy.py").write_text(
            proposal_source, encoding="utf-8"
        )
        payload = {
            "generation": generation,
            "accepted": result.accepted,
            "gate": asdict(result.gate),
            "train_parent": asdict(result.train_parent),
            "train_child": asdict(result.train_child),
            "candidate_count": result.candidate_count,
            "parent_hash": decision_policy_hash(
                result.parent, screening_policy_hash=actual_screening_hash
            ),
            "proposal_hash": decision_policy_hash(
                result.proposal, screening_policy_hash=actual_screening_hash
            ),
            "child_hash": decision_policy_hash(result.child, screening_policy_hash=actual_screening_hash),
        }
        write_json(output / f"generation_{generation:03d}_selector_result.json", payload)
        generations.append(payload)

    frozen_path = output / "frozen_decision_policy.py"
    frozen_path.write_text(
        render_decision_source(evolution.frozen, screening_policy_hash=actual_screening_hash),
        encoding="utf-8",
    )
    final_train = evaluate_decision(evolution.frozen, train_cases)
    final_dev = evaluate_decision(evolution.frozen, dev_cases)
    manifest = {
        "schema_version": 1,
        "phase": "task_conditioned_numerical_selector",
        "train_tasks": len(train),
        "dev_tasks": len(dev),
        "train_validation_folds": args.train_validation_folds,
        "screening_policy_sha256": actual_screening_hash,
        "frozen_decision_policy_sha256": _sha256(frozen_path),
        "train": asdict(final_train),
        "dev": asdict(final_dev),
        "train_search_parent": asdict(evolution.train_parent),
        "train_search_winner": asdict(evolution.train_winner_score),
        "dev_parent": asdict(evolution.dev_parent),
        "dev_train_winner": asdict(evolution.dev_winner),
        "final_dev_gate": asdict(evolution.final_gate),
        "dev_accepted": evolution.final_gate.accepted,
        "train_winner_sha256": decision_policy_hash(
            evolution.train_winner, screening_policy_hash=actual_screening_hash
        ),
        "frozen_global_ranking": list(
            _global_ranking(final_outcomes, tuple(task.task_id for task in train))
        ),
        "accepted_train_generations": [
            row["generation"] for row in generations if row["accepted"]
        ],
        "accepted_generations": (
            [row["generation"] for row in generations if row["accepted"]]
            if evolution.final_gate.accepted
            else []
        ),
        "generations": generations,
        "cache": {
            **final_cache,
            "hindcast_hits": store.hits,
            "hindcast_misses": store.misses,
        },
        "elapsed_seconds": time.monotonic() - started,
        "public_test_accessed": False,
    }
    write_json(output / "selector_manifest.json", manifest)
    (output / "SELECTOR_REPORT.md").write_text(_report(manifest), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _build_case(
    task,
    screening,
    screening_hash,
    final_by_key,
    store,
    config,
    *,
    group_id: str = "",
) -> DecisionCase:
    active = materialize_active_dictionary(screening, profile_task(task))
    diagnostics = {}
    forecasts = {}
    families = {}
    specs = [
        (candidate.name, candidate.family, candidate.matched_clause >= 0)
        for candidate in active.active
    ]
    known = {name for name, _, _ in specs}
    specs.extend(
        (name, "tsfm", False)
        for name in getattr(store, "tsfm", {})
        if name not in known
    )
    for name, family, _ in specs:
        families[name] = family
        diagnostics[name] = diagnose_candidate(
            task, name, family, store.forecast, config,
            screening_policy_hash=screening_hash,
            runtime_settings={"portfolio": "flagship5"},
        )
        outcome = final_by_key.get((name, task.task_id))
        if outcome is not None and outcome.status == SUCCESS:
            forecasts[name] = tuple(outcome.forecast)
    return DecisionCase(
        task,
        tuple(name for name, _, _ in specs),
        diagnostics,
        forecasts,
        families,
        tuple(name for name, _, matched in specs if matched),
        group_id,
    )


def _valid_forecast(raw: Sequence[float], horizon: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw)
    if len(values) != horizon or not all(map(math.isfinite, values)):
        raise ValueError("candidate returned an invalid forecast")
    return values


def _write_cases(path: Path, cases: Sequence[DecisionCase]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            payload = {
                "task_id": case.task.task_id,
                "active_names": list(case.active_names),
                "diagnostics": {name: asdict(value) for name, value in case.diagnostics.items()},
                "final_forecasts": {name: list(values) for name, values in case.forecasts.items()},
                "families": dict(case.families),
                "conditioned_names": list(case.conditioned_names),
                "group_id": case.group_id,
            }
            handle.write(json.dumps(_finite_json(payload), sort_keys=True, allow_nan=False) + "\n")


def _global_ranking(
    outcomes: Sequence[Outcome], task_ids: Sequence[str], *, failure_penalty: float = 20.0
) -> tuple[str, ...]:
    """Freeze the legacy cross-task ranker from Train labels only."""
    requested = tuple(task_ids)
    by_method: dict[str, dict[str, Outcome]] = {}
    for outcome in outcomes:
        if outcome.task_id in requested:
            by_method.setdefault(outcome.method, {})[outcome.task_id] = outcome
    scores = []
    for name, rows in by_method.items():
        values = []
        for task_id in requested:
            outcome = rows.get(task_id)
            values.append(
                float(outcome.mase)
                if outcome is not None and outcome.status == SUCCESS and outcome.mase is not None
                else failure_penalty
            )
        scores.append((sum(values) / len(values), name))
    return tuple(name for _, name in sorted(scores))


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(manifest: Mapping[str, object]) -> str:
    train = manifest["train"]
    dev = manifest["dev"]
    assert isinstance(train, Mapping) and isinstance(dev, Mapping)
    return "\n".join((
        "# Frozen Numerical Selector Report",
        "",
        f"- Train / Dev: {manifest['train_tasks']} / {manifest['dev_tasks']}",
        f"- Accepted generations: {manifest['accepted_generations']}",
        f"- Screening SHA-256: `{manifest['screening_policy_sha256']}`",
        f"- Decision SHA-256: `{manifest['frozen_decision_policy_sha256']}`",
        f"- Dev accepted: `{manifest.get('dev_accepted', False)}`",
        f"- Final Dev gate: {manifest.get('final_dev_gate', {}).get('reason', 'not recorded')}",
        f"- Public Test accessed: `{manifest['public_test_accessed']}`",
        "",
        "| Split | Coverage | Mean sMAE | sMAE SE | Mean sRMSE | sRMSE SE | P90/P95 sMAE | Clipped sMAE/sRMSE | Mean MASE | Mean MAE | Mean sMAPE | Oracle regret | Methods | Families | Ensemble | Assumptions | Verifier pool | Pool families | Assumption kinds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _score_row("Train", train),
        _score_row("Dev", dev),
        "",
    ))


def _score_row(label: str, score: Mapping[str, object]) -> str:
    return (
        f"| {label} | {score['coverage']:.4f} | {score['mean_smae']:.6f} | "
        f"{score['se_smae']:.6f} | {score['mean_srmse']:.6f} | {score['se_srmse']:.6f} | "
        f"{score['p90_smae']:.6f}/{score['p95_smae']:.6f} | "
        f"{score['smae_clipped_count']}/{score['srmse_clipped_count']} | "
        f"{score['mean_mase']:.6f} | {score['mean_mae']:.6f} | {score['mean_smape']:.6f} | "
        f"{score['mean_active_oracle_regret']:.6f} | "
        f"{score['method_diversity']} | {score['family_diversity']} | {score['ensemble_rate']:.4f} | "
        f"{score['mean_assumption_count']:.2f} | {score['mean_considered_candidates']:.2f} | "
        f"{score['mean_considered_families']:.2f} | {score['assumption_kind_diversity']} |"
    )


if __name__ == "__main__":
    raise SystemExit(main())
