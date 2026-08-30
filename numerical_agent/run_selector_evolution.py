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
from common.evolution_core.contracts import (
    METRIC_POLICY_FINGERPRINT,
    metric_policy_metadata,
    metric_report_metadata,
    require_active_metric_policy,
    load_active_release,
)
from common.llm import CodexCLIClient, CodexCLIConfig
from common.metrics import joint_scaled_error
from common.payload import (
    read_json_object,
    standards_json_value,
    strict_json_loads,
    write_json,
)

from .evolution.execution import CRASHED, INVALID, NOT_APPLICABLE, SUCCESS, Outcome, Task, load_methods
from .evolution.module import MethodModule, read_module
from .evolution.numerical_selector import (
    DecisionPolicy,
    HindcastConfig,
    diagnose_candidate,
)
from .evolution.portfolio import (
    CombinedPolicy,
    InvalidTSFMForecastError,
    PolicyNotApplicable,
    PolicyPortfolio,
    TSFMPolicy,
    combine_materialized_outcome,
    forecast_tsfm,
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


class CacheIntegrityError(ValueError):
    """An existing active cache row is corrupt and must abort the lifecycle."""


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
        runtime_identity: Mapping[str, object] | None = None,
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
        identity = {
            "module_source": module_path.read_text(encoding="utf-8"),
            "skills_source": (
                skills_path.read_text(encoding="utf-8") if skills_path is not None else None
            ),
            "portfolio": asdict(portfolio),
            "reviewed_manifests_sha256": _sha256(
                Path(__file__).parent / "tsfm" / "runtime_manifests.json"
            ),
            "runtime": dict(runtime_identity or {}),
        }
        self.identity_hash = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        ).hexdigest()
        self.hits = 0
        self.misses = 0
        self._materialized_leaf_outcomes: dict[str, Outcome] = {}

    def forecast(
        self, name: str, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        key = self._key(name, history, horizon, frequency)
        path = self.root / f"{key}.json"
        if not path.exists():
            payload = None
        else:
            try:
                payload = strict_json_loads(
                    path.read_text(encoding="utf-8"),
                    context="active hindcast cache row",
                )
            except (OSError, TypeError, ValueError) as error:
                raise CacheIntegrityError(
                    "active hindcast cache row is malformed"
                ) from error
        if payload is not None:
            try:
                if not isinstance(payload, Mapping):
                    raise CacheIntegrityError(
                        "active hindcast cache row must be an object"
                    )
                require_active_metric_policy(payload, context="active hindcast cache row")
                if (
                    type(payload.get("cache_schema")) is not int
                    or payload["cache_schema"] != 3
                ):
                    raise CacheIntegrityError(
                        "active hindcast cache row schema mismatch"
                    )
                if payload.get("key") != key:
                    raise CacheIntegrityError("active hindcast cache row key mismatch")
                if payload.get("status") == SUCCESS:
                    values = tuple(float(value) for value in payload["forecast"])
                    if len(values) != horizon or not all(map(math.isfinite, values)):
                        raise CacheIntegrityError(
                            "active hindcast cache row forecast is noncanonical"
                        )
                    self.hits += 1
                    return values
                if payload.get("status") == NOT_APPLICABLE:
                    self.hits += 1
                    raise self.not_applicable(
                        str(payload.get("detail", "not applicable"))
                    )
                raise CacheIntegrityError(
                    "active hindcast cache row status is noncanonical"
                )
            except CacheIntegrityError:
                raise
            except self.not_applicable:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise CacheIntegrityError(
                    str(error) or "active hindcast cache row is noncanonical"
                ) from error
        self.misses += 1
        try:
            values = self._execute(name, history, horizon, frequency)
        except self.not_applicable as error:
            self._write(path, {
                "cache_schema": 3,
                **metric_policy_metadata(),
                "key": key,
                "status": NOT_APPLICABLE,
                "detail": str(error)[:200],
            })
            raise
        self._write(path, {
            "cache_schema": 3,
            **metric_policy_metadata(),
            "key": key,
            "status": SUCCESS,
            "forecast": list(values),
        })
        return values

    def _execute(
        self, name: str, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        if name in self.statistical_names:
            return self._statistical.forecast(name, history, horizon, frequency)
        if policy := self.tsfm.get(name):
            try:
                return forecast_tsfm(
                    policy,
                    history=history,
                    horizon=horizon,
                    frequency=frequency,
                    runtimes=self.runtimes,
                )
            except PolicyNotApplicable as error:
                raise self.not_applicable(str(error)) from None
            except InvalidTSFMForecastError as error:
                raise _StructuralForecastInvalid(str(error)) from None
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
        parent_outcomes = {
            parent: self._materialized_leaf_outcome(parent, history, horizon, frequency)
            for parent in policy.parents
        }
        combined = combine_materialized_outcome(
            policy,
            parent_outcomes,
            task_id="history-only",
            history=history,
            horizon=horizon,
            frequency=frequency,
        )
        if combined.status == SUCCESS:
            return _valid_forecast(combined.forecast, horizon)
        if combined.status == NOT_APPLICABLE:
            raise self.not_applicable(combined.detail)
        if combined.status == INVALID:
            raise ValueError(combined.detail or combined.status)
        raise RuntimeError(combined.detail or combined.status)

    def _materialized_leaf_outcome(
        self,
        name: str,
        history: tuple[float, ...],
        horizon: int,
        frequency: str,
    ) -> Outcome:
        """Resolve one cached leaf; Combined policies cannot be parents in this graph."""
        key = self._key(name, history, horizon, frequency)
        if key in self._materialized_leaf_outcomes:
            return self._materialized_leaf_outcomes[key]
        try:
            outcome = Outcome(
                name,
                "history-only",
                SUCCESS,
                forecast=self.forecast(name, history, horizon, frequency),
            )
        except CacheIntegrityError:
            raise
        except self.not_applicable as error:
            outcome = Outcome(
                name, "history-only", NOT_APPLICABLE, detail=str(error)[:200]
            )
        except _StructuralForecastInvalid as error:
            outcome = Outcome(name, "history-only", INVALID, detail=str(error)[:200])
        except ValueError as error:
            if name in self.tsfm:
                outcome = Outcome(
                    name,
                    "history-only",
                    CRASHED,
                    detail=f"{type(error).__name__}: {error}"[:200],
                )
            else:
                outcome = Outcome(
                    name, "history-only", INVALID, detail=str(error)[:200]
                )
        except Exception as error:
            outcome = Outcome(
                name,
                "history-only",
                CRASHED,
                detail=f"{type(error).__name__}: {error}"[:200],
            )
        self._materialized_leaf_outcomes[key] = outcome
        return outcome

    def _key(self, name, history, horizon, frequency) -> str:
        payload = json.dumps({
            "schema": 3,
            "metric_policy_fingerprint": METRIC_POLICY_FINGERPRINT,
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


class _StructuralForecastInvalid(ValueError):
    """A known finite/horizon failure that maps to the canonical INVALID status."""


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
    load_active_release(screening_manifest)
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
            runtime_identity=_forecast_runtime_identity(args),
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
            "schema_version": 2,
            **metric_report_metadata(),
            "generation": generation,
            "accepted": result.accepted,
            "gate": asdict(result.gate),
            "train_parent": asdict(result.train_parent),
            "train_child": asdict(result.train_child),
            "paired_joint_wtl": _score_pair_wtl(
                result.train_parent, result.train_child
            ),
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
        "schema_version": 2,
        **metric_report_metadata(),
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
        "paired_joint_wtl": {
            "train": _score_pair_wtl(evolution.train_parent, evolution.train_winner_score),
            "dev": _score_pair_wtl(evolution.dev_parent, evolution.dev_winner),
        },
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
    print(json.dumps(
        standards_json_value(manifest),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))
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
                "schema_version": 2,
                **metric_policy_metadata(),
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
    outcomes: Sequence[Outcome], task_ids: Sequence[str], *, failure_penalty: float = 5.0
) -> tuple[str, ...]:
    """Freeze the canonical joint scaled-error ranker from Train labels only."""
    requested = tuple(task_ids)
    by_method: dict[str, dict[str, Outcome]] = {}
    for outcome in outcomes:
        if outcome.task_id in requested:
            by_method.setdefault(outcome.method, {})[outcome.task_id] = outcome
    scores = []
    for name, rows in by_method.items():
        smaes = []
        srmses = []
        for task_id in requested:
            outcome = rows.get(task_id)
            if outcome is not None and outcome.status == SUCCESS:
                if outcome.smae is None or outcome.srmse is None:
                    raise ValueError(
                        "active global ranking cannot consume legacy metric policy outcomes"
                    )
                smaes.append(float(outcome.smae))
                srmses.append(float(outcome.srmse))
            else:
                smaes.append(failure_penalty)
                srmses.append(failure_penalty)
        mean_smae = sum(smaes) / len(smaes)
        mean_srmse = sum(srmses) / len(srmses)
        scores.append((joint_scaled_error(mean_smae, mean_srmse), name))
    return tuple(name for _, name in sorted(scores))


def _score_pair_wtl(parent, child) -> dict[str, int]:
    result = {"wins": 0, "ties": 0, "losses": 0, "missing": 0, "unscored": 0}
    parent_pairs = parent.task_scaled_pairs
    child_pairs = child.task_scaled_pairs
    observed = set(parent_pairs) | set(child_pairs)
    for task_id in sorted(observed):
        if task_id not in parent_pairs or task_id not in child_pairs:
            result["missing"] += 1
            continue
        parent_joint = joint_scaled_error(*parent_pairs[task_id])
        child_joint = joint_scaled_error(*child_pairs[task_id])
        if child_joint < parent_joint - 1e-12:
            result["wins"] += 1
        elif child_joint > parent_joint + 1e-12:
            result["losses"] += 1
        else:
            result["ties"] += 1
    expected = max(parent.task_count, child.task_count, len(observed))
    result["unscored"] = expected - len(observed)
    return result


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("active hindcast payload cannot contain NaN")
        return {
            "status": "positive_infinity" if value > 0 else "negative_infinity",
            "value": None,
        }
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forecast_runtime_identity(args: argparse.Namespace) -> dict[str, object]:
    """Canonical provider/deployment/checkpoint identity for hindcast cache keys."""
    deployment = getattr(args, "tsfm_workers_config", None)
    deployment_hash = None
    if deployment:
        deployment_path = Path(deployment).expanduser().resolve()
        deployment_hash = _sha256(deployment_path)
    return {
        "tsfm_runtimes": getattr(args, "tsfm_runtimes", "") or "",
        "chronos_device_map": getattr(args, "chronos_device_map", "cpu"),
        "model_cache_dir": str(getattr(args, "model_cache_dir", None) or ""),
        "deployment_sha256": deployment_hash,
        "acknowledged_model_licenses": getattr(
            args, "acknowledged_model_licenses", ""
        ) or "",
    }


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
        f"- Metric policy SHA-256: `{manifest['metric_policy_fingerprint']}`",
        "- Primary metrics: sMAE, sRMSE",
        "- Diagnostic only: MASE, MAE, sMAPE, RMSSE",
        f"- Dev accepted: `{manifest.get('dev_accepted', False)}`",
        f"- Final Dev gate: {manifest.get('final_dev_gate', {}).get('reason', 'not recorded')}",
        f"- Public Test accessed: `{manifest['public_test_accessed']}`",
        "",
        "| Split | Coverage | Mean sMAE | Median sMAE | sMAE SE | Mean sRMSE | Median sRMSE | sRMSE SE | P90/P95 sMAE | P90/P95 sRMSE | Raw P90/P95 sMAE | Raw P90/P95 sRMSE | Clipped sMAE/sRMSE | Oracle sMAE/sRMSE regret | Methods | Families | Ensemble | Assumptions | Verifier pool | Pool families | Assumption kinds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _score_row("Train", train),
        _score_row("Dev", dev),
        "",
        "| Split | Wins / Ties / Losses / Missing / Unscored |",
        "|---|---:|",
        _paired_report_row("Train", manifest["paired_joint_wtl"]["train"]),
        _paired_report_row("Dev", manifest["paired_joint_wtl"]["dev"]),
        "",
        "## Diagnostic-only metrics",
        "",
        "| Split | Mean MASE | Mean MAE | Mean sMAPE |",
        "|---|---:|---:|---:|",
        f"| Train | {train['mean_mase']:.6f} | {train['mean_mae']:.6f} | {train['mean_smape']:.6f} |",
        f"| Dev | {dev['mean_mase']:.6f} | {dev['mean_mae']:.6f} | {dev['mean_smape']:.6f} |",
        "",
    ))


def _score_row(label: str, score: Mapping[str, object]) -> str:
    return (
        f"| {label} | {score['coverage']:.4f} | {score['mean_smae']:.6f} | "
        f"{score['median_smae']:.6f} | {score['se_smae']:.6f} | "
        f"{score['mean_srmse']:.6f} | {score['median_srmse']:.6f} | {score['se_srmse']:.6f} | "
        f"{score['p90_smae']:.6f}/{score['p95_smae']:.6f} | "
        f"{score['p90_srmse']:.6f}/{score['p95_srmse']:.6f} | "
        f"{score['p90_smae_raw']:.6f}/{score['p95_smae_raw']:.6f} | "
        f"{score['p90_srmse_raw']:.6f}/{score['p95_srmse_raw']:.6f} | "
        f"{score['smae_clipped_count']}/{score['srmse_clipped_count']} | "
        f"{score['mean_active_oracle_smae_regret']:.6f}/"
        f"{score['mean_active_oracle_srmse_regret']:.6f} | "
        f"{score['method_diversity']} | {score['family_diversity']} | {score['ensemble_rate']:.4f} | "
        f"{score['mean_assumption_count']:.2f} | {score['mean_considered_candidates']:.2f} | "
        f"{score['mean_considered_families']:.2f} | {score['assumption_kind_diversity']} |"
    )


def _paired_report_row(label: str, counts: Mapping[str, object]) -> str:
    return (
        f"| {label} | {counts['wins']} / {counts['ties']} / {counts['losses']} / "
        f"{counts['missing']} / {counts['unscored']} |"
    )


if __name__ == "__main__":
    raise SystemExit(main())
