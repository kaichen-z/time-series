"""Optional single-writer task progress for cooperative evaluation.

This is evaluator resume, not permission to resume a terminal failed root run.
Only atomically completed, identity-matched, typed scores can be reused.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT
from evolving_loop.package_metrics import PackageTaskScore
from evolving_loop.package_registry import (
    numerical_package_fingerprint, task_registry_fingerprint,
)

# Bump for execution/scoring changes not already bound by the identities below.
TASK_EXECUTION_CONTRACT = 'cooperative-task-execution-v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _diagnostic_json(value):
    """Rejected tool inputs may contain NaN; keep it as text, never a score."""
    if isinstance(value, Mapping):
        return {key: _diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_diagnostic_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


class PackageTaskStore:
    def __init__(self, directory: str | Path, *, runtime_identity: str):
        if not isinstance(runtime_identity, str) or not runtime_identity:
            raise ValueError('task store requires a runtime identity')
        self.directory = Path(directory)
        self.runtime_identity = runtime_identity

    def identity(self, *, task, package, candidate_sha256, stage, metric_cap,
                 expected_retrieval_sha256, expected_decision_prompt_sha256):
        return {
            'task_id': task.numeric.task_id,
            'task_sha256': task_registry_fingerprint(task),
            'numerical_package_sha256': numerical_package_fingerprint(package),
            'candidate_sha256': candidate_sha256,
            'stage': stage,
            'runtime_identity': self.runtime_identity,
            'execution_contract': TASK_EXECUTION_CONTRACT,
            'metric_policy': METRIC_POLICY_FINGERPRINT,
            'metric_cap': metric_cap,
            'retrieval_sha256': expected_retrieval_sha256,
            'decision_prompt_sha256': expected_decision_prompt_sha256,
        }

    def load(self, identity):
        path = self.directory / f'{_digest(identity)}.json'
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('identity') != identity:
            raise ValueError('task progress identity mismatch')
        if record.get('status') != 'completed':
            return None
        row = PackageTaskScore(**record['score'])
        diagnostics = record['diagnostics']
        if (row.task_id != identity['task_id']
                or row.numerical_package_sha256 != identity['numerical_package_sha256']
                or type(diagnostics) is not dict
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in diagnostics.values())):
            raise ValueError('invalid completed task score binding or diagnostics')
        content = {key: record[key] for key in ('score', 'diagnostics', 'artifacts')}
        if record.get('result_sha256') != _digest(content):
            raise ValueError('completed task result digest mismatch')
        return row, diagnostics

    def write(self, identity, status, *, score=None, diagnostics=None, artifacts=None, error=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f'{_digest(identity)}.json'
        record = {
            'schema_version': 1, 'identity': identity, 'status': status,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        record['started_at'] = (
            json.loads(path.read_text(encoding='utf-8'))['started_at']
            if status != 'started' and path.exists() else record['updated_at']
        )
        if score is not None:
            content = {'score': score.to_payload(), 'diagnostics': dict(diagnostics), 'artifacts': _diagnostic_json(artifacts)}
            record.update(content, result_sha256=_digest(content))
        if error is not None:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
        temporary = path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
