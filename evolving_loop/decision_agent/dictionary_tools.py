"""Decision-owned evaluation of methods from a frozen P2 execution cache.

Only historical fold targets enter this tool. Task future labels are neither
accepted nor needed. Cached method executions are combined and scored afresh.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics

from common.metrics import drcik_point_metrics
from evolving_loop.decision_agent.agent import DecisionCandidate


def decision_dictionary_fingerprints(package, dictionary_sha):
    """Bind cache-backed execution, not nonexistent legacy Selector configs.

    The schema tag distinguishes these descriptors from fresh-fit policies.
    Consumers recompute them from the frozen package before any agent call.
    """
    from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                        ensure_ascii=False, allow_nan=False).encode()).hexdigest()

    rows = [dict(name=m.name, family=m.family, forecast=list(m.forecast),
                 eligible=m.diagnostics.eligible,
                 fold_forecasts=[list(v) for v in m.diagnostics.fold_forecasts],
                 fold_truths=[list(v) for v in m.diagnostics.fold_truths])
            for m in package.ranked_alternatives]
    schema = 'decision_dictionary_cache_v1'
    return {
        'decision_cache_contract': digest(schema),
        'decision_dictionary': dictionary_sha,
        'metric_policy_fingerprint': METRIC_POLICY_FINGERPRINT,
        'task_profile': digest(package.task_profile.to_public_payload()),
        'active_dictionary': digest({'schema': schema, 'dictionary': dictionary_sha,
                                     'methods': [r['name'] for r in rows]}),
        'screening_policy': digest({'schema': schema, 'selection': 'all_materialized_methods'}),
        'combined_policies': digest({'schema': schema, 'combination': 'nonnegative_sum_one'}),
        'decision_policy': digest({'schema': schema, 'owner': 'decision_agent',
                                   'baseline': package.protected_baseline.name,
                                   'max_evaluations_per_batch': 4}),
        'hindcast_config': digest({'schema': schema, 'mode': 'frozen_fold_replay', 'rows': rows}),
    }


class DictionaryExecutionTool:
    def __init__(self, methods, *, history=(), frequency=None):
        methods = tuple(methods)
        self._methods = {item.name: item for item in methods}
        if not methods or len(self._methods) != len(methods):
            raise ValueError('Dictionary methods must be nonempty and unique')
        self.history = tuple(history)
        self.frequency = frequency

    def catalog(self):
        return [dict(method_id=item.name, family=item.family,
                     eligible=item.diagnostics.eligible,
                     successful_folds=item.diagnostics.successful_folds)
                for item in self._methods.values()]

    def evaluate(self, request):
        if type(request) is not dict or set(request) != {'method_ids', 'weights'}:
            raise ValueError('Expected method_ids and weights')
        names, weights = request['method_ids'], request['weights']
        if (type(names) is not list or not names
                or any(type(name) is not str for name in names)
                or len(set(names)) != len(names)
                or not set(names) <= set(self._methods)
                or type(weights) is not list or len(names) != len(weights)
                or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights)
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-8)):
            raise ValueError('Invalid methods or convex weights')
        methods = [self._methods[name] for name in names]
        truths = methods[0].diagnostics.fold_truths
        if not truths or any(not truth for truth in truths):
            raise ValueError('History-only hindcast evidence is required')
        if any(not m.diagnostics.eligible or m.diagnostics.fold_truths != truths
               or len(m.diagnostics.fold_forecasts) != len(truths) for m in methods):
            raise ValueError('Methods require eligible aligned historical folds')

        def combine(vectors):
            length = len(vectors[0])
            if not length or any(len(v) != length or any(not math.isfinite(x) for x in v) for v in vectors):
                raise ValueError('Invalid forecast vectors')
            return tuple(sum(w * v[i] for w, v in zip(weights, vectors)) for i in range(length))

        forecast = combine([m.forecast for m in methods])
        scores = [drcik_point_metrics(truth, combine([m.diagnostics.fold_forecasts[i] for m in methods]))
                  for i, truth in enumerate(truths)]
        identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
        return DecisionCandidate(
            candidate_id='dictionary_' + identity, forecast=forecast,
            assumption='Decision-selected frozen methods: ' + ', '.join(names),
            failure_condition='Historical method behavior does not transfer to the forecast window.',
            hindcast_smae=statistics.mean(s['smae'] for s in scores),
            hindcast_srmse=statistics.mean(s['srmse'] for s in scores),
            tags=('dictionary_execution',),
        )
