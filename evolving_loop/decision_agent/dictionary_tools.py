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
