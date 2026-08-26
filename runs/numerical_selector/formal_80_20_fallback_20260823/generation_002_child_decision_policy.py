"""Frozen task-conditioned numerical Decision policy."""

SCREENING_POLICY_HASH = '9fb0be569e2bc4597a206ba9e4b0fa44aa99f0abbb07bdee82c219e74fa95096'
DECISION_POLICY = {'ranking_order': ['median_mase', 'worst_mase', 'recent_mase', 'mase_mad', 'normalized_bias'],
 'recent_regime_first': False,
 'min_successful_folds': 4,
 'catastrophic_mase': 5.0,
 'ensemble_enabled': True,
 'ensemble_max_members': 2,
 'ensemble_min_diversity': 0.2,
 'ensemble_min_improvement': 0.05,
 'fallback_to_best_available': True}
