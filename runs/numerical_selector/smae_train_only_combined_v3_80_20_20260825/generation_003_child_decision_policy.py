"""Frozen task-conditioned numerical Decision policy."""

SCREENING_POLICY_HASH = '9fb0be569e2bc4597a206ba9e4b0fa44aa99f0abbb07bdee82c219e74fa95096'
DECISION_POLICY = {'ranking_order': ['worst_mase', 'median_mase', 'recent_mase', 'mase_mad', 'normalized_bias'],
 'recent_regime_first': True,
 'min_successful_folds': 3,
 'catastrophic_mase': 4.0,
 'ensemble_enabled': True,
 'ensemble_max_members': 2,
 'ensemble_min_diversity': 0.05,
 'ensemble_min_improvement': 0.0,
 'ensemble_weight_grid': [0.7, 0.8, 0.9],
 'ensemble_residual_strengths': [],
 'ensemble_correction_clip': 0.25,
 'ensemble_min_fold_wins': 2,
 'ensemble_max_worst_fold_regret': 0.1,
 'fallback_to_best_available': True}
