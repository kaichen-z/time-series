"""Frozen task-conditioned numerical Decision policy."""

SCREENING_POLICY_HASH = '9fb0be569e2bc4597a206ba9e4b0fa44aa99f0abbb07bdee82c219e74fa95096'
DECISION_POLICY = {'ranking_order': ['worst_mase', 'median_mase', 'mase_mad', 'recent_mase', 'normalized_bias'],
 'recent_regime_first': False,
 'min_successful_folds': 3,
 'catastrophic_mase': 5.0,
 'ensemble_enabled': True,
 'ensemble_max_members': 2,
 'ensemble_min_diversity': 0.2,
 'ensemble_min_improvement': 0.1,
 'ensemble_weight_grid': [0.9, 0.95],
 'ensemble_residual_strengths': [0.05, 0.1],
 'ensemble_correction_clip': 0.5,
 'ensemble_min_fold_wins': 3,
 'ensemble_max_worst_fold_regret': 0.01,
 'fallback_to_best_available': True}
