"""Frozen task-conditioned numerical Decision policy."""

SCREENING_POLICY_HASH = '5727d1c9bce4fd1ac509593fff455597fdedc3b11c8fa18f39b97e00facf37a4'
DECISION_POLICY = {'ranking_order': ['recent_mase', 'median_mase', 'worst_mase', 'mase_mad', 'normalized_bias'],
 'recent_regime_first': True,
 'min_successful_folds': 3,
 'catastrophic_mase': 8.0,
 'ensemble_enabled': True,
 'ensemble_max_members': 2,
 'ensemble_min_diversity': 0.2,
 'ensemble_min_improvement': 0.08,
 'ensemble_weight_grid': [0.7, 0.85, 0.95],
 'ensemble_residual_strengths': [0.1, 0.2],
 'ensemble_correction_clip': 0.5,
 'ensemble_min_fold_wins': 3,
 'ensemble_max_worst_fold_regret': 0.01,
 'fallback_to_best_available': True}
