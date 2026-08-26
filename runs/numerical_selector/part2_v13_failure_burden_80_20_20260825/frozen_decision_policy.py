"""Frozen task-conditioned numerical Decision policy."""

SCREENING_POLICY_HASH = '5727d1c9bce4fd1ac509593fff455597fdedc3b11c8fa18f39b97e00facf37a4'
DECISION_POLICY = {'ranking_order': ['median_mase', 'recent_mase', 'worst_mase', 'mase_mad', 'normalized_bias'],
 'recent_regime_first': True,
 'min_successful_folds': 3,
 'catastrophic_mase': 10.0,
 'ensemble_enabled': True,
 'ensemble_max_members': 2,
 'ensemble_min_diversity': 0.1,
 'ensemble_min_improvement': 0.05,
 'ensemble_weight_grid': [0.8, 0.9],
 'ensemble_residual_strengths': [0.1, 0.25],
 'ensemble_correction_clip': 1.0,
 'ensemble_min_fold_wins': 2,
 'ensemble_max_worst_fold_regret': 0.02,
 'fallback_to_best_available': True}
