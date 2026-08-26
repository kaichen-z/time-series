"""Frozen task-conditioned numerical Decision policy."""

SCREENING_POLICY_HASH = '2277acd9ef0b5e219960e84aa1b02fbfa13bf1b54cf106e26b00c5f3fc2babc2'
DECISION_POLICY = {'ranking_order': ['recent_mase', 'worst_mase', 'median_mase', 'median_rmsse', 'mase_mad'],
 'recent_regime_first': True,
 'min_successful_folds': 3,
 'catastrophic_mase': 6.0,
 'baseline_strategy': 'toto_first',
 'ensemble_enabled': True,
 'ensemble_max_members': 2,
 'ensemble_min_diversity': 0.05,
 'ensemble_min_improvement': 0.005,
 'ensemble_weight_grid': [0.6, 0.75, 0.9],
 'ensemble_residual_strengths': [0.1, 0.2],
 'ensemble_correction_clip': 0.35,
 'ensemble_min_fold_wins': 1,
 'ensemble_max_worst_fold_regret': 0.1,
 'fallback_to_best_available': True}
