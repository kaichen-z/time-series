"""Frozen task-conditioned numerical Decision policy."""

SCREENING_POLICY_HASH = '2277acd9ef0b5e219960e84aa1b02fbfa13bf1b54cf106e26b00c5f3fc2babc2'
DECISION_POLICY = {'ranking_order': ['median_mase', 'worst_mase', 'recent_mase', 'mase_mad', 'normalized_bias'],
 'recent_regime_first': False,
 'min_successful_folds': 3,
 'catastrophic_mase': 5.0,
 'ensemble_enabled': True,
 'ensemble_max_members': 1,
 'ensemble_min_diversity': 0.2,
 'ensemble_min_improvement': 0.1,
 'ensemble_weight_grid': [0.65, 0.75],
 'ensemble_residual_strengths': [0.05, 0.15],
 'ensemble_correction_clip': 0.5,
 'ensemble_min_fold_wins': 3,
 'ensemble_max_worst_fold_regret': 0.01,
 'fallback_to_best_available': True}
