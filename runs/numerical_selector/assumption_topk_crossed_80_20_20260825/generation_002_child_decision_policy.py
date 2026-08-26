"""Frozen task-conditioned numerical Decision policy."""

SCREENING_POLICY_HASH = '2277acd9ef0b5e219960e84aa1b02fbfa13bf1b54cf106e26b00c5f3fc2babc2'
DECISION_POLICY = {'ranking_order': ['recent_mase', 'worst_mase', 'median_mase', 'mase_mad', 'normalized_bias'],
 'recent_regime_first': True,
 'min_successful_folds': 3,
 'catastrophic_mase': 5.0,
 'baseline_strategy': 'minimax_tsfm',
 'assumption_guidance_enabled': True,
 'assumption_top_k': 3,
 'assumption_candidates_per_hypothesis': 2,
 'assumption_min_confidence': 0.5,
 'ensemble_enabled': True,
 'ensemble_max_members': 2,
 'ensemble_min_diversity': 0.1,
 'ensemble_min_improvement': 0.05,
 'ensemble_weight_grid': [],
 'ensemble_residual_strengths': [0.1, 0.25],
 'ensemble_correction_clip': 0.5,
 'ensemble_min_fold_wins': 3,
 'ensemble_max_worst_fold_regret': 0.02,
 'fallback_to_best_available': True}
