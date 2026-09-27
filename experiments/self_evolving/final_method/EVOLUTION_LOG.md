# Evolution log: per-generation results and changes

## 1. Retrieval extraction instructions (tl2_evolve.py)

- Fitness: evidence F1 against `gt_evidence` on a 30-task train minibatch.
- Mutator: GPT-6-sol; 2 children per generation; a child is accepted only if F1 improves.

| Generation | Child F1 | Accepted? | Current best F1 |
|---|---|---|---|
| 0 (initial instructions) | — | — | 0.054 |
| 1 | 0.206, 0.014 | yes | 0.206 |
| 2 | 0.248, 0.276 | yes | 0.276 |
| 3 | 0.292, 0.258 | yes | 0.292 |
| 4 | 0.170, 0.464 | yes | 0.464 |

Full training set with the final instructions: F1 0.039 → 0.477, recall 5.4% → 48.3%, precision 9.1% → 67.0%. Only the final instruction text is saved (`artifacts/tl2_evolved_extraction_instructions.json`); intermediate texts were not stored.

## 2. Three-agent co-evolution (nrd4.py, full train, 3 seeds)

Team fitness = mean over 3 stratified train folds of (mean gain + 0.5 × mean negative gain) − 0.25 × fold std.
Generation 0 = exact Toto (fitness 0). A new team is kept only if its fitness beats the current team; otherwise the round rolls back.

### Seed 1

| Gen | Team fitness | Fold scores | Accepted? | Numerical calibrator (mult) | Decision (strength, trust gate) |
|---|---|---|---|---|---|
| 0 | +0.0000 | +0.000, +0.000, +0.000 | — | — | — |
| 1 | +0.0710 | +0.091, +0.025, +0.129 | **yes** | 1.000 | 0.68, 0.00 |
| 2 | +0.0754 | +0.080, +0.022, +0.170 | **yes** | 1.033 | 0.91, 0.00 |
| 3 | +0.0809 | +0.086, +0.022, +0.185 | **yes** | 1.033 | 1.00, 0.00 |
| 4 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 5 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 6 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 7 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 8 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 9 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 10 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 11 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 12 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 13 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 14 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 15 | +0.0820 | +0.092, +0.022, +0.182 | **yes** | 1.022 | 1.00, 0.00 |

Retrieval validator self-evolution (per-correction credit, first round): 3.5879 → 5.2852 → 6.2187 → 6.2187 → 6.2187 → 6.2187
Numerical calibrator self-evolution (history-only error, first round): 0.6707 → 0.6707 → 0.6707 → 0.6707; last round: 0.6694 → 0.6694 → 0.6694 → 0.6694
Final validator weights: absmag +3.25, conf -0.07, wfrac +0.63, docbase -2.77, ratio -0.01, trust +1.02, bias -2.75

### Seed 2

| Gen | Team fitness | Fold scores | Accepted? | Numerical calibrator (mult) | Decision (strength, trust gate) |
|---|---|---|---|---|---|
| 0 | +0.0000 | +0.000, +0.000, +0.000 | — | — | — |
| 1 | +0.0807 | +0.082, +0.025, +0.185 | **yes** | 1.000 | 1.00, 0.07 |
| 2 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 3 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 4 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 5 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 6 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 7 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 8 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 9 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 10 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 11 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 12 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 13 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 14 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 15 | +0.0820 | +0.092, +0.022, +0.182 | **yes** | 1.000 | 1.00, 0.07 |

Retrieval validator self-evolution (per-correction credit, first round): 0.0 → 3.9185 → 6.5352 → 6.8236 → 6.8236 → 7.6951
Numerical calibrator self-evolution (history-only error, first round): 0.6707 → 0.6707 → 0.6707 → 0.6707; last round: 0.6696 → 0.6696 → 0.6696 → 0.6696
Final validator weights: absmag +1.49, conf +0.58, wfrac +0.66, docbase -2.38, ratio +0.03, trust +0.34, bias -1.55

### Seed 3

| Gen | Team fitness | Fold scores | Accepted? | Numerical calibrator (mult) | Decision (strength, trust gate) |
|---|---|---|---|---|---|
| 0 | +0.0000 | +0.000, +0.000, +0.000 | — | — | — |
| 1 | +0.0790 | +0.082, +0.021, +0.185 | **yes** | 1.000 | 1.00, 0.16 |
| 2 | +0.0801 | +0.085, +0.021, +0.185 | **yes** | 1.000 | 1.00, 0.16 |
| 3 | +0.0802 | +0.086, +0.021, +0.185 | **yes** | 1.027 | 1.00, 0.16 |
| 4 | +0.0808 | +0.086, +0.022, +0.185 | **yes** | 1.027 | 1.00, 0.16 |
| 5 | +0.0809 | +0.086, +0.022, +0.185 | **yes** | 1.027 | 1.00, 0.05 |
| 6 | +0.0828 | +0.092, +0.022, +0.185 | **yes** | 1.027 | 1.00, 0.07 |
| 7 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 8 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 9 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 10 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 11 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 12 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 13 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 14 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 15 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |

Retrieval validator self-evolution (per-correction credit, first round): 3.2516 → 5.0894 → 6.726 → 7.3474 → 7.5743 → 7.585
Numerical calibrator self-evolution (history-only error, first round): 0.6707 → 0.6702 → 0.6702 → 0.6702; last round: 0.6694 → 0.6694 → 0.6694 → 0.6694
Final validator weights: absmag +3.87, conf -1.18, wfrac +1.07, docbase -3.72, ratio +0.02, trust +0.77, bias -1.75

## 3. Numerical dictionary self-evolution (stage 1 of nrd11)

- Value per generation = mean over (frequency × seasonality) cells of [elite hindcast joint error − Toto hindcast joint error].
- Negative means the elites beat Toto. History only, no labels.

- seed 1: +0.092 → +0.016 → -0.006 → -0.009 → -0.024 → -0.025 → -0.030 → -0.033 → -0.067 → -0.069 → -0.069 → -0.069 → -0.069 → -0.069 → -0.069 → -0.069
- seed 2: +0.092 → +0.012 → +0.003 → +0.003 → -0.038 → -0.038 → -0.048 → -0.048 → -0.056 → -0.062 → -0.062 → -0.062 → -0.062 → -0.062 → -0.062 → -0.062
- seed 3: +0.092 → +0.081 → +0.008 → +0.008 → +0.008 → +0.008 → -0.019 → -0.032 → -0.032 → -0.032 → -0.032 → -0.046 → -0.046 → -0.052 → -0.054 → -0.054
