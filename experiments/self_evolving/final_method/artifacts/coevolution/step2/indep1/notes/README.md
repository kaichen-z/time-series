# Consolidated research notes

Current accepted method: `shared/best_config.json` (seed visible fitness 0.18894).

- [Episode 1](i1_ep1.md): broader validator acceptance, lower repair margin, and linear fill failed to improve the full criterion. Linear fill passed hidden but lost visible fitness.
- [Episode 2](i1_ep2.md): 2% STL-ETS blend improved visible folds but failed hidden.
- [Episode 3](i1_ep3.md): stronger correction, 3% combined TimesFM seasonal, and lower shrink all failed hidden. Two submissions were intentionally left unused.

The repeated failure mode is a small gain on both visible folds without hidden transfer. Preserve the seed until a change has a stronger mechanism and evidence beyond local tuning. The visible trace numerical probe is in `shared/skills/`.
