# s0 episode 1: document-aware grid surge calibration

Read `shared/harness/0000_seed.py`, all 54 visible traces, and 80 unlabeled Train views. The notes and skills directories were initially empty. Spent exactly five submissions.

## Observation

The visible traces contain a group of brief, positive electricity-demand spikes with corrections between 1.5x and 5x. The seed caps their forecast lift at 50%. For 5x events (tasks 128 and 138), visible truth is around 5x the base forecast, so the cap materially undercorrects. A 1.5x event (task 137) is rejected by the seed even though its direction helps. Many long, low-magnitude corrections are correctly rejected. The `corrections_solo` gain is incremental to `base_only_gain`, and sMAE+sRMSE uses mean absolute truth as scale (verified on visible single-correction tasks).

## Submissions

1. Document terms for grid/electricity demand; accept short positive events and raise their cap to 100%: visible fitness **0.2149**, hidden check **fail**.
2. Same with 75% cap: **0.2068**, hidden **fail**.
3. Original cap, only accept short grid events (changes task 137 alone among visible/Train views): **0.1972**, hidden **fail**.
4. Seed acceptance unchanged; 75% cap only if the historical 95th percentile matches the local base scale: **0.2038**, hidden **fail**.
5. Same but only documented 5x grid events: **0.1978**, hidden **fail**.

No candidate was promoted; `shared/best_harness.py` remains the seed. The visible grid-spike pattern is strong but does not pass the independent hidden check, even when narrowed drastically. Do not repeat this class of rule without a new mechanism explaining hidden failure. The current private `ws_s0/my_harness.py` is submission 5; `candidate_grid100.py` is submission 1.
