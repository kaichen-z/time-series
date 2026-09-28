# i3 episode 3 observations

- Read the seed, accepted method, all existing notes, and visible traces. Starting fitness was 0.2484.
- In nine visible single-correction short upward events, the final corrected step had truth/base near 1, while earlier steps reflected the spike. The correction window was one step too long. Four input-only tasks met the same eligibility rule.
- Submission 5 trimmed that final step, changing one forecast point on 13 Train inputs. Visible fitness rose to 0.2807; hidden check passed; no runtime errors.
- Submission 6 then allowed the full extracted 2–5x magnitude on the remaining short-spike steps. Earlier event truth/base ratios were near the extracted multiplier. Visible fitness rose to 0.3216; hidden check passed; no runtime errors.
- Stopped after two submissions. Overlap cases remain too sparse to justify a composition rule. The accepted code is `shared/best_harness.py`.
- Consolidated prior notes in `shared/notes/consolidated.md` and added the reusable visible-only audit script in `shared/skills/`.
