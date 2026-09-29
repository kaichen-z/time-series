# r1, episode 3

- Reassessed after the previous episode produced no accepted improvement. Read
  all six prior episode notes, the latest shared best, allowed visible traces,
  and all 80 Train views. The promising remaining work had shifted from broad
  routing to very specific document or series mechanisms.
- Audited the current accepted method against base-only and solo-correction
  gains. It improves every visible task it changes. The only unused positive
  solo correction above 0.01 is task_97's late mild drop (+0.026); mixed
  maintenance and return-to-service documents make this too weak and ambiguous
  to warrant a submitted rule.
- During the audit, r2's independent Toto-level surge calibration was accepted
  (fitness 0.49537, hidden pass). Copied that latest shared best to
  `ws_r1/my_harness.py`; verified H finite outputs on all 80 Train views.
- No submission made this episode. A candidate based on task_97 alone would
  repeat the failed pattern of optimizing a single visible case without a
  robust mechanism.
- Added `shared/skills/audit_corrections.py` and its README, plus a consolidated
  notes file. The consolidated file records accepted mechanisms and approaches
  that never worked or lack support.
