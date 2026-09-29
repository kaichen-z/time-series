# r2, episode 4

- Started from accepted `0013_r2` (fitness 0.4954); reviewed shared notes and all Train correction windows.
- Solar radiation views with repeated exact nighttime zeros in history still had small positive base forecasts at those clock hours. A narrow H=24 hourly rule requires solar terminology in the first five documents, at least 48 hours of history, and at least six zero hours common to up to seven previous days. It sets only those hours to zero.
- The rule changed four Train views: visible tasks 119, 161, and 162 all had lower L1 and L2 errors; task 121 is unlabeled. Submission accepted with hidden check pass, no runtime errors, and visible fitness 0.4992. The accepted version is `shared/best_harness.py`.
- Audited Toto against the visible truths and the remaining correction windows. Toto was worse than the accepted forecast on most visible tasks, including the short surges after their document multipliers were applied. The remaining overlap claims and long, weak corrections had no consistent support, so I kept the accepted handling. One submission was used in this episode.
