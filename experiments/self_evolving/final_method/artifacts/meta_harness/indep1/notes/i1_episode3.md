# i1, episode 3 observations

Reviewed the episode 1 and 2 notes, the accepted harness, and only the allowed Train views and visible traces. Four submissions (0011–0014) were made; all were accepted and passed the hidden check. One of the episode's five submission slots was left unused because the remaining ideas depended on one or two visible tasks without a reliable broader mechanism.

Accepted attempt 0014 is the new shared best. Visible fitness rose from 0.247024 to 0.282379. Relative to attempt 0006, exactly five visible tasks changed: task_121 +1.0505, task_183 +0.3147, task_179 +0.2264, task_112 +0.1626, and task_159 +0.1366. These gains are concentrated, so hidden-check pass alone does not establish broad calibration.

- **Maintenance fill (0011):** In hour-seasonal traffic series, historical scheduled maintenance produced near-zero readings at the same hours every day. When the first document concerns maintenance and an upward correction targets those historically zero hours, a multiplier applied to a near-zero base cannot restore the operational level. Fill only those hours from the forecast immediately after the maintenance window. This improved task_179 and task_183; other steps retain the existing forecast.
- **Solar collapse fallback (0012):** If an hourly 24-step solar/sky document accompanies a base whose daytime peak is under 5% of the previous day's peak, blend 70% of the previous day's cycle into the base. This changed task_121 only. Its gain was large, but evidence is one visible task.
- **Mild solar persistence (0013):** For a less extreme solar base under half the previous day's peak, blend 25% of yesterday's cycle. This changed task_112 only and improved it. More general use on solar tasks remains untested.
- **Post-holiday rebound (0014):** The holiday rule rejected all upward corrections. A capped 20% upward adjustment starting on day two is consistent with a post-holiday rebound and improved task_159. Same-day upward corrections remain rejected.

The final function is `shared/best_harness.py`, with a private copy at `ws_i1/my_harness.py`. It returned the required number of finite values on all 80 provided views. `shared/skills/visible_delta.py` compares allowed visible attempt results and counts affected tasks.
