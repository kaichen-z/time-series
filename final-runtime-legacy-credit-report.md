# Legacy mutation-credit regression investigation

The exact credit test passed unchanged at `d5856a1` (1 passed in 49.80s).
Comparing the clean-worktree failure (`pytest-1029`) with this passing run
(`pytest-1030`) showed identical proposals and genome IDs, but the failed run's
first child stopped during materialization after nine task executions and one
subprocess. Its failed receipt had zero FakeClock wall time. Only two children
reached the first rung; both were feasible. The failed run consumed 1737 tasks,
whereas the passing run consumed the intended 2472-task boundary.

The evidence-identity guard cannot cause this early failure: it runs after
fitting, and legacy materialization does not enter the evidence-aware branch.
Its freeze check also explicitly returns when evidence is absent. No production
semantics were changed.

The credit-only test inherited a 10 ms real isolated-worker IPC timeout while
its resource accounting used FakeClock. Scheduler delay can therefore discard a
valid child before the behavior under test. Increased only this test's timeout
to one second; retained FakeClock, the 2472-task ceiling, and exact assertions
for three attempts, three feasible children, two promotions and zero insertions.
Dedicated timeout tests keep their existing configuration.

Validation: the focused gate reported **6 passed in 140.31s** before being
interrupted at the parent's request. Completed checks were the exact credit
regression, all three terminal-credit bracket cases, the explicit partial-rung
timeout test, and the same-ID changed-policy evidence rejection. The remaining
specialist-failure/integration and projection checks did not finish in this run;
no full-gate success is claimed. The parent will run a final smoke on the commit.
`git diff --check` passed. Protected P3 files were not edited or staged.
