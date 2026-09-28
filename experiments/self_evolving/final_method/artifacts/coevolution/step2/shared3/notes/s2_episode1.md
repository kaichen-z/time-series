# s2 episode 1: decision parameters

Starting config: visible fitness 0.18894. Repair trace has 13 tasks with variants, and current phase-median back-test margin is 0.3. The margin is evidently relative to raw error: at 0.1, task_112 (0.338 absolute reduction on 2.582 raw) changes dramatically; it was unchanged at 0.3.

## Submission 1: margin 0.1

Visible fitness 0.1787, hidden check failed. Task_179 improved gain from 0.005 to 0.488, but task_112 worsened from -0.242 to -0.729. Broadly admitting modest back-test gains is unsafe: back-test improvement can fail to transfer to the forecast horizon.

## Submission 2: margin 0.25

Hidden check passed but visible fitness fell to 0.1878. Only task_47 changed, with gain lower by 0.055. Keep 0.3 margin. The current stricter rule protects against moderate back-test gains that do not transfer.

Correction trace inspection: several accepted short upward events (tasks 128/138, 131/141, 132) are still substantially underforecast at the event hours. This provides a mechanistic reason to test slightly higher correction strength.

## Submission 3: strength 1.2

Visible fitness 0.18835; hidden check failed. Only tasks 121 and 158 materially changed, both for the worse (158 gain down 0.042). Large upward event corrections stayed identical, apparently because their applied multiplier is already bounded by the evaluator. Increased strength therefore targets the wrong cases.

## Submission 4: strength 0.8

Visible fitness 0.1888, hidden check failed. Lowering strength also does not improve the main method. Keep strength 1.0.

Final candidate: linear fill. Task_193 shows phase-median repaired back-test error 0.278 versus linear 0.001, with both much better than raw history. This is a large, mechanism-based contrast worth a single evaluation.

## Submission 5: linear repair fill

Hidden check passed, but visible fitness dropped to 0.1826. Task_193 gain fell from 1.9472 to 1.4933 despite linear's much lower history back-test error. In this case, excellent back-test fit did not predict future gain. Phase median remains better.

## Episode conclusion

Five submissions used. None beat the seed (visible fitness 0.18894 with hidden pass), so no change to shared best. Keep decision defaults: phase_median, margin 0.3, strength 1.0, trust_gate 0.0. The repair margin is useful protection against transferring modest back-test improvements; even a dramatic back-test improvement under another fill can hurt the forecast. Both higher and lower correction strength failed the hidden check, while the large upward-event corrections appear bounded and did not respond to strength changes. Do not revisit these local variations without a stronger mechanism.
