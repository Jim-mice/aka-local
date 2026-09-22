# Final planner-benchmark stop rule

This is the final Planner benchmark. No v4, v5, or replacement benchmark may be designed merely to obtain a more favorable result.

After three fresh independent external runs of `public/case_final`:

- if at least 2 of 3 are full HIT, set `INTUITION_V3_STATUS = HELD_OUT_MECHANISM_REASONING`;
- otherwise retain `INTUITION_V3_STATUS = CONSTRAINT_AWARE_MECHANISM_REASONING`.

In either outcome, Planner research stops. The next activity is the real optimization loop: hypothesis selection, implementation attempt, L0/L1/L2 evidence, and rejection or promotion under existing policy. This rule is frozen before any v3 result exists and must not be changed based on result quality.
