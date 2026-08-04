貼法:回覆 8VrD 08-04 14:59 那則,Official Comment。先貼這則,AC 收尾留言最後貼。
不要求改分(他已明說維持 4);不要引 62.6x(NLL 篩選成本相同,會被反駁)。
Title 欄:Three Clarifications Before the Window Closes

Thank you for reading the full record again before the window closes. We accept the scope you describe: task-relevant reference data, calibration that is leave-one-out within a setting, and closely related variants of a shared checkpoint. Four clarifications follow, one of which names a gap we should own.

**Joint updates are not the trigger.** The ambiguity requires the compensating pair $(Z_P,H_P)\mapsto(Z_PA,A^{-1}H_P)$; modifying both the backbone and the head is not sufficient for it. The evaluated GGUF configurations are the case in point: they quantize both the backbone and `lm_head`, tensor by tensor in the inherited coordinates, and the attribution is unaffected. What we do not claim is the jointly gradient-optimized case, where the inherited convention is not guaranteed; the revision places that regime outside the validated scope rather than asserting it either way.

**Likelihood screening.** The NLL gap is part of the same measurement: PRISM's teacher-forced pass yields the token-level loss, and Theorem 1 bounds that very gap. The revision will report the measured gap beside the bound on the same runs, at no additional cost. What the same pass buys beyond the scalar is the decomposition: which axis moved, with single-axis interventions moving only their own term (at least 2.6e5x selectivity, bound holds 26/26); a differentiable penalty on the quantity diagnosed, which leads EWC, L2-SP, replay and layer-freezing at matched plasticity; and an upper bound rather than an estimate, which we agree matters most where the gap cannot be measured directly. We do not claim a ranking advantage over the screen, and the revision will not assert one.

**General text.** We agree this is the open direction, and we treat it as out-of-distribution generalization rather than a claim of this paper: for the diagnostic, a generic-text reference is unpaired use and is stated as out of scope. The one datum we have is on the regularizer and is single-seed: a generic WikiText reference of 19 sequences still reduces forgetting, to 0.773 against 0.724 to 0.736 with the task's own reference. We report it as an observation, not a result.

**On the workflow.** We accept that broad reliability is not established. What is established: the head case closes the loop end to end, with 75.77 of 76.95 attributed to the head term and the targeted fix verified at a 20x lower bound, and the rollout-conditioned test reaches +0.947 against +0.958 teacher-forced. Broader generation settings remain open, and the revision scopes the workflow claim to the settings in which it was validated.

Thank you again for the care in this final reading.
