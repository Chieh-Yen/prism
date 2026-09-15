貼法:**forum 最外層新開一則 Official Comment**(頂端的 Official Comment 按鈕,不是任何 note 的 reply),
readers 用預設的最廣一組(PC / SAC / AC / Reviewers / Submitted / Authors)。最後貼,讓它是 forum 最後一則。
**不提任何 reviewer 的分數或升分意向**(G3T9 紀錄分數仍是 3;有利的話已由 reviewer 自己說出,代述會變成遊說)。
Title 欄:Closing Note for the Area Chair: What the Final Exchanges Settled

As the discussion window closes, a short note on what the final exchanges settled. Each comment posted after the summaries has been answered in its own thread, and the evidence for the meta-review's five requests is indexed in the Point A to Point E comments, so we do not repeat it here.

**Where the reviewers stand.** Reviewer G3T9 now describes the initial concerns as "substantially addressed", and finds the practical value "more convincing" given the calibration, the added baselines, and the free-running evaluation. Reviewer 8VrD calls the theoretical construction "clean and technically sound" and several of the clarifications and additional experiments "helpful". We replied in the same thread to the concerns about practical significance that remain. Reviewer eQL6 wrote that many concerns had been addressed, and the point pressed there is the sharpening we value most. We are grateful for these readings.

**The one open point: the coordinate convention.** Reviewer eQL6's counterexample showed that PRISM's attribution is coordinate-relative. Applying the paired map $(Z_P,H_P)\mapsto(Z_PA,A^{-1}H_P)$ before the diagnosis leaves the model's outputs unchanged while redistributing the scale, shape, and head terms.

- The ambiguity requires that the two changes be **exact inverses** of each other. Changing both the backbone and the head is not sufficient, and no evaluated procedure applies such a pair.
- Both reviewers who examined this agree on its reach. Reviewer eQL6 noted the issue is "likely not an issue in the post-training variants studied in the paper"; Reviewer 8VrD judged the convention "reasonable for the PTQ and frozen-head LoRA settings studied here" and not a defect in the bound.
- Where the backbone and head are jointly updated, both regard the root-cause interpretation as unsettled, and that is where we now make no claim.
- No reported value changes, Theorem 1 remains valid for every specified $W\in O(d)$, and no component must remain frozen.

**What the revision will say.** It will state the convention in the main text, add the exact $O(d)$-equivariance remark, place attribution after joint backbone-head optimization outside the validated scope, replace the earlier "identifiable" wording, and leave canonicalization for that regime as future work.

The scope is therefore stated more precisely than at submission, in the direction the meta-review invited, and each narrowing rests on an experiment or an analysis posted in this discussion rather than on wording alone. We thank the reviewers and the Area Chair for a discussion that sharpened both the evidence and the statement of scope.
