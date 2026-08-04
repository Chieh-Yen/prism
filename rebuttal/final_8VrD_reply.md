貼法:回覆 8VrD 08-04 14:59 那則,Official Comment。先貼這則,AC 收尾留言最後貼。
不要求改分(他已明說維持 4)。62.6x 只用在對比生成式評測,對 NLL 明說沒有成本優勢。
Title 欄:What PRISM Adds at the Same Cost, and Where We Stop Claiming

Thank you for going back through the paper and the discussion before the window closes. It gives us the chance to state precisely what the framework adds at the same cost, and how far the evidence currently reaches.

**What PRISM is for.** It is not a replacement for generative evaluation or for an NLL screen. The measured cross-entropy gap comes from the same teacher-forced pass, and Theorem 1 bounds precisely that gap (with high rank correlation); what the pass additionally yields is a decomposition of it, into a scale term that is a norm ratio, a shape term that is an alignment residual, and a head term that is a readout mismatch.
Against generation, the saving is real: 8.9 s against 556.7 s of greedy decoding on identical prompts, 62.6x.
Because the shape term is differentiable, a penalty as regularizer on the very quantity diagnosed, which leads EWC, L2-SP, replay, and layer-freezing at matched plasticity.
That ordering is also stable at small reference size: 8 sequences rank the variants as 512 do, r_s 0.932 at both over three fresh seeds.

**Joint updates, and why full SFT is future work.** Two conditions must hold together for the ambiguity to bite: (1) both sides change, and (2) the change is a compensating PAIRED $(Z_P,H_P)\mapsto(Z_PA,A^{-1}H_P)$.
No evaluated procedure applies such a pair, and the paper claims nothing about them.
The GGUF configurations show the two conditions coming apart: they quantize both the backbone and `lm_head`, tensor by tensor in the inherited coordinates, and still rank with the rest of the family at r_s 0.943.
The same sensitivity is shared by any shape-based measure, linear CKA included, which is invariant to orthogonal maps and isotropic scaling but not to general $GL(d)$, too.
In practice the final normalization layer and weight decay keep the feature scale controlled, and gradient training has no pressure to realise an exact inverse pair, but neither is a guarantee, so the discussion belongs at the level of a theoretical limit.
It therefore moves from future work into the main text and the limitations, with attribution under full SFT and RLHF outside the validated scope until a canonicalization criterion exists.

**On task data.** For an evaluation, having validation text is a general assumption rather than a special one: if a variant can be assessed at all, that text exists, and PRISM needs only a teacher-forced pass over it rather than a generative run. What we do not claim is generic-text use for the diagnostic: that is unpaired use and is stated as out of scope. However, one step further back: a generic WikiText reference of 19 sequences still reduced forgetting (0.773 against 0.724 to 0.736 with the task's own reference).

**On the workflow.** What we do show is the loop closing in two concrete cases. When the shape term is what moved, it is also what can be trained against, and that penalty is the regularizer we report. When the head term dominates, the diagnosis points at the readout rather than the backbone, and the indicated fix (keep head, regularizing the norm, ...) was potential to reduce at a 20x lower bound. Full SFT, RLHF, and broader generation settings remain open.

Thank you again for the care in this final reading.

