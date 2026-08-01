Thank you for the precise counterexample. It showed that our previous response
conflated choosing $W$ for a fixed $(Z_P,H_P)$ with replacing that pair by the
functionally equivalent $(Z_PA,A^{-1}H_P)$, $A\in GL(d)$; we apologize for the
confusion and excess detail.

**(1) Correction to our previous response.** Our previous function-level
identifiability rationale was too strong. Such a paired $GL(d)$
reparameterization preserves the proxy logits but can change the numerical
split across PRISM's three axes. Thus, the attribution is defined relative to
specified representation coordinates and a stated orthogonal alignment $W$,
rather than being determined solely by the input--output function. This narrows
the interpretation of the decomposition, while Theorem 1 provides a valid
certificate for each specified $W\in O(d)$, including $W=I$ and the reported
$W_N$ specialization. We will make this scope explicit and replace the paper's
informal phrase "identifiable
from the bound's decomposition" with "isolated by the bound's
decomposition."

**(2) What does hold under $O(d)$.** Let $Q\in O(d)$ and reparameterize the
proxy as $\widetilde{Z}_P=Z_PQ$ and $\widetilde{H}_P=Q^\top H_P$. For every
$W\in O(d)$, transport the alignment as $\widetilde{W}=Q^\top W$. Then
$\widetilde{Z}_P\widetilde{W}=Z_PW$, so the feature residual, $\rho_P$, and
$\delta$ are unchanged. Moreover,
$$
\widetilde{\Sigma}_P = Q^\top \Sigma_P Q,\qquad
\widetilde{\Sigma}_P^{1/2}
\left(\widetilde{W}H_T-\widetilde{H}_P\right)
=
Q^\top\Sigma_P^{1/2}\left(WH_T-H_P\right).
$$
Thus $\widetilde{\gamma}(\widetilde{W})=\gamma(W)$, and $W\mapsto Q^\top W$
bijects the certificate families, leaving $\Omega_N$ and $\delta_N$ invariant.
In plain terms, after transporting $W$, an orthogonal reparameterization only
relabels coordinates; the certificate and feature/head split are unchanged.
Geometrically, $\delta(W)$ is the residual after mapping proxy features to target
coordinates, whereas $\gamma(W)$ compares $H_P$ with the target head transported
by $W$ on directions supported by $Z_P$; reducing one need not reduce the other.
This equivariance is guaranteed for $Q\in O(d)$ when the alignment is
transported as $\widetilde{W}=Q^\top W$; it need not hold for a general
$A\in GL(d)$ or when $W$ is held fixed. We will state these conditions
explicitly.

**(3) The representation coordinates used in this paper.** Our experiments use
the native coordinates inherited from a shared base checkpoint.
PTQ rounding, GGUF, BnB, and frozen-head LoRA perturb that checkpoint without
inserting a compensating $(A,A^{-1})$ pair at the backbone--head interface;
$Z_P$ is the post-final-norm state measured in those native coordinates. Within
this coordinate system, we use $W=I$ for axis-level diagnosis and the
regularizer, and separately report the feature-optimal $W_N$ specialization for
ranking, with the alignment stated in each case. The corresponding interventions
(e.g., retaining the base `lm_head`) are defined in the same coordinates.
Comparisons after an independent change of representation coordinates,
including deliberately $GL(d)$-reparameterized artifacts, are outside the
validated scope and will be stated as a limitation.

**(4) Where the representation hypotheses enter, more precisely.** You are
right that our previous response moved too quickly from shared structure to an
orthogonal map. LRH suggests that semantic variables can be linearly accessible;
for checkpoints from the same base, this motivates testing whether such
directions correspond. PRH suggests that learned representations may share
relational structure; here it motivates testing whether same-lineage checkpoints
preserve comparable geometry. Together, they motivate testing $W$ as a bridge
via the hybrid logits $Z_PWH_T$, but not restricting it to $O(d)$.

That restriction comes from the geometry PRISM aims to preserve. Its axes use
Euclidean norms, distances, and angles, which orthogonal maps preserve; a
general linear fit can absorb scale and shear into $W$, so the residual loses
the same axis interpretation. This yields Proposition 1's exact scale--shape
identity and the $O(d)$-equivariance above.

By "cheap," we meant empirical feature-alignment cost, not proof of
orthogonality or raw-bound tightness. In our top-$r$ Llama ablation,
unrestricted alignment reduces the Q2_K residual by 18.3% on
average over MMLU and SQuAD relative to scaled-orthogonal alignment. This is
limited evidence of adequacy in the measured subspaces; it neither resolves the
$GL(d)$ ambiguity nor bears on theorem validity.

Finally, "the workflow the tool serves" meant only same-lineage screening in
inherited coordinates, not a mathematical premise; we will remove that phrase.

We would value knowing whether the explicit $GL(d)$ limitation, the
$O(d)$-equivariance statement, and the native-coordinate scope resolve the
ambiguity you identified.
