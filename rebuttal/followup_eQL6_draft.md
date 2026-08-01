Thank you for the precise counterexample. It exposed that our previous response
conflated choosing $W$ for fixed artifacts with reparameterizing the artifacts
themselves; we apologize for the resulting confusion and unnecessary exposition.

**(1) Correction to our previous response.** Our previous identifiability
rationale was too strong. Your counterexample shows that a general $GL(d)$
reparameterization can preserve the proxy logits while changing
the numerical split across PRISM's three axes. The precise conclusion is that
the attribution is defined relative to specified representation coordinates
and a stated orthogonal alignment $W$, rather than being determined solely by
the input--output function. This narrows the interpretation of the
decomposition, while Theorem 1 provides a valid certificate for every chosen
$W\in O(d)$, including $W=I$ and the reported $W_N$ specialization. We will
make this scope explicit and replace the paper's informal phrase "identifiable
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
Therefore, $\widetilde{\gamma}(\widetilde{W})=\gamma(W)$. Hence
$W\mapsto Q^\top W$ is a bijection between the two certificate families, and
the feature-optimal values $\Omega_N$ and $\delta_N$ are invariant.
Geometrically, $\delta(W)$ is the residual after mapping proxy features to target
coordinates, whereas $\gamma(W)$ compares $H_P$ with the target head transported
by $W$ on directions supported by $Z_P$; reducing one need not reduce the other.
This equivariance does not extend to a general $A$, and it requires transporting
$W$: if the proxy is rotated while $W=I$ is kept fixed, the reported split can
change. We will add this precise equivariance statement as a formal remark.

**(3) The representation coordinates used in this paper.** Our experiments
explicitly use the native coordinates inherited from a shared base checkpoint.
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
orthogonal map. LRH suggests linearly accessible concept directions; PRH
suggests that learned representations may share relational structure. Together,
they motivate testing an explicit map $W$ as a bridge between the spaces,
operationalized by the hybrid logits $Z_PWH_T$; they motivate the bridge, not
its restriction to $O(d)$.

That restriction comes from the geometry PRISM aims to preserve. Its axes use
Euclidean norms, distances, and angles, which orthogonal maps preserve; a
general linear fit can absorb scale and shear into $W$, so the residual loses
the same axis interpretation. This yields Proposition 1's exact scale--shape
identity and the $O(d)$-equivariance above.

By "cheap," we meant the empirical cost at the feature-alignment level, not
that LRH/PRH prove orthogonality or that the raw bound is tight. In our top-$r$
Llama ablation, unrestricted alignment reduces the Q2_K residual by 18.3% on
average over MMLU and SQuAD relative to scaled-orthogonal alignment. This is
limited evidence of adequacy in the measured subspaces; it neither resolves the
$GL(d)$ ambiguity nor bears on theorem validity.

Finally, by the previous phrase "the workflow the tool serves," we meant only
PRISM's intended use: screening same-lineage post-trained variants and
attributing drift in their inherited coordinates. It was operational context,
not a mathematical premise, and we will remove that phrase.

We would value knowing whether the explicit $GL(d)$ limitation, the
$O(d)$-equivariance statement, and the native-coordinate scope resolve the
ambiguity you identified.
