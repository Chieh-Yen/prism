Thank you for the precise counterexample. You are correct: it exposes that our
previous response conflated two different operations---choosing an alignment
$W$ for fixed artifacts, and first reparameterizing the proxy artifact by
$(Z_P,H_P)\mapsto(Z_PA,A^{-1}H_P)$. We apologize for the unclear and overlong
explanation.

**(1) What the $GL(d)$ counterexample establishes.** For a general
$A\in GL(d)$,
\[
    (Z_PA)(A^{-1}H_P)=Z_PH_P,
\]
so the logits and risk gap are unchanged, whereas $\delta$ and $\gamma$ generally
change. Each parameterization still gives a valid, potentially different upper
bound on the same $|\Delta\mathcal R|$, but the three-axis attribution is not an
invariant of the input--output function over the full $GL(d)$ equivalence class.
Thus, without a specified gauge, there is no unique function-level ``true
failure.'' We will state this limitation explicitly and replace our formal
identifiability claim with a coordinate-relative one.

**(2) What does hold under $O(d)$.** Let $Q\in O(d)$ and reparameterize the
proxy as $\widetilde Z_P=Z_PQ$ and $\widetilde H_P=Q^\top H_P$. For every
$W\in O(d)$, transport the alignment as $\widetilde W=Q^\top W$. Then
$\widetilde Z_P\widetilde W=Z_PW$, so the feature residual, $\rho_P$, and
$\delta$ are unchanged. Moreover,
\[
\widetilde\Sigma_P=Q^\top\Sigma_PQ,
\qquad
\widetilde\Sigma_P^{1/2}
(\widetilde W H_T-\widetilde H_P)
=Q^\top\Sigma_P^{1/2}(WH_T-H_P),
\]
which gives $\widetilde\gamma(\widetilde W)=\gamma(W)$. Hence
$W\mapsto Q^\top W$ is a bijection between the two certificate families, and
the feature-optimal values $\Omega_N$ and $\delta_N$ are invariant. This does
not extend to a general $A$, and it requires transporting $W$: if the proxy is
rotated while $W=I$ is kept fixed, the reported split can change. We will add
this precise equivariance statement as a formal remark.

**(3) The operational gauge in this paper.** Our experiments explicitly use
the native coordinates inherited from a shared base checkpoint. PTQ rounding,
GGUF, BnB, and frozen-head LoRA perturb that checkpoint without inserting a
compensating $(A,A^{-1})$ pair at the backbone--head interface; $Z_P$ is the
post-final-norm state measured in those native coordinates. We therefore read
$W=I$ as a fixed-coordinate diagnosis for the same-lineage variants studied,
and the corresponding interventions (e.g., retaining the base `lm_head`) are
defined in that same coordinate system. Comparisons across independently chosen
gauges, or deliberately $GL(d)$-reparameterized artifacts, are outside the
validated scope and will be stated as a limitation.

**(4) Why we use $O(d)$, and where the representation hypotheses enter.** You
are right that neither hypothesis implies an orthogonal map between pre- and
post-trained checkpoints. LRH concerns linear concept representations and does
not privilege the Euclidean inner product; PRH concerns convergence of
relational structure. We will withdraw the implication and cite them only as
context. The actual reasons for choosing $O(d)$ are two provable properties:
orthogonal maps preserve the Euclidean geometry used by the axes and yield
Proposition 1's exact scale--shape identity, and the certificate family has the
$O(d)$-equivariance shown above. A general linear alignment carries scale and
shear itself, so it can absorb those components into the fitted map and no
longer supports the same axis interpretation. As a limited empirical check, not
a resolution of the $GL(d)$ ambiguity, our top-$r$ Llama ablation compares an
unrestricted linear map with a scaled-orthogonal map, factoring out the global
scale modeled separately by PRISM. The unrestricted map reduces the Q2_K
residual by 18.3% on average over MMLU and SQuAD; this supports adequacy in the
measured subspaces only, not theorem validity. Theorem 1 remains valid for every
$W\in O(d)$.

Finally, by the previous phrase ``the workflow the tool serves,'' we meant only
PRISM's intended use: screening same-lineage post-trained variants and
attributing drift in their inherited coordinates. It was operational context,
not a mathematical premise, and we will remove that phrase.

We would value knowing whether the explicit $GL(d)$ limitation, the
$O(d)$-equivariance statement, and the native-coordinate scope resolve the
ambiguity you identified.
