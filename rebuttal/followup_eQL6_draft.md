Thank you for the precise follow-up. Our previous response conflated
reparameterizing the proxy representation/head pair with choosing an alignment
for a fixed pair; we apologize for the resulting confusion and unnecessary
exposition.

**(1) Coordinate dependence of the diagnosis.** Your counterexample is
correct: PRISM's attribution is not invariant over the full $GL(d)$ equivalence
class. Functionally equivalent parameterizations can receive different axis
diagnoses, so there is no coordinate-free "true" split. Our evaluated setting
fixes the convention by construction: target and proxy are stored variants of
the same checkpoint, compared at the same final hidden-state interface
immediately before the prediction head, and none of the studied PTQ or
frozen-head LoRA procedures inserts a compensating basis change at that
interface. This limits the interpretation; it does not
affect Theorem 1, which remains valid for every stated $W\in O(d)$, including
$W=I$ and the reported $W_N$. We will replace "identifiable from the bound's
decomposition" with "measured under the stated coordinate and alignment
convention."

**(2) What remains invariant under $O(d)$.** For a general $A$, transporting the
alignment gives $\widetilde W=A^{-1}W$, which generally falls outside $O(d)$.
If $A=Q\in O(d)$, set $\widetilde Z_P=Z_PQ$,
$\widetilde H_P=Q^\top H_P$, and $\widetilde W=Q^\top W$. Then
$$
\widetilde Z_P\widetilde W=Z_PW,\qquad
\widetilde\Sigma_P^{1/2}
(\widetilde W H_T-\widetilde H_P)
=Q^\top\Sigma_P^{1/2}(WH_T-H_P).
$$
Orthogonal invariance of the Frobenius norm, together with
$\widetilde Z_P\widetilde W=Z_PW$, leaves the scale, shape, and head terms
unchanged. Thus, the attribution is invariant to orthogonal coordinate changes
when $W$ is transported with them, but not in general to $GL(d)$ changes or when
$W$ is held fixed. We will add this boundary as a formal remark.

**(3) Scope of the reported diagnosis.** Within this fixed convention, we use
$W=I$ for axis-level diagnosis and the regularizer, and report the
feature-optimal $W_N$ specialization for ranking, with the alignment stated in
each case. The reported attributions are therefore well-defined for these fixed
artifacts. The theorem still applies to a reparameterized artifact for each
specified orthogonal $W$; what we do not claim is coordinate-free attribution
across deliberately $GL(d)$-reparameterized artifacts. We will state this
limitation explicitly.

**(4) Where the representation hypotheses enter.** You are also right that
neither LRH nor PRH by itself implies an orthogonal relation between pre- and
post-trained checkpoints. LRH motivates testing whether linearly accessible
semantic directions correspond across same-base checkpoints; PRH motivates
testing whether their relational geometry remains comparable. Together, they
motivate testing a linear bridge through the hybrid logits $Z_PWH_T$, but not
the restriction $W\in O(d)$.

That restriction instead preserves the geometry PRISM decomposes: orthogonal
maps preserve Euclidean norms, distances, and angles, whereas a general linear
fit can absorb scale and shear into $W$ and thereby change the axes'
interpretation. Orthogonality also yields Proposition 1's exact scale--shape
identity and the equivariance above.

By "cheap," we meant only the empirical cost of excluding anisotropic scale and
shear after factoring out one global scale, not proof of orthogonality or
raw-bound tightness. In a separate top-$r$ Llama test, an unconstrained linear
least-squares map reduces the Q2_K residual by 18.3% on average over MMLU and
SQuAD relative to rotation plus one global scale. This is limited evidence of
adequacy in the measured subspaces; it neither resolves the $GL(d)$ ambiguity
nor bears on theorem validity.

Finally, "tool" meant PRISM, and "workflow" meant screening same-lineage
variants in their inherited coordinates; neither is a mathematical premise. We
will remove the original phrase.

Your counterexample helped us sharpen this theoretical boundary. We would value
knowing whether the explicit $GL(d)$ limitation, the exact $O(d)$-equivariance
boundary, and the native-coordinate scope resolve the ambiguity you identified.
