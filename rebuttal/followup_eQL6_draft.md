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
interface. This limits the interpretation; it does not affect Theorem 1, which
remains valid for every stated $W\in O(d)$. In the reported experiments, $W=I$
is the fixed convention for axis-level diagnosis and regularization; $W_N$ is
reported separately in the ranking ablation as an orthogonal Procrustes solution
minimizing the feature-alignment residual. We will replace "identifiable from
the bound's decomposition" with "measured under the stated coordinate and
alignment convention."

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

**(3) Where the representation hypotheses enter.** You are right that neither
LRH nor PRH implies an orthogonal relation between pre- and post-trained
checkpoints. We use them only as high-level motivation for the structure PRISM
measures: LRH suggests that linearly accessible directions can carry meaningful
variables, while PRH suggests that relational geometry can be comparable across
learned representations. Together with the model's actual linear `lm_head`,
these observations motivate testing a linear bridge through the hybrid logits
$Z_PWH_T$. Intuitively, $W$ maps proxy features into target coordinates:
$Z_PW$ can be compared with $Z_T$, while $WH_T$ expresses the target readout in
proxy coordinates for comparison with $H_P$. This tests whether one orthogonal
map can jointly align the observed feature directions; it does not assume that
individual coordinates encode individual concepts. Neither hypothesis requires
$W\in O(d)$ or guarantees that the resulting alignment residual is small.

The orthogonal restriction instead comes from the geometry PRISM preserves and
decomposes. Among linear maps, $O(d)$ is exactly the group preserving the
standard Euclidean inner product, and hence norms, distances, and angles.
Allowing anisotropic scale or shear would let $W$ absorb the very scale and
shape changes PRISM aims to measure. Orthogonality therefore yields Proposition
1's exact scale--shape identity and the equivariance above. We will revise the
motivation to distinguish this empirical inspiration from the mathematical
design choice.

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
