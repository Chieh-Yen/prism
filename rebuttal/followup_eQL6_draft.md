Thank you for the precise follow-up. Our previous response conflated
reparameterizing the proxy representation/head pair with choosing an alignment
for a fixed pair; we apologize for the resulting confusion and unnecessary
exposition.

**(1) Coordinate dependence of the diagnosis.** Your counterexample is
correct: PRISM's attribution is not invariant over the full $GL(d)$ equivalence
class. Functionally equivalent parameterizations can receive different axis
diagnoses, so PRISM does not define a coordinate-free "true" split over that
equivalence class. Our evaluated setting fixes the convention by construction:
target and proxy are stored variants of the same checkpoint, compared at the
same final hidden-state interface
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
\widetilde\Sigma_P=Q^\top\Sigma_PQ.
$$
Since right-multiplication by $Q$ preserves the Frobenius norm,
$\widetilde\rho_P=\rho_P$. Together,
$\widetilde Z_P\widetilde W=Z_PW$ and
$\|\widetilde Z_P\|_F=\|Z_P\|_F$ give
$\widetilde\Omega_{\widetilde W}=\Omega_W$. Moreover,
$\widetilde W=Q^\top W\in O(d)$, so the transported alignment remains in
Theorem 1's admissible family.
For the principal PSD square root,
$\widetilde\Sigma_P^{1/2}=Q^\top\Sigma_P^{1/2}Q$, and hence
$$
\widetilde\Sigma_P^{1/2}
(\widetilde W H_T-\widetilde H_P)
=Q^\top\Sigma_P^{1/2}(WH_T-H_P).
$$
These equalities, together with orthogonal invariance of the Frobenius norm,
leave the scale, shape, and head terms unchanged. Thus, the attribution is
invariant to orthogonal coordinate changes
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
proxy coordinates for comparison with $H_P$. The construction uses one shared
$W$ across all observed feature directions; it does not equate coordinate axes
with individual concepts. Neither hypothesis guarantees that the resulting
alignment residual is small.

We restrict $W$ to $O(d)$ to preserve the Euclidean geometry PRISM decomposes.
The elements of $O(d)$ are precisely the linear maps that preserve the standard
Euclidean inner product, and therefore norms, distances, and angles. A general
linear map can absorb scaling and shear into $W$, so the residual no longer has
the same scale–shape interpretation. This preservation yields Proposition 1's
exact scale–shape identity and the equivariance above. We will revise Secs. 2–3,
Appendix C, and the limitation statement to distinguish this empirical inspiration
from the mathematical design choice.

Finally, we agree that the quoted sentence was opaque. By "tool" we meant
PRISM; by "workflow" we meant screening same-lineage variants in their inherited
coordinates with one teacher-forced pass per variant and reporting axis-level
drift. That use case explains our evaluation setting, but it is not the
mathematical justification for restricting $W$ to $O(d)$; we withdraw that
sentence.

Your counterexample helped us sharpen this theoretical boundary. We would value
knowing whether the explicit $GL(d)$ limitation, the exact $O(d)$-equivariance
boundary, and the native-coordinate scope resolve the ambiguity you identified.
