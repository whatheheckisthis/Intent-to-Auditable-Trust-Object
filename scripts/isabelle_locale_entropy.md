locale entropy_kkt_locale =
  fixes f :: "real^n ⇒ real"
    and H :: "real^n ⇒ real"
    and Phi :: "real^n ⇒ real"
    and g :: "real^n ⇒ real^m"
    and h :: "real^n ⇒ real^p"
    and K :: "real^n set"
    and tau :: real
  assumes C1: "∀x. differentiable (f x) (…) "
      and C2: "J x = f x - tau * H x + Phi x"
      and C3: "closed K"
      and C4: "coercive_on K J"
      and C5: "LICQ_holds …"
begin

definition Lagrangian where
  "Lagrangian x λ ν = f x + λ ⋅ g x + ν ⋅ h x - tau * H x + Phi x"

definition PGD_step where
  "PGD_step x η = project K (x - η * grad (J x))"

lemma descent_lemma: assumes "0<η" "η<2/L" shows "J(PGD_step x η) ≤ J x - c * ‖grad J x‖^2" sorry
lemma bounded_iterates: assumes "x0 ∈ K" shows "∀t. PGD_iterates t ∈ K ∧ bounded ..." sorry
lemma accumulation_stationary: assumes "x* is accumulation point" shows "∃λ ν. grad_x Lagrangian x* λ ν = 0 ∧ g x* ≤ 0 ∧ h x* = 0 ∧ x* ∈ K" sorry
lemma variational_bound: shows "⟨ψ_x* | H | ψ_x*⟩ ≥ E0" sorry

end
