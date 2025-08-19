# First-Order KKT and Variational Setup

## 1. KKT Conditions
Assume standard regularity (LICQ) at x*. Then any local minimizer x* admits multipliers λ*, ν* such that:

1. **Stationarity**:  
   ∇ₓ L_stab(x*, λ*, ν*) = 0

2. **Primal feasibility**:  
   g_i(x*) ≤ 0, h_j(x*) = 0, x* ∈ K

3. **Dual feasibility & complementary slackness**:  
   λ_i* ≥ 0, λ_i* g_i(x*) = 0 ∀i

## 2. Projected Gradient Descent
x_{t+1} = Π_K(x_t - η ∇J(x_t)), η ∈ (0, 2/L)

Accumulation points satisfy KKT conditions under smoothness, coercivity, and well-defined projections.

## 3. Lawful Closure
- Probabilistic closure: lim_{t→∞} ∇H(x_t) = 0, lim ∇Φ(x_t) = 0  
- Deterministic closure: ∇L_stab(x*, λ*, ν*) = 0, x* ∈ K  
- Enforcement closure: append-only audit mapping π(x*, t*) ensures traceability
