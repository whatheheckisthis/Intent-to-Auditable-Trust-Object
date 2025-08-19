# Stabilized Lagrangian and Dynamics

L_stab(x, λ, ν) = f(x) + Σ λ_i g_i(x) + Σ ν_j h_j(x) - τ H(x) + Φ(x)

PGD step:
x_{t+1} = Π_K(x_t - η ∇J(x_t))

Assumptions:
- f,H,Φ,g,h ∈ C¹
- J coercive and L-smooth
- K closed, projection Π_K well-defined
- Constraint qualification (LICQ)
- Cryptographic audit mapping π ensures tamper-evident trace
