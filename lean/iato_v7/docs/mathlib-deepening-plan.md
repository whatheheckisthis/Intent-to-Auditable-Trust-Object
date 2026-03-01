# Mathlib Deepening Plan for IATO-V7 (Lean 4 + mathlib4)

This plan expands mathlib usage for non-interference, domain isolation, FEAT_RME transitions, and compliance-oriented proof artifacts.

## 1) Key mathlib modules to import and use immediately

Below are 12 concrete modules that map directly to IATO-V7 concerns.

### 1. `Mathlib.Order.Lattice`
- **Why:** Core lattice abstractions (`SemilatticeSup`, `DistribLattice`, `CompleteLattice`) for confidentiality/integrity labels and dependency joins.
- **Where:** `IATO/V7/Basic.lean`, then reused by all files.
- **Snippet:**
```lean
import Mathlib.Order.Lattice

variable {L : Type*} [CompleteLattice L]

def canFlowTo (a b : L) : Prop := a ≤ b

theorem flow_refl (a : L) : canFlowTo a a := le_rfl
```

### 2. `Mathlib.Order.Hom.CompleteLattice`
- **Why:** Use `SupHom`/`InfHom`/`OrderHom` for monotone sanitizers, projection functions, and policy-preserving transformations.
- **Where:** `Worker.lean`, `Architecture.lean`.
- **Snippet:**
```lean
import Mathlib.Order.Hom.CompleteLattice

variable {L : Type*} [CompleteLattice L]

structure Sanitizer where
  f : L →o L
```

### 3. `Mathlib.Order.GaloisConnection`
- **Why:** Best fit for compliance abstraction/refinement links (operational states ↔ audit controls) with adjointness guarantees.
- **Where:** New compliance mapping module; referenced in `Architecture.lean`.
- **Snippet:**
```lean
import Mathlib.Order.GaloisConnection

variable {A B : Type*} [Preorder A] [Preorder B]
variable (α : A → B) (γ : B → A)

#check GaloisConnection α γ
```

### 4. `Mathlib.Order.FixedPoints`
- **Why:** Define least fixed points for iterative hardening/migration closure (scan until no conflicts remain).
- **Where:** `Scanner.lean`, migration planning section in `Architecture.lean`.
- **Snippet:**
```lean
import Mathlib.Order.FixedPoints

variable {L : Type*} [CompleteLattice L]
variable (f : L →o L)

#check OrderHom.lfp
```

### 5. `Mathlib.Order.Monotone.Basic`
- **Why:** Immediate monotonicity lemmas for proving realm transitions preserve ordering/non-interference invariants.
- **Where:** `Worker.lean`, `RMEModel.lean`.
- **Snippet:**
```lean
import Mathlib.Order.Monotone.Basic

variable {α β : Type*} [Preorder α] [Preorder β]
variable {f : α → β}

example (hf : Monotone f) {a b} (h : a ≤ b) : f a ≤ f b := hf h
```

### 6. `Mathlib/Order/Relation`
- **Why:** Relational composition/reflexive-transitive closure utilities for machine-step and unwinding-style reasoning.
- **Where:** `Worker.lean`, `RMEModel.lean`.
- **Snippet:**
```lean
import Mathlib.Order.Relation

variable {σ : Type*}
variable (step : σ → σ → Prop)

#check Relation.TransGen step
```

### 7. `Mathlib/Data/Set/Lattice`
- **Why:** Set-theoretic lattice modeling for capability sets, accessible granules, and compartment reachability.
- **Where:** `Basic.lean`, `Architecture.lean`.
- **Snippet:**
```lean
import Mathlib.Data.Set.Lattice

variable {α : Type*}

def disjointCaps (A B : Set α) : Prop := Disjoint A B
```

### 8. `Mathlib/Data/Finset/Lattice`
- **Why:** Stronger support than ad hoc unions for finite dependency sets and conflict lemmas (`disjoint_left`, subset algebra).
- **Where:** `Basic.lean`, `Scanner.lean`, `Worker.lean`.
- **Snippet:**
```lean
import Mathlib.Data.Finset.Lattice

variable {α : Type*} [DecidableEq α]
example (s t : Finset α) : s ⊔ t = s ∪ t := rfl
```

### 9. `Mathlib/Data/Rel`
- **Why:** Generic relational reasoning for low-equivalence and observational equivalence parameterization.
- **Where:** New non-interference file, then used by `Worker.lean`.
- **Snippet:**
```lean
import Mathlib.Data.Rel

variable {α : Type*}
variable (R : α → α → Prop)

#check Relator.LeftTotal
```

### 10. `Mathlib/Logic/Function/Basic`
- **Why:** Function-level lemmas for compositional security pipelines (scanner ∘ normalization ∘ enforcement).
- **Where:** `Scanner.lean`, `Architecture.lean`.
- **Snippet:**
```lean
import Mathlib.Logic.Function.Basic

example {α β γ} (f : α → β) (g : β → γ) : Function.Surjective g → Function.Surjective (g ∘ f) := by
  intro hg
  exact hg.comp (fun _ => by aesop)
```

### 11. `Mathlib/Order/CompletePartialOrder`
- **Why:** Domain-theoretic foundation for iterative/static analyses if scanner evolves to monotone dataflow engine.
- **Where:** advanced scanner analysis module.
- **Snippet:**
```lean
import Mathlib.Order.CompletePartialOrder

#check ωSup
```

### 12. `Mathlib/Tactic` (or selective tactic imports)
- **Why:** Stable proof ergonomics (`aesop`, `omega`, `linarith`, `simp`) for maintainable proofs.
- **Where:** all theorem-heavy files.
- **Snippet:**
```lean
import Mathlib.Tactic

example (a b : Nat) (h : a ≤ b) : a + 1 ≤ b + 1 := by omega
```

---

## 2) Proposed enhancements to existing files

## `IATO/V7/Basic.lean`
1. **Promote dependency labels to a reusable security lattice class.**
```lean
class SecurityLabel (L : Type*) extends CompleteLattice L

def CanFlow {L} [SecurityLabel L] (ℓ₁ ℓ₂ : L) : Prop := ℓ₁ ≤ ℓ₂
```
2. **Add `Disjoint`-based separation theorem for finite dependencies.**
```lean
theorem disjoint_symm_dep (φ ψ : DepSet) : Disjoint φ ψ → Disjoint ψ φ := by
  intro h; simpa [disjoint_comm] using h
```
3. **Monotone join growth theorem.**
```lean
theorem deps_join_monotone_left (φ ψ : DepSet) : φ ≤ φ ⊔ ψ := by exact le_sup_left
```

## `IATO/V7/Worker.lean`
1. **Low-equivalence parameterized by observer domain.**
```lean
def lowEq (obs : Domain) (w₁ w₂ : Worker) : Prop :=
  (w₁.domain = obs → w₂.domain = obs) ∧ (w₁.deps = w₂.deps)
```
2. **Define worker transition as relation + unwinding obligations.**
```lean
def Step : Worker → Worker → Prop := fun w w' => w'.deps ≤ w.deps ⊔ w'.deps

def locallyRespecting (obs : Domain) : Prop :=
  ∀ {w₁ w₂ w₁'}, lowEq obs w₁ w₂ → Step w₁ w₁' → ∃ w₂', Step w₂ w₂' ∧ lowEq obs w₁' w₂'
```
3. **Composition monotonicity theorem.**
```lean
theorem compose_deps_upper_bound (ws : List Worker) (w : Worker) (h : w ∈ ws) :
  w.deps ≤ (Worker.compose ws).deps := by
  -- prove via foldl monotonic accumulation over `⊔`
  admit
```

## `IATO/V7/Architecture.lean`
1. **Global non-overlap invariant as pairwise `Disjoint`.**
```lean
def pairwiseDisjointDeps (ws : List Worker) : Prop :=
  ws.Pairwise (fun w₁ w₂ => Disjoint w₁.deps w₂.deps)
```
2. **Secure architecture theorem decomposed into reusable lemmas.**
```lean
theorem architectureSecure_of_pairwise (h : pairwiseDisjointDeps ws) : architectureSecure ws := by
  -- bridge `Pairwise` to index-based statement
  admit
```
3. **Policy refinement theorem via monotone map.**
```lean
variable (policyRefine : DepSet →o DepSet)

theorem secure_under_refinement (hmono : True) :
  architectureSecure ws → architectureSecure (ws.map (fun w => { w with deps := policyRefine w.deps })) := by
  intro hs; admit
```

## `IATO/V7/Scanner.lean`
1. **Scanner soundness against a conflict predicate.**
```lean
def conflicts (w ref : Worker) : Prop := ¬ Worker.compatible w ref

theorem scanLegacy_sound (lw : LegacyWorker) (w : Worker)
  (h : scanLegacy lw = (some w, false)) : conflicts w newReferenceWorker := by
  -- unfold `scanLegacy`; split on parsers
  admit
```
2. **Monotonicity of report incompatibility under list extension.**
```lean
theorem scanAll_incompat_mono (xs ys : List LegacyWorker) :
  (scanAll xs).incompatible.length ≤ (scanAll (xs ++ ys)).incompatible.length := by
  admit
```
3. **Normalization idempotence lemmas (trim/lower/domain parse).**
```lean
theorem parseDomain_idem (s : String) : parseDomain s = parseDomain (s.trim.toLower) := by
  simp [parseDomain]
```

---

## 3) New mathlib-inspired concepts/files to add

1. **`IATO/V7/SecurityLattice.lean`**
- Define confidentiality/integrity product lattice (`Lconf × Lint`) with flow relation.
- Provide reusable lemmas: `flow_trans`, `join_min_upper`, `meet_max_lower`.
- Heavy use: `Order.Lattice`, `Order.Hom.CompleteLattice`.

2. **`IATO/V7/NonInterference.lean`**
- Parameterized transition system `(State, Step)` and observer projection `obsView`.
- Define low-equivalence and prove unwinding theorem:
  - `output_consistency`
  - `step_consistency`
  - `local_respect` ⇒ `nonInterference` (via transitive closure).
- Heavy use: `Order.Relation`, `Data.Rel`, `Relation.TransGen`.

3. **`IATO/V7/RMETransitions.lean`**
- Model Realm/Root/Secure/Normal transitions as relation over machine state.
- Add theorem: **realm entry preserves low-equivalence for non-observable components**.
- Use monotone state transformers (`State →o State`) for transition policies.

4. **`IATO/V7/ComplianceGalois.lean`**
- Formalize abstraction from concrete execution traces to control-evidence lattice.
- `α : ExecState → ControlEvidence`, `γ : ControlEvidence → Set ExecState`, prove `GaloisConnection α γ`.
- Enables proof that strengthened policy implies stronger evidence obligations.

5. **`IATO/V7/ConflictClosure.lean`**
- Define closure operator for dependency conflict resolution and prove idempotent/monotone/extensive properties.
- Use `Order.FixedPoints` for least fixed-point remediation plan guarantees.

---

## 4) General advice

- **Avoid reinventing:** Before custom definitions, check whether mathlib already has them (`OrderHom`, `GaloisConnection`, `CompleteLattice`, `Pairwise`, `Disjoint`, closures/fixed points).
- **Import discipline:**
  - `Basic.lean`: only foundational algebra/order imports and typeclasses.
  - `Worker.lean`: relation + monotone reasoning imports.
  - `Scanner.lean`: finite collections + list monotonicity + parser lemmas.
  - `Architecture.lean`: composition lemmas, pairwise invariants, compliance mapping.
  - Keep heavy tactic imports local to theorem files to reduce compile footprint.
- **RME modeling pitfalls:**
  - Don’t conflate *hardware privilege* with *information-flow levels*; model distinct lattices and connect via monotone/Galois maps.
  - Be explicit about observability (what each world can read) to avoid vacuous non-interference.
  - Separate deterministic transition lemmas from nondeterministic scheduler assumptions.
- **Audit credibility uplift:**
  - Reusable generic theorems (unwinding, monotone preservation, adjoint abstraction) let you instantiate controls repeatedly for SOC2/ISM evidence.
  - A small set of abstract mathlib-backed core lemmas lowers reviewer risk versus many bespoke proofs.

## Useful docs
- Mathlib docs index: <https://leanprover-community.github.io/mathlib4_docs/>
- Search API: <https://leansearch.net/>
