# Armv9-A Confidential Compute Control Model

## FEAT_SSBS and Realm Management Extension (RME)

### Formal Specification, Architectural Mapping, and Security Invariants

---

## 1. Scope

This document formally specifies the interaction between:

* FEAT_SSBS
* Realm Management Extension

within the Arm Confidential Compute Architecture.

The objective is to define:

1. Exact architectural control mappings
2. Microarchitectural behavioral constraints
3. Formal confidentiality invariants
4. Proof sketches ensuring non-speculative leakage
5. Integration constraints consistent with a causally bounded inference lifecycle

The threat model assumes adversarial control of non-Realm execution environments and speculative side-channel capabilities.

---

## 2. Architectural Context

### 2.1 FEAT_SSBS

FEAT_SSBS provides architectural control over **Speculative Store Bypass (SSB)** behavior.

Speculative Store Bypass occurs when:

* A younger load executes speculatively
* Prior store-to-load dependency resolution is unresolved
* Load incorrectly forwards stale or unverified data

FEAT_SSBS enforces:

> Speculative loads MUST NOT bypass unresolved stores when SSBS=1.

#### Architectural Controls

| Control  | Register      | Bit                    | Meaning                              |
| -------- | ------------- | ---------------------- | ------------------------------------ |
| SSBS     | `PSTATE.SSBS` | [SSBS]                 | 1 = Disable Speculative Store Bypass |
| SSBS_CTL | `SCTLR_ELx`   | implementation-defined | System-level enforcement             |

Execution rule:

```
If PSTATE.SSBS == 1:
    Speculative loads must wait for store dependency resolution
```

---

### 2.2 Realm Management Extension (RME)

RME introduces **Realm world** isolation with hardware-enforced memory and execution domains:

World types:

* Non-secure
* Secure
* Realm

Key enforcement primitives:

| Component                      | Function                |
| ------------------------------ | ----------------------- |
| Realm Translation Tables       | Isolated address space  |
| Granule Protection Table (GPT) | Physical memory tagging |
| Realm Management Monitor (RMM) | Realm lifecycle control |

Realm invariant:

```
Non-Realm worlds MUST NOT read or infer Realm memory.
```

---

## 3. Combined Security Objective

Goal:

Prevent speculative execution from becoming a covert channel across Realm boundaries.

Define:

* Secret data domain: S
* Microarchitectural state: μ
* Architectural state: σ
* Speculative state: μ_spec
* Memory granule tag function: G(p)

Confidentiality requirement:

For any time t ≥ M (post-isolation boundary),

```
∂μ_spec / ∂S = 0
```

i.e., speculative microarchitectural state must not encode Realm secrets.

---

## 4. Formal Invariants

### Invariant 1 — Store-Load Ordering Safety

If:

* Load L reads address A
* Store S writes address A
* S precedes L in program order
* Store resolution incomplete

Then:

```
SSBS = 1 ⇒ L cannot execute speculatively
```

Formal constraint:

```
∀ L, S:
    (PO(S, L) ∧ Addr(S) = Addr(L) ∧ unresolved(S))
    ⇒ ¬SpecExec(L)
```

---

### Invariant 2 — Realm Non-Interference

Let:

* W_r = Realm world
* W_nr = Non-Realm world

Then:

```
∀ observable O:
    O(W_nr) ⟂ S(W_r)
```

Meaning observables in non-Realm world are statistically independent of Realm secrets.

Formally:

```
I(S; O_nonrealm) = 0
```

Where I is mutual information.

---

### Invariant 3 — Speculative Blindfold Condition

Define Blindfold Condition B:

```
B := (SSBS = 1) ∧ (ExecutionWorld = Realm)
```

Then for any speculative path π:

```
μ_spec(π) must not encode values derived from S
```

Equivalent to:

```
Encode(μ_spec) ∩ Encode(S) = ∅
```

---

### Invariant 4 — Post-Boundary Stability (t ≥ M)

Let M be the isolation boundary (Realm entry).

For all t ≥ M:

```
State_t = f(State_{t-1}, Inputs_t)
```

Subject to:

```
Inputs_t ∩ NonRealm = ∅
```

Thus no cross-world dependency enters Realm after M.

---

## 5. Proof Sketches

### Proof A — Prevention of Speculative Store Bypass Leakage

Assume:

* Load L speculates
* S contains secret value s ∈ S
* Load reads stale value revealing bit of s

Under SSBS=1:

1. Store dependency must resolve before L executes.
2. Therefore, L cannot execute before correct value committed.
3. No speculative exposure of stale secret.

Contradiction arises if L executes early, violating architectural rule.

Hence:

```
SSBS ⇒ No SSB-based leakage
```

---

### Proof B — Realm Microarchitectural Isolation

Given:

* GPT tags enforce memory domain separation.
* Translation tables separate virtual spaces.
* RMM mediates entry/exit.

Even if speculation occurs:

* Access to non-Realm memory from Realm is blocked by GPT.
* Access to Realm memory from non-Realm faults architecturally.

Thus speculative side effects cannot cross domains.

---

### Proof C — Mutual Information Bound

Let channel capacity C_spec represent speculative channel capacity.

With SSBS enabled and RME enforced:

```
C_spec → 0
```

Because:

* Speculative loads blocked on dependency
* Cross-world memory inaccessible
* Observable timing variations bounded

Therefore:

```
I(S; O) ≤ ε
```

Where ε approaches architectural noise floor.

---

## 6. Architectural Mapping Summary

| Security Property        | Mechanism     | Hardware Source |
| ------------------------ | ------------- | --------------- |
| Store-load serialization | PSTATE.SSBS   | FEAT_SSBS       |
| Realm memory isolation   | GPT + RMM     | RME             |
| Address space isolation  | Realm TTBR    | RME             |
| World switch control     | SCR_EL3 / RMM | RME             |

---

## 7. Integration with Deterministic Inference Architecture

Given prior closed-loop, entropy-governed inference lifecycle:

Let:

* c_t^{(i)} = channel states
* e_t = Bayesian belief state
* μ_t = microarchitectural state

Security requirement:

```
μ_t must not introduce entropy correlated with S
```

Thus:

```
H(μ_t | S) = H(μ_t)
```

Ensuring microarchitectural independence from secret domain.

SSBS provides ordering determinism.
RME provides spatial domain determinism.

Combined:

They form a **speculative isolation envelope**.

---

## 8. Conclusion

* FEAT_SSBS provides fine-grained speculative execution suppression at the store-load boundary.
* RME provides world-level physical and virtual memory isolation.
* Together under Armv9-A CCA, they enforce:

  * Causal separation
  * Speculative blindness
  * Realm confidentiality invariants

This establishes a hardware-enforced non-interference guarantee across speculative and architectural dimensions.

---

## 9. Formalization Artifacts Produced

This repository now includes:

- **4 TLA+ safety models** under `docs/notebooks/tla/armv9_cca/`
- **14 Coq non-interference proof files** under `docs/notebooks/coq/armv9_cca/`

These files provide mechanically-checkable scaffolding for the invariants and proof obligations above.

---


## 10. Manual Alignment Audit Checklist (ARM DDI 0487 cross-check)

Use this checklist to verify generated models are aligned with Armv9-A CCA assumptions and not merely generic barrier semantics:

| Audit Area | Required String | Expected Modeling Behavior |
| --- | --- | --- |
| Security State | `SCR_EL3.NSE`, `SCR_EL3.NS` | World-selection logic must switch execution world from these two bits (`Realm`, `NonRealm`, `Secure`). |
| Speculation Control | `PSTATE.SSBS` | Predicate where `PSTATE.SSBS == 1` prevents speculative load bypass of unresolved prior stores to same address (store-to-load forwarding gate). |
| Memory Tagging | `GPI` and `GPT` | Every `Load`/`Store` transition must check GPT status (indexed by granule/GPI metadata) before memory access is considered granted. |

Threat-model alignment requirement:

* The Hypervisor must be modeled as a **Non-Realm adversary** (`hypervisorTrusted = FALSE`).
* Any model that assumes a "trusted hypervisor" is out-of-scope for Arm CCA Realm threat assumptions.

