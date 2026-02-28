# IATO v7 Formal Assurance Notebooks

This index links the generated notebook suite (24 notebooks) and the ACSL C sketch used for verification sketches.

## Alloy (`module IatoV7Security`)
- [alloy_iato_v7_01_model_core.ipynb](./alloy_iato_v7_01_model_core.ipynb)
- [alloy_iato_v7_02_dependencies.ipynb](./alloy_iato_v7_02_dependencies.ipynb)
- [alloy_iato_v7_03_scrubbing_obligations.ipynb](./alloy_iato_v7_03_scrubbing_obligations.ipynb)
- [alloy_iato_v7_04_nonce_freshness.ipynb](./alloy_iato_v7_04_nonce_freshness.ipynb)
- [alloy_iato_v7_05_audit_log_signatures.ipynb](./alloy_iato_v7_05_audit_log_signatures.ipynb)
- [alloy_iato_v7_06_zero_persistent_secrets.ipynb](./alloy_iato_v7_06_zero_persistent_secrets.ipynb)

## TLA+ (`---- MODULE IatoV7Spec ----`)
- [tla_iato_v7_01_state_machine.ipynb](./tla_iato_v7_01_state_machine.ipynb)
- [tla_iato_v7_02_transition_safety.ipynb](./tla_iato_v7_02_transition_safety.ipynb)
- [tla_iato_v7_03_liveness_progress.ipynb](./tla_iato_v7_03_liveness_progress.ipynb)
- [tla_iato_v7_04_unmitigated_dependency_detection.ipynb](./tla_iato_v7_04_unmitigated_dependency_detection.ipynb)
- [tla_iato_v7_05_nonce_and_replay_protection.ipynb](./tla_iato_v7_05_nonce_and_replay_protection.ipynb)
- [tla_iato_v7_06_audit_integrity.ipynb](./tla_iato_v7_06_audit_integrity.ipynb)

## Coq (`Module IatoV7Proofs.`)
- [coq_iato_v7_01_foundations.ipynb](./coq_iato_v7_01_foundations.ipynb)
- [coq_iato_v7_02_security_states.ipynb](./coq_iato_v7_02_security_states.ipynb)
- [coq_iato_v7_03_hw_enforcement_lemmas.ipynb](./coq_iato_v7_03_hw_enforcement_lemmas.ipynb)
- [coq_iato_v7_04_scrubbing_soundness.ipynb](./coq_iato_v7_04_scrubbing_soundness.ipynb)
- [coq_iato_v7_05_nonce_freshness_proofs.ipynb](./coq_iato_v7_05_nonce_freshness_proofs.ipynb)
- [coq_iato_v7_06_audit_log_authenticity.ipynb](./coq_iato_v7_06_audit_log_authenticity.ipynb)

## C/ACSL proof sketches
- [c_proof_sketch_iato_v7_01_contracts.ipynb](./c_proof_sketch_iato_v7_01_contracts.ipynb)
- [c_proof_sketch_iato_v7_02_memory_isolation.ipynb](./c_proof_sketch_iato_v7_02_memory_isolation.ipynb)
- [c_proof_sketch_iato_v7_03_scrub_and_zeroization.ipynb](./c_proof_sketch_iato_v7_03_scrub_and_zeroization.ipynb)
- [c_proof_sketch_iato_v7_04_nonce_lifecycle.ipynb](./c_proof_sketch_iato_v7_04_nonce_lifecycle.ipynb)
- [c_proof_sketch_iato_v7_05_signed_audit_pipeline.ipynb](./c_proof_sketch_iato_v7_05_signed_audit_pipeline.ipynb)
- [c_proof_sketch_iato_v7_06_verification_summary.ipynb](./c_proof_sketch_iato_v7_06_verification_summary.ipynb)

## C source sketch
- [iato_v7_sketch.c](./iato_v7_sketch.c)

## Assurance coverage
- Hardware-enforced separation
- Unmitigated dependency control and scrubbing obligations
- Nonce freshness and replay resistance
- Signed audit log integrity
- Zero persistent secrets after transitions
