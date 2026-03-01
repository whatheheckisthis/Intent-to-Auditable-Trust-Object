# ARMv9-A RME Compliance Evidence

Generated at: `2026-03-01T02:21:45.196470+00:00`
Git commit: `d32ecab6e5df56ca3d6f0d665d9be2616aaed544`

| Framework | Control | Requirement | Model Element | Proof Reference | Source File | Status |
|---|---|---|---|---|---|---|
| ARM CCA / RME | RMI_DATA_CREATE_AUTHZ | Data granules can only be created for active realms from delegated granules. | `step / RmiCall.DataCreate` | `wf_dataCreate` | `IATO_V7/IATO/V7/RMEModel.lean` | satisfied |
| ARM CCA / RME | RMI_REALM_DESTROY_SCRUB | Realm destroy invalidates realm-owned/data granules. | `realmDestroyGranule` | `realmDestroy_no_data_owner` | `IATO_V7/IATO/V7/RMEModel.lean` | satisfied |
| ARM CCA / RME | RMI_TRACE_INVARIANT | Well-formedness is preserved across arbitrary RMI traces. | `execTrace` | `wf_execTrace` | `IATO_V7/IATO/V7/RMEModel.lean` | satisfied |
| SOC2 | CC6.1 | Logical access and privilege boundaries are enforced. | `RmeState.wf` | `wf_step` | `IATO_V7/IATO/V7/RMEModel.lean` | satisfied |
| ISM | 0460 | Privileged domain separation controls are verifiable. | `GranuleState + RmiCall` | `IATO.V7.RMEModel` | `IATO_V7/IATO/V7/RMEModel.lean` | satisfied |
