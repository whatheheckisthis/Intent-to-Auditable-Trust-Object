# ARMv9-A RME Compliance Evidence

Generated at: `2026-03-01T00:55:17.435149+00:00`

| Framework | Control | Requirement | Model Element | Proof Reference | Status |
|---|---|---|---|---|---|
| ARM CCA / RME | RMI_DATA_CREATE_AUTHZ | Data granules can only be created for active realms from delegated granules. | `step / RmiCall.DataCreate` | `wf_dataCreate` | satisfied |
| ARM CCA / RME | RMI_REALM_DESTROY_SCRUB | Realm destroy invalidates realm-owned/data granules. | `realmDestroyGranule` | `realmDestroy_no_data_owner` | satisfied |
| SOC2 | CC6.1 | Logical access and privilege boundaries are enforced. | `RmeState.wf` | `wf_step` | satisfied |
| ISM | 0460 | Privileged domain separation controls are verifiable. | `GranuleState + RmiCall` | `IATO.V7.RMEModel` | satisfied |

Evidence source: `IATO_V7/IATO/V7/RMEModel.lean`.
