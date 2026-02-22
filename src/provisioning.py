from __future__ import annotations

import os
import time

from src.cnthp_sweep import CnthpSweep
from src.el2_timer import CnthpTimerBackend
from src.key_ceremony import KeyCeremony
from src.smmu_controller import SmmuController, SteCredential
from src.spdm_session_factory import make_spdm_session
from src.tpm_enrollment import TpmEnrollmentSeal


class ProvisioningOrchestrator:
    def __init__(self, tpm, ceremony, spdm, smmu, timer, sweep, binding_table):
        self.tpm = tpm
        self.ceremony = ceremony
        self.spdm = spdm
        self.smmu = smmu
        self.timer = timer
        self.sweep = sweep
        self.binding_table = binding_table

    @classmethod
    def create(cls):
        tpm = TpmEnrollmentSeal()
        ceremony = KeyCeremony(tpm)
        if os.environ.get("IATO_HW_MODE", "0") == "1":
            if not ceremony._pub_path().exists():
                ceremony.enroll()
            ceremony.load_public_key()
            smmu = SmmuController(mmio_base=int(os.environ.get("IATO_SMMU_BASE", "0x09050000"), 16))
            timer = CnthpTimerBackend()
        else:
            if not ceremony._pub_path().exists():
                ceremony.re_enroll()
            smmu = SmmuController()
            timer = None
        spdm = make_spdm_session(b"dummy-ca")
        binding_table = {}
        sweep = CnthpSweep(smmu, binding_table, timer=timer)
        return cls(tpm, ceremony, spdm, smmu, timer, sweep, binding_table)

    def provision(self, stream_id, raw_credential) -> str:
        cred = SteCredential(stream_id=stream_id, pa_range_base=raw_credential["pa_base"], pa_range_limit=raw_credential["pa_limit"], permissions=raw_credential["permissions"])
        self.smmu.write_ste(cred)
        self.binding_table[stream_id] = {"expires_at": time.time() + raw_credential.get("ttl", 30)}
        return "OK"

    def start_sweep(self) -> None:
        self.sweep.start()

    def stop_sweep(self) -> None:
        self.sweep.stop()

    def shutdown(self) -> None:
        for stream_id, rec in list(self.smmu._table.items()):
            if rec.get("state") == "PERMITTED":
                self.smmu.fault_everything(stream_id)
        self.stop_sweep()
