import os
import pytest


@pytest.mark.parametrize("stream_id", [1, 2, 3])
def test_smc_to_smmu_sim_path(smc_interface, smmu_reader, stream_id):
    cred = bytes([0] * 149)
    rc = smc_interface.provision(stream_id, cred)
    assert rc == 0
    word0 = smmu_reader.read_ste_word0(stream_id)
    assert (word0 & 0x1) == 0x1


def test_replay_rejected(smc_interface):
    cred = bytes([1] * 149)
    assert smc_interface.provision(4, cred) == 0
    assert smc_interface.provision(4, cred) != 0


@pytest.mark.hw
def test_hw_mode_placeholder(hw_mode):
    if not hw_mode:
        pytest.skip("hw mode only")
    assert os.path.exists('/dev')
