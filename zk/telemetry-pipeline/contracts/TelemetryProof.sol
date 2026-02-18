// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IGroth16Verifier {
    function verifyProof(
        uint256[2] calldata _pA,
        uint256[2][2] calldata _pB,
        uint256[2] calldata _pC,
        uint256[] calldata _pubSignals
    ) external view returns (bool);
}

/**
 * @title TelemetryProof
 * @notice Syslog-style audit contract for SNARK-based telemetry attestations.
 *
 * The verifier key itself lives in a generated Groth16 verifier contract; this contract stores
 * its address and routes proof checks to it while persisting sequence+epoch log records.
 */
contract TelemetryProof {
    IGroth16Verifier public immutable verifier;

    struct AuditRecord {
        uint64 epochId;
        uint64 sequenceNo;
        bytes32 witnessHash;
        bool verified;
        uint64 timestamp;
    }

    mapping(uint64 => bool) public results;
    mapping(uint64 => AuditRecord) public records;

    event SyslogAudit(uint64 indexed sequenceNo, uint64 indexed epochId, bytes32 witnessHash, bool verified);

    constructor(address verifierAddress) {
        require(verifierAddress != address(0), "invalid verifier");
        verifier = IGroth16Verifier(verifierAddress);
    }

    /**
     * @notice Submit a proof bound to epoch + sequence number and receive an auditable log event.
     */
    function submitProof(
        uint64 epochId,
        uint64 sequenceNo,
        bytes32 witnessHash,
        uint256[2] calldata pA,
        uint256[2][2] calldata pB,
        uint256[2] calldata pC,
        uint256[] calldata pubSignals
    ) external returns (bool verified) {
        verified = verifier.verifyProof(pA, pB, pC, pubSignals);

        results[sequenceNo] = verified;
        records[sequenceNo] = AuditRecord({
            epochId: epochId,
            sequenceNo: sequenceNo,
            witnessHash: witnessHash,
            verified: verified,
            timestamp: uint64(block.timestamp)
        });

        emit SyslogAudit(sequenceNo, epochId, witnessHash, verified);
    }
}
