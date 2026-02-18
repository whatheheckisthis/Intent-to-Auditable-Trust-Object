#ifndef EL2_TYPES_H
#define EL2_TYPES_H

#include <stdint.h>
#include <stddef.h>

typedef uint32_t  stream_id_t;
typedef uint64_t  pa_t;
typedef uint64_t  el2_time_t;
typedef uint8_t   nonce_t[32];
typedef uint8_t   sha256_t[32];
typedef uint8_t   hmac_t[32];
typedef uint8_t   cert_der_t[];

typedef struct {
    pa_t      base;
    pa_t      limit;
    uint32_t  flags;
    uint32_t  _pad;
} pa_range_t;

typedef struct {
    stream_id_t  stream_id;
    pa_range_t   pa_range;
    uint32_t     permissions;
    uint32_t     _pad;
    el2_time_t   expiry_ns;
    sha256_t     spdm_session_id;
    nonce_t      nonce;
} ste_credential_t;

typedef struct {
    uint8_t      ciphertext[256];
    uint8_t      ephemeral_pub[65];
    hmac_t       mac;
    uint32_t     version;
    uint32_t     _pad;
} nfc_blob_t;

typedef enum {
    BINDING_EMPTY        = 0,
    BINDING_SPDM_DONE    = 1,
    BINDING_NFC_DONE     = 2,
    BINDING_ACTIVE       = 3,
    BINDING_EXPIRED      = 4,
} binding_status_t;

typedef struct {
    stream_id_t      stream_id;
    sha256_t         device_cert_hash;
    sha256_t         spdm_session_id;
    ste_credential_t credential;
    binding_status_t status;
    el2_time_t       expiry_ns;
    uint32_t         _pad;
} binding_entry_t;

#define STE_WORD0_V          (1u << 0)
#define STE_WORD0_CFG_BYPASS (0x4u << 1)
#define STE_WORD0_CFG_TRANS  (0x5u << 1)
#define STE_WORD0_FAULT      (0x0u << 1)
#define SMMU_STE_DWORDS      8
typedef uint64_t smmu_ste_t[SMMU_STE_DWORDS];

typedef enum {
    EL2_OK                   =  0,
    EL2_ERR_BAD_SIGNATURE    = -1,
    EL2_ERR_NONCE_REPLAYED   = -2,
    EL2_ERR_EXPIRED          = -3,
    EL2_ERR_SESSION_MISMATCH = -4,
    EL2_ERR_TRUST_SEALED     = -5,
    EL2_ERR_SPDM_FAILED      = -6,
    EL2_ERR_NO_BINDING       = -7,
    EL2_ERR_SMMU_FAULT       = -8,
    EL2_ERR_ENROLLMENT_MODE  = -9,
    EL2_ERR_PCR_EXTEND       = -10,
} el2_err_t;

#define MAX_BINDINGS 64
#define MAX_SPDM_CA_CERTS 4

#endif
