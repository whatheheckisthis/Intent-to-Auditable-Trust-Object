/*
 * Conceptual OP-TEE TA snippet for Ed25519 signing.
 * The host C dispatcher sends only a digest; the private key stays in TEE secure storage.
 */

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#define TA_CMD_SIGN_DIGEST 0x00000001

static TEE_ObjectHandle g_key = TEE_HANDLE_NULL;

static TEE_Result load_or_create_ed25519_key(void) {
    TEE_Result res;
    const char object_id[] = "ed25519_osint_signing_key";

    res = TEE_OpenPersistentObject(
        TEE_STORAGE_PRIVATE,
        object_id,
        sizeof(object_id),
        TEE_DATA_FLAG_ACCESS_READ | TEE_DATA_FLAG_ACCESS_WRITE,
        &g_key
    );

    if (res == TEE_SUCCESS) {
        return TEE_SUCCESS;
    }

    res = TEE_AllocateTransientObject(TEE_TYPE_ED25519_KEYPAIR, 256, &g_key);
    if (res != TEE_SUCCESS) {
        return res;
    }

    res = TEE_GenerateKey(g_key, 256, NULL, 0);
    if (res != TEE_SUCCESS) {
        TEE_FreeTransientObject(g_key);
        g_key = TEE_HANDLE_NULL;
        return res;
    }

    return TEE_CreatePersistentObject(
        TEE_STORAGE_PRIVATE,
        object_id,
        sizeof(object_id),
        TEE_DATA_FLAG_ACCESS_READ | TEE_DATA_FLAG_ACCESS_WRITE,
        g_key,
        NULL,
        0,
        &g_key
    );
}

TEE_Result TA_InvokeCommandEntryPoint(void *sess_ctx, uint32_t cmd_id, uint32_t param_types, TEE_Param params[4]) {
    (void)sess_ctx;

    if (cmd_id != TA_CMD_SIGN_DIGEST) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    const uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT, TEE_PARAM_TYPE_MEMREF_OUTPUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
    if (param_types != exp) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    TEE_Result res = load_or_create_ed25519_key();
    if (res != TEE_SUCCESS) {
        return res;
    }

    TEE_OperationHandle op = TEE_HANDLE_NULL;
    res = TEE_AllocateOperation(&op, TEE_ALG_ED25519, TEE_MODE_SIGN, 256);
    if (res != TEE_SUCCESS) {
        return res;
    }

    res = TEE_SetOperationKey(op, g_key);
    if (res != TEE_SUCCESS) {
        TEE_FreeOperation(op);
        return res;
    }

    uint32_t sig_len = params[1].memref.size;
    res = TEE_AsymmetricSignDigest(op, NULL, 0, params[0].memref.buffer, params[0].memref.size, params[1].memref.buffer, &sig_len);
    params[1].memref.size = sig_len;

    TEE_FreeOperation(op);
    return res;
}
