#include "el2_binding_table.h"
#include "el2_smmu.h"
#include <string.h>

static binding_entry_t g_bindings[MAX_BINDINGS];
static uint32_t g_lock;

static void lock_table(void) {
#ifdef __aarch64__
    uint32_t tmp, res;
    do {
        do { __asm__ volatile("ldaxr %w0, [%1]" : "=&r"(tmp) : "r"(&g_lock) : "memory"); } while (tmp);
        __asm__ volatile("stlxr %w0, %w2, [%1]" : "=&r"(res) : "r"(&g_lock), "r"(1) : "memory");
    } while (res);
#else
    while (__sync_lock_test_and_set(&g_lock, 1)) {}
#endif
}
static void unlock_table(void) {
#ifdef __aarch64__
    __asm__ volatile("stlr %w0, [%1]" :: "r"(0), "r"(&g_lock) : "memory");
#else
    __sync_lock_release(&g_lock);
#endif
}

static binding_entry_t *find_or_alloc(stream_id_t sid) {
    binding_entry_t *free_e = NULL;
    for (size_t i = 0; i < MAX_BINDINGS; i++) {
        if (g_bindings[i].status != BINDING_EMPTY && g_bindings[i].stream_id == sid) return &g_bindings[i];
        if (!free_e && g_bindings[i].status == BINDING_EMPTY) free_e = &g_bindings[i];
    }
    return free_e;
}

el2_err_t el2_binding_commit(binding_entry_t *entry) {
    el2_err_t rc = el2_smmu_write_ste(entry->stream_id, &entry->credential.pa_range,
                                      entry->credential.permissions, entry->credential.expiry_ns);
    if (rc != EL2_OK) return rc;
    entry->status = BINDING_ACTIVE;
    entry->expiry_ns = entry->credential.expiry_ns;
    return EL2_OK;
}

el2_err_t el2_binding_set_spdm(stream_id_t stream_id, const sha256_t cert_hash, const sha256_t session_id) {
    lock_table();
    binding_entry_t *e = find_or_alloc(stream_id);
    if (!e) { unlock_table(); return EL2_ERR_NO_BINDING; }
    e->stream_id = stream_id;
    memcpy(e->device_cert_hash, cert_hash, sizeof(sha256_t));
    memcpy(e->spdm_session_id, session_id, sizeof(sha256_t));
    if (e->status == BINDING_NFC_DONE) {
        if (memcmp(e->credential.spdm_session_id, e->spdm_session_id, sizeof(sha256_t)) != 0) {
            /* SECURITY: NFC credential bound to different SPDM transcript; prevents cross-device/session credential reuse. */
            unlock_table();
            return EL2_ERR_SESSION_MISMATCH;
        }
        el2_err_t rc = el2_binding_commit(e);
        unlock_table();
        return rc;
    }
    e->status = BINDING_SPDM_DONE;
    unlock_table();
    return EL2_OK;
}

el2_err_t el2_binding_set_nfc(stream_id_t stream_id, const ste_credential_t *cred) {
    lock_table();
    binding_entry_t *e = find_or_alloc(stream_id);
    if (!e || (e->status == BINDING_EMPTY && e->stream_id == 0)) {
        unlock_table();
        return EL2_ERR_NO_BINDING;
    }
    e->stream_id = stream_id;
    memcpy(&e->credential, cred, sizeof(*cred));
    if (e->status == BINDING_SPDM_DONE || e->status == BINDING_ACTIVE || e->status == BINDING_EXPIRED) {
        if (memcmp(e->spdm_session_id, cred->spdm_session_id, sizeof(sha256_t)) != 0) {
            /* SECURITY: session mismatch rejects credentials not tied to attested device transcript hash. */
            unlock_table();
            return EL2_ERR_SESSION_MISMATCH;
        }
        el2_err_t rc = el2_binding_commit(e);
        unlock_table();
        return rc;
    }
    e->status = BINDING_NFC_DONE;
    unlock_table();
    return EL2_OK;
}

el2_err_t el2_binding_fault(stream_id_t stream_id) {
    lock_table();
    for (size_t i = 0; i < MAX_BINDINGS; i++) {
        if (g_bindings[i].stream_id == stream_id && g_bindings[i].status != BINDING_EMPTY) {
            el2_smmu_fault_ste(stream_id);
            g_bindings[i].status = BINDING_EXPIRED;
            unlock_table();
            return EL2_OK;
        }
    }
    unlock_table();
    return EL2_ERR_NO_BINDING;
}

const binding_entry_t *el2_binding_get(stream_id_t stream_id) {
    for (size_t i=0;i<MAX_BINDINGS;i++) {
        if (g_bindings[i].stream_id == stream_id && g_bindings[i].status != BINDING_EMPTY) return &g_bindings[i];
    }
    return NULL;
}

binding_entry_t *el2_binding_table_raw(void) { return g_bindings; }
uint32_t *el2_binding_lock_word(void) { return &g_lock; }
