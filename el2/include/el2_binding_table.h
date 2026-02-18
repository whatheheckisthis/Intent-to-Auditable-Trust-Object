#ifndef EL2_BINDING_TABLE_H
#define EL2_BINDING_TABLE_H

#include "el2_types.h"

el2_err_t el2_binding_set_spdm(stream_id_t stream_id, const sha256_t cert_hash, const sha256_t session_id);
el2_err_t el2_binding_set_nfc(stream_id_t stream_id, const ste_credential_t *cred);
el2_err_t el2_binding_commit(binding_entry_t *entry);
el2_err_t el2_binding_fault(stream_id_t stream_id);
const binding_entry_t *el2_binding_get(stream_id_t stream_id);
binding_entry_t *el2_binding_table_raw(void);
uint32_t *el2_binding_lock_word(void);

#endif
