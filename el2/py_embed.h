#ifndef IATO_PY_EMBED_H
#define IATO_PY_EMBED_H

#include <stddef.h>
#include <stdint.h>

#define IATO_PY_OK              0
#define IATO_PY_ERR_INIT        -1
#define IATO_PY_ERR_IMPORT      -2
#define IATO_PY_ERR_CALL        -3
#define IATO_PY_ERR_RESULT      -4
#define IATO_PY_ERR_LOCK        -5

int iato_py_embed_init(void);
int iato_py_validate_credential(const uint8_t *cred, size_t cred_len, uint32_t stream_id,
                                uint64_t *out_pa_base, uint64_t *out_pa_limit, uint8_t *out_permissions);
void iato_py_embed_shutdown(void);

#endif
