#ifndef PKCS11_SIGNER_H
#define PKCS11_SIGNER_H

#include <stddef.h>

int pkcs11_initialize_from_env(void);
int sign_log_hsm(unsigned char *data, size_t len);
int sign_log_hsm_batch(const unsigned char *digests,
                       size_t digest_len,
                       size_t batch_size,
                       unsigned char *signatures,
                       size_t *signature_lens,
                       size_t signature_stride);
const unsigned char *pkcs11_last_signature(size_t *len_out);
void pkcs11_cleanup(void);

#endif
