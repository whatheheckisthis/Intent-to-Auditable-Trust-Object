#ifndef PKCS11_SIGNER_H
#define PKCS11_SIGNER_H

#include <stddef.h>

int pkcs11_initialize_from_env(void);
int sign_log_hsm(unsigned char *data, size_t len);
const unsigned char *pkcs11_last_signature(size_t *len_out);
void pkcs11_cleanup(void);

#endif
