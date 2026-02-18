#ifndef MBEDTLS_GCM_H
#define MBEDTLS_GCM_H
#include <stddef.h>
#define MBEDTLS_CIPHER_ID_AES 1
#define MBEDTLS_GCM_ENCRYPT 1
typedef struct { unsigned char key[32]; } mbedtls_gcm_context;
void mbedtls_gcm_init(mbedtls_gcm_context *ctx);
void mbedtls_gcm_free(mbedtls_gcm_context *ctx);
int mbedtls_gcm_setkey(mbedtls_gcm_context *ctx, int cipher, const unsigned char *key, unsigned int keybits);
int mbedtls_gcm_auth_decrypt(mbedtls_gcm_context *ctx, size_t length, const unsigned char *iv, size_t iv_len,
                             const unsigned char *add, size_t add_len, const unsigned char *tag, size_t tag_len,
                             const unsigned char *input, unsigned char *output);
int mbedtls_gcm_crypt_and_tag(mbedtls_gcm_context *ctx, int mode, size_t length, const unsigned char *iv,
                              size_t iv_len, const unsigned char *add, size_t add_len,
                              const unsigned char *input, unsigned char *output,
                              size_t tag_len, unsigned char *tag);
#endif
