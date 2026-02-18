#ifndef MBEDTLS_SHA256_H
#define MBEDTLS_SHA256_H
#include <stddef.h>
typedef struct { unsigned char s[32]; } mbedtls_sha256_context;
void mbedtls_sha256_init(mbedtls_sha256_context *ctx);
void mbedtls_sha256_free(mbedtls_sha256_context *ctx);
void mbedtls_sha256_starts(mbedtls_sha256_context *ctx, int is224);
void mbedtls_sha256_update(mbedtls_sha256_context *ctx, const unsigned char *input, size_t ilen);
void mbedtls_sha256_finish(mbedtls_sha256_context *ctx, unsigned char output[32]);
int mbedtls_sha256(const unsigned char *input, size_t ilen, unsigned char output[32], int is224);
#endif
