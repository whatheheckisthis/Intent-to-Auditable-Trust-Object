#ifndef MBEDTLS_ECDH_H
#define MBEDTLS_ECDH_H
#include "ecp.h"
typedef struct { mbedtls_ecp_group grp; mbedtls_ecp_point Qp; } mbedtls_ecdh_context;
void mbedtls_ecdh_init(mbedtls_ecdh_context *ctx);
int mbedtls_ecdh_setup(mbedtls_ecdh_context *ctx, int gid);
void mbedtls_ecdh_free(mbedtls_ecdh_context *ctx);
#endif
