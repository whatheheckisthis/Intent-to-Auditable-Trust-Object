#ifndef MBEDTLS_ECP_H
#define MBEDTLS_ECP_H
#include <stddef.h>
#define MBEDTLS_ECP_DP_SECP256R1 1
#define MBEDTLS_ECP_PF_UNCOMPRESSED 0

typedef struct { unsigned char x[65]; } mbedtls_ecp_point;
typedef struct { int id; } mbedtls_ecp_group;
typedef struct { mbedtls_ecp_group grp; mbedtls_ecp_point Q; } mbedtls_ecp_keypair;
void mbedtls_ecp_keypair_init(mbedtls_ecp_keypair *k);
int mbedtls_ecp_gen_key(int gid, mbedtls_ecp_keypair *k, int (*f_rng)(void *, unsigned char *, size_t), void *p_rng);
int mbedtls_ecp_point_write_binary(const mbedtls_ecp_group *grp, const mbedtls_ecp_point *P, int format, size_t *olen, unsigned char *buf, size_t buflen);
int mbedtls_ecp_point_read_binary(const mbedtls_ecp_group *grp, mbedtls_ecp_point *P, const unsigned char *buf, size_t buflen);
#endif
