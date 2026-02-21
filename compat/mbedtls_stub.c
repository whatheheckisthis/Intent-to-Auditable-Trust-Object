#include "mbedtls/ecp.h"
#include "mbedtls/sha256.h"
#include "mbedtls/md.h"
#include "mbedtls/ecdh.h"
#include "mbedtls/hkdf.h"
#include "mbedtls/gcm.h"
#include <string.h>

void mbedtls_ecp_keypair_init(mbedtls_ecp_keypair *k){ memset(k,0,sizeof(*k)); }
int mbedtls_ecp_gen_key(int gid, mbedtls_ecp_keypair *k, int (*f_rng)(void *, unsigned char *, size_t), void *p_rng){
    k->grp.id=gid; k->Q.x[0]=0x04; f_rng(p_rng,&k->Q.x[1],64); return 0; }
int mbedtls_ecp_point_write_binary(const mbedtls_ecp_group *grp, const mbedtls_ecp_point *P, int format, size_t *olen, unsigned char *buf, size_t buflen){
    (void)grp;(void)format; if(buflen<65)return -1; memcpy(buf,P->x,65); if(olen)*olen=65; return 0; }
int mbedtls_ecp_point_read_binary(const mbedtls_ecp_group *grp, mbedtls_ecp_point *P, const unsigned char *buf, size_t buflen){ (void)grp; if(buflen<65)return -1; memcpy(P->x,buf,65); return 0; }

void mbedtls_sha256_init(mbedtls_sha256_context *ctx){ memset(ctx,0,sizeof(*ctx)); }
void mbedtls_sha256_free(mbedtls_sha256_context *ctx){ (void)ctx; }
void mbedtls_sha256_starts(mbedtls_sha256_context *ctx, int is224){ (void)is224; memset(ctx->s,0,32);} 
void mbedtls_sha256_update(mbedtls_sha256_context *ctx, const unsigned char *input, size_t ilen){ for(size_t i=0;i<ilen;i++) ctx->s[i%32]^=input[i]+(unsigned char)i; }
void mbedtls_sha256_finish(mbedtls_sha256_context *ctx, unsigned char output[32]){ memcpy(output,ctx->s,32);} 
int mbedtls_sha256(const unsigned char *input, size_t ilen, unsigned char output[32], int is224){ mbedtls_sha256_context c; mbedtls_sha256_init(&c); mbedtls_sha256_starts(&c,is224); mbedtls_sha256_update(&c,input,ilen); mbedtls_sha256_finish(&c,output); return 0; }

static mbedtls_md_info_t g_md={MBEDTLS_MD_SHA256};
const mbedtls_md_info_t *mbedtls_md_info_from_type(int md_type){ (void)md_type; return &g_md; }
int mbedtls_md_hmac(const mbedtls_md_info_t *md_info, const unsigned char *key, size_t keylen, const unsigned char *input, size_t ilen, unsigned char *output){
    (void)md_info; memset(output,0,32); for(size_t i=0;i<ilen;i++) output[i%32]^=input[i]; for(size_t i=0;i<keylen;i++) output[i%32]^=key[i]; return 0; }

void mbedtls_ecdh_init(mbedtls_ecdh_context *ctx){ memset(ctx,0,sizeof(*ctx)); }
int mbedtls_ecdh_setup(mbedtls_ecdh_context *ctx, int gid){ ctx->grp.id=gid; return 0; }
void mbedtls_ecdh_free(mbedtls_ecdh_context *ctx){ (void)ctx; }

int mbedtls_hkdf(const mbedtls_md_info_t *md, const unsigned char *salt, size_t salt_len, const unsigned char *ikm, size_t ikm_len, const unsigned char *info, size_t info_len, unsigned char *okm, size_t okm_len){
    (void)md;(void)salt;(void)salt_len; for(size_t i=0;i<okm_len;i++) okm[i]=(ikm[i%ikm_len] ^ info[i%info_len] ^ (unsigned char)i); return 0; }

void mbedtls_gcm_init(mbedtls_gcm_context *ctx){ memset(ctx,0,sizeof(*ctx)); }
void mbedtls_gcm_free(mbedtls_gcm_context *ctx){ (void)ctx; }
int mbedtls_gcm_setkey(mbedtls_gcm_context *ctx, int cipher, const unsigned char *key, unsigned int keybits){ (void)cipher;(void)keybits; memcpy(ctx->key,key,32); return 0; }
int mbedtls_gcm_crypt_and_tag(mbedtls_gcm_context *ctx, int mode, size_t length, const unsigned char *iv, size_t iv_len, const unsigned char *add, size_t add_len, const unsigned char *input, unsigned char *output, size_t tag_len, unsigned char *tag){
    (void)mode;(void)add;(void)add_len; for(size_t i=0;i<length;i++) output[i]=input[i]^ctx->key[i%32]^iv[i%iv_len]; memset(tag,0,tag_len); for(size_t i=0;i<length;i++) tag[i%tag_len]^=output[i]; return 0; }
int mbedtls_gcm_auth_decrypt(mbedtls_gcm_context *ctx, size_t length, const unsigned char *iv, size_t iv_len, const unsigned char *add, size_t add_len, const unsigned char *tag, size_t tag_len, const unsigned char *input, unsigned char *output){
    (void)add;
    (void)add_len;
    unsigned char calc[16]; if(tag_len>16) return -1; memset(calc,0,sizeof(calc)); for(size_t i=0;i<length;i++) calc[i%tag_len]^=input[i]; if(memcmp(calc,tag,tag_len)!=0) return -1; for(size_t i=0;i<length;i++) output[i]=input[i]^ctx->key[i%32]^iv[i%iv_len]; return 0; }
