#ifndef IATO_TELEMETRY_FOLD_SVE2_H
#define IATO_TELEMETRY_FOLD_SVE2_H

#include <arm_sve.h>
#include <stdint.h>

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error "telemetry_fold_sve2 requires AArch64 SVE2"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define IATO_TELEMETRY_COEFFS 256u
#define IATO_MOD_Q 3329u
#define IATO_MONT_QINV 3327u
#define IATO_ZREG_FILE_SIZE 32u
#define IATO_PIPELINE_ACC_REGS 8u   /* Z16-Z23 */
#define IATO_PIPELINE_WORK_REGS 8u  /* Z0-Z7 */
#define IATO_PIPELINE_TEMP_REGS 8u  /* Z24-Z31 */

#if defined(__ARM_FEATURE_SVE_BITS)
_Static_assert(__ARM_FEATURE_SVE_BITS == 512,
               "Pipeline assumes 512-bit SVE2 vectors.");
#endif

_Static_assert(IATO_PIPELINE_ACC_REGS + IATO_PIPELINE_WORK_REGS +
                   IATO_PIPELINE_TEMP_REGS <= IATO_ZREG_FILE_SIZE,
               "Pipeline register plan exceeds Z0-Z31");

/* 256 coefficients over 16-bit lanes in 512-bit vectors => 8 vectors. */
#define IATO_COEFF_LANES_PER_ZREG (512u / 16u)
#define IATO_POLY_ZREGS (IATO_TELEMETRY_COEFFS / IATO_COEFF_LANES_PER_ZREG)
_Static_assert(IATO_TELEMETRY_COEFFS % IATO_COEFF_LANES_PER_ZREG == 0,
               "Vector lanes must fully tile each polynomial.");
_Static_assert(IATO_POLY_ZREGS == IATO_PIPELINE_ACC_REGS,
               "Accumulator lane plan must map to Z16-Z23.");

typedef struct {
    svuint16_t v[IATO_POLY_ZREGS];
} iato_poly_zreg_t;

typedef struct {
    svuint8_t limb[4]; /* 4x128-bit limbs => 512-bit witness */
} iato_witness_512_t;

typedef struct {
    svuint16_t hi;
    svuint16_t lo;
} iato_ntt_pair_t;

svuint16_t montgomery_reduce_sve2(svuint32_t zn);
iato_ntt_pair_t ntt_butterfly_sve2(svuint16_t a, svuint16_t b, svuint16_t twiddle);
svuint16_t fold_accumulator(svuint16_t z_in, svuint16_t z_acc, uint16_t r_i);
svuint16_t integer_divergence_track(svuint16_t z_acc, svuint16_t z_noise_seed);
void snark_wrap_edwards(const iato_poly_zreg_t *z_acc, iato_witness_512_t *witness_out);
void run_recursive_fold_epoch(uint64_t telemetry_items,
                              const iato_poly_zreg_t *stream,
                              uint16_t challenge_r,
                              iato_witness_512_t *witness_out);
void run_recursive_fold_100m(const iato_poly_zreg_t *stream,
                             uint16_t challenge_r,
                             iato_witness_512_t *witness_out);
void scrub_pipeline_registers(void);

#ifdef __cplusplus
}
#endif

#endif
