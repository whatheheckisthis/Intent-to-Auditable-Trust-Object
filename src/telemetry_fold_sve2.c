#include "iato/telemetry_fold_sve2.h"

#include <stddef.h>

/* Constant-time modular helpers over Z_q (q=3329). */
static inline svuint16_t iato_mod_q_add(svuint16_t x, svuint16_t y) {
    const svbool_t pg = svptrue_b16();
    const svuint16_t q = svdup_u16((uint16_t)IATO_MOD_Q);
    svuint16_t s = svadd_u16_x(pg, x, y);
    svuint16_t s_minus_q = svsub_u16_x(pg, s, q);
    svbool_t ge_q = svcmpge_u16(pg, s, q);
    return svsel_u16(ge_q, s_minus_q, s);
}

static inline svuint16_t iato_mod_q_sub(svuint16_t x, svuint16_t y) {
    const svbool_t pg = svptrue_b16();
    const svuint16_t q = svdup_u16((uint16_t)IATO_MOD_Q);
    svuint16_t d = svsub_u16_x(pg, x, y);
    svbool_t borrow = svcmplt_u16(pg, x, y);
    return svsel_u16(borrow, svadd_u16_x(pg, d, q), d);
}

/* NTT Furnace: Montgomery REDC over each 32-bit lane. */
svuint16_t montgomery_reduce_sve2(svuint32_t zn) {
    const svbool_t pg = svptrue_b32();
    const svuint32_t q = svdup_u32(IATO_MOD_Q);
    const svuint32_t qinv = svdup_u32(IATO_MONT_QINV);

    svuint32_t m = svmul_u32_x(pg, zn, qinv);
    m = svand_n_u32_x(pg, m, 0xFFFFu);

    svuint32_t t = svmul_u32_x(pg, m, q);
    t = svadd_u32_x(pg, t, zn);
    t = svlsr_n_u32_x(pg, t, 16);

    svuint32_t t_minus_q = svsub_u32_x(pg, t, q);
    svbool_t ge_q = svcmpge_u32(pg, t, q);
    svuint32_t reduced = svsel_u32(ge_q, t_minus_q, t);

    return svqxtnt_u16(svdup_u16(0), reduced);
}

/* NTT Furnace: branchless butterfly at each lane. */
iato_ntt_pair_t ntt_butterfly_sve2(svuint16_t a, svuint16_t b, svuint16_t twiddle) {
    const svbool_t pg = svptrue_b16();

    svuint32_t bw = svmul_u32_x(pg, svunpklo_u32(b), svunpklo_u32(twiddle));
    svuint16_t bw_red = montgomery_reduce_sve2(bw);

    iato_ntt_pair_t out;
    out.hi = iato_mod_q_add(a, bw_red);
    out.lo = iato_mod_q_sub(a, bw_red);
    return out;
}

/* LWE Accumulator: in-stride fold z_acc = z_acc + r_i * z_in (mod q). */
svuint16_t fold_accumulator(svuint16_t z_in, svuint16_t z_acc, uint16_t r_i) {
    const svbool_t pg = svptrue_b16();
    svuint32_t scaled = svmul_n_u32_x(pg, svunpklo_u32(z_in), r_i);
    svuint16_t scaled_red = montgomery_reduce_sve2(scaled);
    return iato_mod_q_add(z_acc, scaled_red);
}

/* LWE Accumulator: integer-point divergence to midpoint Gaussian seed. */
svuint16_t integer_divergence_track(svuint16_t z_acc, svuint16_t z_noise_seed) {
    const svbool_t pg = svptrue_b16();
    svuint16_t delta = iato_mod_q_sub(z_acc, z_noise_seed);
    svuint16_t neg_delta = iato_mod_q_sub(z_noise_seed, z_acc);
    svbool_t sign = svcmple_u16(pg, z_acc, z_noise_seed);
    svuint16_t abs_delta = svsel_u16(sign, neg_delta, delta);

    /* Squared divergence (mod q) keeps branchless deterministic timing. */
    svuint32_t sq = svmul_u32_x(pg, svunpklo_u32(abs_delta), svunpklo_u32(abs_delta));
    return montgomery_reduce_sve2(sq);
}

/* SNARK Wrapper: project folded result into BabyJubJub witness limbs. */
void snark_wrap_edwards(const iato_poly_zreg_t *z_acc, iato_witness_512_t *witness_out) {
    const svbool_t pg16 = svptrue_b16();

    svuint16_t limb0 = z_acc->v[0];
    svuint16_t limb1 = z_acc->v[1];
    svuint16_t limb2 = z_acc->v[2];
    svuint16_t limb3 = z_acc->v[3];

    for (size_t i = 4; i < IATO_POLY_ZREGS; ++i) {
        limb0 = iato_mod_q_add(limb0, z_acc->v[i]);
        limb1 = iato_mod_q_sub(limb1, z_acc->v[i]);
        limb2 = svxar_n_u16_z(pg16, limb2, z_acc->v[i], 3);
        limb3 = svxar_n_u16_z(pg16, limb3, z_acc->v[i], 7);
    }

    /* SNARK Wrapper stage: 4 limb packing for downstream BabyJubJub circuit IO. */
    witness_out->limb[0] = svreinterpret_u8(limb0);
    witness_out->limb[1] = svreinterpret_u8(limb1);
    witness_out->limb[2] = svreinterpret_u8(limb2);
    witness_out->limb[3] = svreinterpret_u8(limb3);
}

/* Exit Path: explicit register scrub to preserve zero-cache / side-channel silence. */
void scrub_pipeline_registers(void) {
    __asm__ volatile(
        "eor z0.d, z0.d, z0.d\n\t"
        "eor z1.d, z1.d, z1.d\n\t"
        "eor z2.d, z2.d, z2.d\n\t"
        "eor z3.d, z3.d, z3.d\n\t"
        "eor z4.d, z4.d, z4.d\n\t"
        "eor z5.d, z5.d, z5.d\n\t"
        "eor z6.d, z6.d, z6.d\n\t"
        "eor z7.d, z7.d, z7.d\n\t"
        "eor z8.d, z8.d, z8.d\n\t"
        "eor z9.d, z9.d, z9.d\n\t"
        "eor z10.d, z10.d, z10.d\n\t"
        "eor z11.d, z11.d, z11.d\n\t"
        "eor z12.d, z12.d, z12.d\n\t"
        "eor z13.d, z13.d, z13.d\n\t"
        "eor z14.d, z14.d, z14.d\n\t"
        "eor z15.d, z15.d, z15.d\n\t"
        "eor z16.d, z16.d, z16.d\n\t"
        "eor z17.d, z17.d, z17.d\n\t"
        "eor z18.d, z18.d, z18.d\n\t"
        "eor z19.d, z19.d, z19.d\n\t"
        "eor z20.d, z20.d, z20.d\n\t"
        "eor z21.d, z21.d, z21.d\n\t"
        "eor z22.d, z22.d, z22.d\n\t"
        "eor z23.d, z23.d, z23.d\n\t"
        "eor z24.d, z24.d, z24.d\n\t"
        "eor z25.d, z25.d, z25.d\n\t"
        "eor z26.d, z26.d, z26.d\n\t"
        "eor z27.d, z27.d, z27.d\n\t"
        "eor z28.d, z28.d, z28.d\n\t"
        "eor z29.d, z29.d, z29.d\n\t"
        "eor z30.d, z30.d, z30.d\n\t"
        "eor z31.d, z31.d, z31.d\n\t"
        :
        :
        : "memory");
}

/* Ingest -> NTT Furnace -> LWE Accumulator -> SNARK Wrapper -> Exit Path */
void run_recursive_fold_epoch(uint64_t telemetry_items,
                              const iato_poly_zreg_t *stream,
                              uint16_t challenge_r,
                              iato_witness_512_t *witness_out) {
    iato_poly_zreg_t z_acc = {0};
    const svuint16_t seed = svdup_u16(1663u); /* midpoint around q/2 for Gaussian seed */

    for (uint64_t item = 0; item < telemetry_items; ++item) {
        const iato_poly_zreg_t *z_in = &stream[item]; /* Ingest Gate: LDR Z0,[Xn] conceptual map */

        for (size_t r = 0; r < IATO_POLY_ZREGS; ++r) {
            /* NTT Furnace stage (placeholder twiddle from lane-local seed). */
            iato_ntt_pair_t bfly = ntt_butterfly_sve2(z_in->v[r], z_in->v[r], seed);

            /* LWE Accumulator stage. */
            z_acc.v[r] = fold_accumulator(bfly.hi, z_acc.v[r], challenge_r);

            /* Integer-point divergence tracking folded in-register. */
            svuint16_t div = integer_divergence_track(z_acc.v[r], seed);
            z_acc.v[r] = iato_mod_q_add(z_acc.v[r], div);
        }
    }

    /* Tail-call SNARK: witness only at epoch end. */
    snark_wrap_edwards(&z_acc, witness_out);

    /* Exit Path: register scrubbing. */
    scrub_pipeline_registers();
}

/* Reference main-loop shape for 10^8 telemetry items in one epoch. */
void run_recursive_fold_100m(const iato_poly_zreg_t *stream,
                             uint16_t challenge_r,
                             iato_witness_512_t *witness_out) {
    run_recursive_fold_epoch(100000000ULL, stream, challenge_r, witness_out);
}
