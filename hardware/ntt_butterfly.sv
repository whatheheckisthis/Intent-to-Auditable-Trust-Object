`timescale 1ns/1ps

// Radix-2 NTT butterfly for Kyber ring arithmetic (mod q=3329).
// - Uses Montgomery reduction with arithmetic right shift (ASR) by 16.
// - Intended for in-stride lane processing (no RAM lookup side channel).
module ntt_butterfly #(
    parameter int W = 16,
    parameter logic signed [W-1:0] Q = 16'sd3329,
    parameter logic [W-1:0] QINV = 16'd62209  // q^{-1} mod 2^16 used by Kyber montgomery reduce
) (
    input  logic                      clk,
    input  logic                      rst_n,
    input  logic                      in_valid,
    input  logic signed [W-1:0]       coeff_a_i,
    input  logic signed [W-1:0]       coeff_b_i,
    input  logic signed [W-1:0]       twiddle_i,
    output logic                      out_valid,
    output logic signed [W-1:0]       coeff_a_o,
    output logic signed [W-1:0]       coeff_b_o
);
    logic signed [W-1:0] z_lane_a;
    logic signed [W-1:0] z_lane_b;
    logic signed [W-1:0] z_lane_tw;

    logic signed [2*W-1:0] mul_s1;
    logic signed [W-1:0]   a_s2;
    logic signed [W-1:0]   mul_red_s2;

    logic [2:0] valid_pipe;

    function automatic logic signed [W-1:0] reduce_q(input logic signed [W:0] x);
        logic signed [W:0] y;
        begin
            y = x;
            if (y >= Q)
                y = y - Q;
            if (y < 0)
                y = y + Q;
            reduce_q = y[W-1:0];
        end
    endfunction

    function automatic logic signed [W-1:0] montgomery_asr_reduce(input logic signed [2*W-1:0] t);
        logic signed [W-1:0] u16;
        logic signed [2*W-1:0] corr;
        logic signed [2*W-1:0] shifted;
        begin
            // Kyber-style Montgomery-ASR:
            //   u = (t * QINV) mod 2^16
            //   r = (t - u*q) >>> 16
            //   return centered modulo q
            u16 = t[W-1:0] * $signed(QINV);
            corr = t - (u16 * Q);
            shifted = corr >>> W;
            montgomery_asr_reduce = reduce_q(shifted[W:0]);
        end
    endfunction

    function automatic logic signed [W-1:0] add_mod_q(
        input logic signed [W-1:0] x,
        input logic signed [W-1:0] y
    );
        begin
            add_mod_q = reduce_q({x[W-1], x} + {y[W-1], y});
        end
    endfunction

    function automatic logic signed [W-1:0] sub_mod_q(
        input logic signed [W-1:0] x,
        input logic signed [W-1:0] y
    );
        begin
            sub_mod_q = reduce_q({x[W-1], x} - {y[W-1], y});
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            z_lane_a   <= '0;
            z_lane_b   <= '0;
            z_lane_tw  <= '0;
            mul_s1     <= '0;
            a_s2       <= '0;
            mul_red_s2 <= '0;
            coeff_a_o  <= '0;
            coeff_b_o  <= '0;
            valid_pipe <= '0;
            out_valid  <= 1'b0;
        end else begin
            valid_pipe <= {valid_pipe[1:0], in_valid};
            out_valid  <= valid_pipe[2];

            if (in_valid) begin
                // Z-register lane load logic for streaming coefficients.
                z_lane_a  <= reduce_q(coeff_a_i);
                z_lane_b  <= reduce_q(coeff_b_i);
                z_lane_tw <= reduce_q(twiddle_i);
            end

            // Stage 1: multiply b*twiddle.
            mul_s1 <= z_lane_b * z_lane_tw;

            // Stage 2: Montgomery-ASR reduction modulo q=3329.
            a_s2       <= z_lane_a;
            mul_red_s2 <= montgomery_asr_reduce(mul_s1);

            // Stage 3: radix-2 butterfly outputs.
            coeff_a_o <= add_mod_q(a_s2, mul_red_s2);
            coeff_b_o <= sub_mod_q(a_s2, mul_red_s2);
        end
    end

endmodule
