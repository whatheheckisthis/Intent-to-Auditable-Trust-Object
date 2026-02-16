`timescale 1ns/1ps

// Gold-master RTL for a 256-point radix-2 NTT core over R_q = Z_q[x]/(x^256 + 1), q=3329.
// - Dual-butterfly issue per clk_core cycle.
// - Branchless Montgomery-ASR reduction for multiply path.
// - Explicit 256-lane Z-register array for coefficient storage.
module ntt_gold_master_core #(
    parameter int W = 16,
    parameter int N = 256,
    parameter int LOGN = 8,
    parameter logic signed [W-1:0] Q = 16'sd3329,
    parameter logic [W-1:0] QINV = 16'd62209
) (
    input  logic                clk_core,
    input  logic                rst_n,
    input  logic                start,
    input  logic                z_wr_en,
    input  logic [7:0]          z_wr_addr,
    input  logic [W-1:0]        z_wr_data,
    output logic                busy,
    output logic                done
);
    // 256 coefficient lanes (Z-register file).
    logic [255:0][15:0] z_lanes;

    typedef enum logic [1:0] {S_IDLE, S_RUN, S_DONE} state_t;
    state_t state;

    logic [2:0] stage;
    logic [6:0] butterfly_idx;

    logic [6:0] bf0;
    logic [6:0] bf1;
    logic       bf0_valid;
    logic       bf1_valid;

    logic [7:0] idx_a0, idx_b0;
    logic [7:0] idx_a1, idx_b1;
    logic [7:0] tw_idx0, tw_idx1;

    logic [W-1:0] tw0;
    logic [W-1:0] tw1;

    logic [W-1:0] a0, b0;
    logic [W-1:0] a1, b1;
    logic [W-1:0] t0, t1;
    logic [W-1:0] y0_add, y0_sub;
    logic [W-1:0] y1_add, y1_sub;

    // Branchless modulo add in [0, q).
    function automatic logic [W-1:0] add_mod_q(
        input logic [W-1:0] x,
        input logic [W-1:0] y
    );
        logic signed [W:0] sum;
        logic signed [W:0] sum_minus_q;
        logic signed [W:0] mask;
        begin
            sum         = $signed({1'b0, x}) + $signed({1'b0, y});
            sum_minus_q = sum - $signed({1'b0, Q});
            mask        = sum_minus_q >>> W;
            add_mod_q   = (sum_minus_q + ($signed({1'b0, Q}) & mask))[W-1:0];
        end
    endfunction

    // Branchless modulo subtract in [0, q).
    function automatic logic [W-1:0] sub_mod_q(
        input logic [W-1:0] x,
        input logic [W-1:0] y
    );
        logic signed [W:0] diff;
        logic signed [W:0] mask;
        begin
            diff      = $signed({1'b0, x}) - $signed({1'b0, y});
            mask      = diff >>> W;
            sub_mod_q = (diff + ($signed({1'b0, Q}) & mask))[W-1:0];
        end
    endfunction

    // Branchless Montgomery-ASR core:
    //   t = a*b
    //   u = (t[15:0] * qinv) mod 2^16
    //   r = (t - u*q) >>> 16
    //   r = r - q; r += (r<0 ? q : 0)   [implemented branchlessly via sign mask]
    function automatic logic [W-1:0] montgomery_asr_mul(
        input logic [W-1:0] x,
        input logic [W-1:0] w
    );
        logic signed [31:0] prod;
        logic [15:0]        u16;
        logic signed [31:0] uq;
        logic signed [31:0] red;
        logic signed [31:0] red_minus_q;
        logic signed [31:0] sign_mask;
        logic signed [31:0] red_norm;
        begin
            prod        = $signed({1'b0, x}) * $signed({1'b0, w});
            u16         = prod[15:0] * QINV;
            uq          = $signed({1'b0, u16}) * $signed({1'b0, Q});
            red         = (prod - uq) >>> 16;
            red_minus_q = red - $signed({1'b0, Q});
            sign_mask   = red_minus_q >>> 31;
            red_norm    = red_minus_q + ($signed({1'b0, Q}) & sign_mask);
            montgomery_asr_mul = red_norm[W-1:0];
        end
    endfunction

    // Explicit stage-by-stage shuffle logic for butterfly index -> lane pair and twiddle index.
    always_comb begin
        bf0 = butterfly_idx;
        bf1 = butterfly_idx + 7'd1;
        bf0_valid = (bf0 < 7'd128);
        bf1_valid = (bf1 < 7'd128);

        // defaults
        idx_a0 = 8'd0; idx_b0 = 8'd0; tw_idx0 = 8'd0;
        idx_a1 = 8'd0; idx_b1 = 8'd0; tw_idx1 = 8'd0;

        case (stage)
            3'd0: begin
                idx_a0 = {bf0, 1'b0};
                idx_b0 = {bf0, 1'b1};
                tw_idx0 = 8'd0;

                idx_a1 = {bf1, 1'b0};
                idx_b1 = {bf1, 1'b1};
                tw_idx1 = 8'd0;
            end
            3'd1: begin
                idx_a0 = ((bf0 >> 1) << 2) | (bf0 & 8'h01);
                idx_b0 = idx_a0 + 8'd2;
                tw_idx0 = (bf0 & 8'h01) << 7;

                idx_a1 = ((bf1 >> 1) << 2) | (bf1 & 8'h01);
                idx_b1 = idx_a1 + 8'd2;
                tw_idx1 = (bf1 & 8'h01) << 7;
            end
            3'd2: begin
                idx_a0 = ((bf0 >> 2) << 3) | (bf0 & 8'h03);
                idx_b0 = idx_a0 + 8'd4;
                tw_idx0 = (bf0 & 8'h03) << 6;

                idx_a1 = ((bf1 >> 2) << 3) | (bf1 & 8'h03);
                idx_b1 = idx_a1 + 8'd4;
                tw_idx1 = (bf1 & 8'h03) << 6;
            end
            3'd3: begin
                idx_a0 = ((bf0 >> 3) << 4) | (bf0 & 8'h07);
                idx_b0 = idx_a0 + 8'd8;
                tw_idx0 = (bf0 & 8'h07) << 5;

                idx_a1 = ((bf1 >> 3) << 4) | (bf1 & 8'h07);
                idx_b1 = idx_a1 + 8'd8;
                tw_idx1 = (bf1 & 8'h07) << 5;
            end
            3'd4: begin
                idx_a0 = ((bf0 >> 4) << 5) | (bf0 & 8'h0F);
                idx_b0 = idx_a0 + 8'd16;
                tw_idx0 = (bf0 & 8'h0F) << 4;

                idx_a1 = ((bf1 >> 4) << 5) | (bf1 & 8'h0F);
                idx_b1 = idx_a1 + 8'd16;
                tw_idx1 = (bf1 & 8'h0F) << 4;
            end
            3'd5: begin
                idx_a0 = ((bf0 >> 5) << 6) | (bf0 & 8'h1F);
                idx_b0 = idx_a0 + 8'd32;
                tw_idx0 = (bf0 & 8'h1F) << 3;

                idx_a1 = ((bf1 >> 5) << 6) | (bf1 & 8'h1F);
                idx_b1 = idx_a1 + 8'd32;
                tw_idx1 = (bf1 & 8'h1F) << 3;
            end
            3'd6: begin
                idx_a0 = ((bf0 >> 6) << 7) | (bf0 & 8'h3F);
                idx_b0 = idx_a0 + 8'd64;
                tw_idx0 = (bf0 & 8'h3F) << 2;

                idx_a1 = ((bf1 >> 6) << 7) | (bf1 & 8'h3F);
                idx_b1 = idx_a1 + 8'd64;
                tw_idx1 = (bf1 & 8'h3F) << 2;
            end
            default: begin // stage 7
                idx_a0 = bf0;
                idx_b0 = idx_a0 + 8'd128;
                tw_idx0 = (bf0 & 8'h7F) << 1;

                idx_a1 = bf1;
                idx_b1 = idx_a1 + 8'd128;
                tw_idx1 = (bf1 & 8'h7F) << 1;
            end
        endcase
    end

    ntt_twiddle_rom_x256 u_tw_rom (
        .addr_a (tw_idx0),
        .addr_b (tw_idx1),
        .dout_a (tw0),
        .dout_b (tw1)
    );

    always_comb begin
        a0 = z_lanes[idx_a0];
        b0 = z_lanes[idx_b0];
        a1 = z_lanes[idx_a1];
        b1 = z_lanes[idx_b1];

        t0 = montgomery_asr_mul(b0, tw0);
        t1 = montgomery_asr_mul(b1, tw1);

        y0_add = add_mod_q(a0, t0);
        y0_sub = sub_mod_q(a0, t0);
        y1_add = add_mod_q(a1, t1);
        y1_sub = sub_mod_q(a1, t1);
    end

    integer i;
    always_ff @(posedge clk_core or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < N; i++) begin
                z_lanes[i] <= '0;
            end
            state <= S_IDLE;
            stage <= '0;
            butterfly_idx <= '0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;

            if (z_wr_en) begin
                z_lanes[z_wr_addr] <= z_wr_data;
            end

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    stage <= 3'd0;
                    butterfly_idx <= 7'd0;
                    if (start) begin
                        busy <= 1'b1;
                        state <= S_RUN;
                    end
                end

                S_RUN: begin
                    if (bf0_valid) begin
                        z_lanes[idx_a0] <= y0_add;
                        z_lanes[idx_b0] <= y0_sub;
                    end
                    if (bf1_valid) begin
                        z_lanes[idx_a1] <= y1_add;
                        z_lanes[idx_b1] <= y1_sub;
                    end

                    butterfly_idx <= butterfly_idx + 7'd2;
                    if (butterfly_idx >= 7'd126) begin
                        butterfly_idx <= 7'd0;
                        if (stage == 3'd7) begin
                            state <= S_DONE;
                        end else begin
                            stage <= stage + 3'd1;
                        end
                    end
                end

                S_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Safety assertion: twiddle addresses are always in range when core is active.
    property p_twiddle_bounds;
        @(posedge clk_core) disable iff(!rst_n)
            (state == S_RUN) |-> ((tw_idx0 < N) && (tw_idx1 < N));
    endproperty

    assert property (p_twiddle_bounds)
        else $fatal(1, "Twiddle address out of range during NTT run");
endmodule
