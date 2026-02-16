`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// SV2 cycle-accurate radix-2 NTT butterfly unit for the Kyber ring:
//   R_q = Z_q[x] / (x^256 + 1), q = 3329
//
// Design goals:
// - 100 MSPS throughput target through a 3-stage pipeline.
// - SIMD-style Z-register file with 256x16-bit coefficients.
// - Two 16-bit coefficients loaded per clock via MMIO data beat.
// - Branchless Montgomery-ASR reduction (no data-dependent if/else/?:).
// - Side-channel silent data path (constant-latency arithmetic flow).
// -----------------------------------------------------------------------------
module ntt_butterfly #(
    parameter int unsigned W               = 16,
    parameter int unsigned N_COEFF         = 256,
    parameter int unsigned MOD_Q           = 3329,
    parameter int unsigned QINV            = 62209,      // -q^{-1} mod 2^16
    parameter int unsigned CLK_FREQ_HZ     = 250_000_000,
    parameter int unsigned FULL_NTT_CYCLES = 256,
    parameter int unsigned MMIO_ADDR_DATA  = 8'h00,
    parameter int unsigned MMIO_ADDR_CTRL  = 8'h04,
    parameter int unsigned MMIO_ADDR_STAT  = 8'h08
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Control
    input  logic                         bf_valid,
    input  logic [7:0]                   left_idx_i,
    input  logic [7:0]                   right_idx_i,
    input  logic [W-1:0]                 twiddle_i,
    output logic                         bf_ready,

    // MMIO stream for loading Z-register lanes and telemetry extraction
    input  logic                         mmio_we,
    input  logic [7:0]                   mmio_addr,
    input  logic [31:0]                  mmio_wdata,
    input  logic                         mmio_re,
    output logic [31:0]                  mmio_rdata,
    output logic                         mmio_rvalid,

    // Butterfly outputs
    output logic                         out_valid,
    output logic [7:0]                   out_left_idx,
    output logic [7:0]                   out_right_idx,
    output logic [W-1:0]                 coeff_left_o,
    output logic [W-1:0]                 coeff_right_o
);
    localparam int unsigned TWO_Q = 2 * MOD_Q;
    localparam int unsigned FULL_NTT_NS = (FULL_NTT_CYCLES * 1_000_000_000) / CLK_FREQ_HZ;

    // 256-lane SIMD style Z-register bank.
    logic [N_COEFF-1:0][W-1:0] z_reg;

    // MMIO state
    logic [7:0] z_wr_ptr;
    logic       z_stream_en;

    // ------------------------------
    // Branchless helper arithmetic
    // ------------------------------
    function automatic logic [W-1:0] ct_reduce_0_2q(input logic [W:0] x);
        logic [W:0] x_minus_q;
        logic       ge_q;
        logic [W-1:0] add_back;
        begin
            x_minus_q = x - MOD_Q;
            ge_q      = ~x_minus_q[W];
            add_back  = {W{~ge_q}} & MOD_Q[W-1:0];
            ct_reduce_0_2q = x_minus_q[W-1:0] + add_back;
        end
    endfunction

    function automatic logic [W-1:0] montgomery_asr_reduce(input logic [2*W-1:0] t);
        logic [W-1:0] m;
        logic [2*W:0] u_wide;
        logic [W:0]   u0;
        logic [W-1:0] r0;
        logic [W:0]   r1_pre;
        logic [W-1:0] r1;
        begin
            // m = (t * QINV) mod 2^W
            m      = (t[W-1:0] * QINV[W-1:0]);
            // u = (t + m*q) >>> W  (arithmetic shift right behavior in fixed pipeline)
            u_wide = {1'b0, t} + (m * MOD_Q);
            u0     = u_wide[2*W:W];

            // Bring into [0, q) with branchless subtract-and-mask steps.
            r0     = ct_reduce_0_2q(u0);
            r1_pre = {1'b0, r0} - MOD_Q;
            r1     = r1_pre[W-1:0] + ({W{r1_pre[W]}} & MOD_Q[W-1:0]);
            montgomery_asr_reduce = r1;
        end
    endfunction

    // ------------------------------
    // Pipeline registers
    // Stage 1: multiply right coeff by twiddle
    // Stage 2: add/sub pre-reduction
    // Stage 3: reduced outputs committed to z_reg
    // ------------------------------
    logic                 s1_valid;
    logic [7:0]           s1_left_idx;
    logic [7:0]           s1_right_idx;
    logic [W-1:0]         s1_a;
    logic [2*W-1:0]       s1_mul;

    logic                 s2_valid;
    logic [7:0]           s2_left_idx;
    logic [7:0]           s2_right_idx;
    logic [W:0]           s2_sum_pre;
    logic [W:0]           s2_dif_pre;

    logic                 s3_valid;
    logic [7:0]           s3_left_idx;
    logic [7:0]           s3_right_idx;
    logic [W-1:0]         s3_sum_red;
    logic [W-1:0]         s3_dif_red;

    // Throughput/latency intent check for integration-time visibility.
    initial begin
        if (FULL_NTT_NS >= 1200) begin
            $error("FULL_NTT_NS=%0d violates <1200 ns target; raise clock or parallelism", FULL_NTT_NS);
        end
    end

    assign bf_ready = 1'b1;

    // MMIO readback exposes pointer/status and latest output word.
    always_comb begin
        mmio_rdata  = 32'h0;
        mmio_rvalid = mmio_re;
        unique case (mmio_addr)
            MMIO_ADDR_CTRL: mmio_rdata = {23'h0, z_stream_en, z_wr_ptr};
            MMIO_ADDR_STAT: mmio_rdata = {15'h0, s3_valid, s2_valid, s1_valid, out_valid,
                                          out_right_idx, out_left_idx};
            default:        mmio_rdata = {coeff_right_o, coeff_left_o};
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            z_wr_ptr      <= '0;
            z_stream_en   <= 1'b0;

            s1_valid      <= 1'b0;
            s1_left_idx   <= '0;
            s1_right_idx  <= '0;
            s1_a          <= '0;
            s1_mul        <= '0;

            s2_valid      <= 1'b0;
            s2_left_idx   <= '0;
            s2_right_idx  <= '0;
            s2_sum_pre    <= '0;
            s2_dif_pre    <= '0;

            s3_valid      <= 1'b0;
            s3_left_idx   <= '0;
            s3_right_idx  <= '0;
            s3_sum_red    <= '0;
            s3_dif_red    <= '0;

            out_valid     <= 1'b0;
            out_left_idx  <= '0;
            out_right_idx <= '0;
            coeff_left_o  <= '0;
            coeff_right_o <= '0;
        end else begin
            // MMIO control register write
            if (mmio_we && (mmio_addr == MMIO_ADDR_CTRL)) begin
                z_stream_en <= mmio_wdata[8];
                z_wr_ptr    <= mmio_wdata[7:0];
            end

            // Two 16-bit coefficients per cycle into Z-lanes.
            if (mmio_we && (mmio_addr == MMIO_ADDR_DATA) && z_stream_en) begin
                z_reg[z_wr_ptr]         <= mmio_wdata[15:0];
                z_reg[z_wr_ptr + 8'd1]  <= mmio_wdata[31:16];
                z_wr_ptr                <= z_wr_ptr + 8'd2;
            end

            // Stage 1: coefficient fetch + multiply
            s1_valid     <= bf_valid;
            s1_left_idx  <= left_idx_i;
            s1_right_idx <= right_idx_i;
            s1_a         <= z_reg[left_idx_i];
            s1_mul       <= z_reg[right_idx_i] * twiddle_i;

            // Stage 2: add/sub pre-reduction
            s2_valid     <= s1_valid;
            s2_left_idx  <= s1_left_idx;
            s2_right_idx <= s1_right_idx;
            s2_sum_pre   <= {1'b0, s1_a} + {1'b0, montgomery_asr_reduce(s1_mul)};
            s2_dif_pre   <= {1'b0, s1_a} + MOD_Q - {1'b0, montgomery_asr_reduce(s1_mul)};

            // Stage 3: integrated branchless reduction and writeback
            s3_valid      <= s2_valid;
            s3_left_idx   <= s2_left_idx;
            s3_right_idx  <= s2_right_idx;
            s3_sum_red    <= ct_reduce_0_2q(s2_sum_pre);
            s3_dif_red    <= ct_reduce_0_2q(s2_dif_pre);

            if (s3_valid) begin
                z_reg[s3_left_idx]  <= s3_sum_red;
                z_reg[s3_right_idx] <= s3_dif_red;
            end

            out_valid     <= s3_valid;
            out_left_idx  <= s3_left_idx;
            out_right_idx <= s3_right_idx;
            coeff_left_o  <= s3_sum_red;
            coeff_right_o <= s3_dif_red;
        end
    end

`ifdef FORMAL
    // Montgomery and butterfly output range safety in every valid stage.
    property p_s3_sum_lt_q;
        @(posedge clk) disable iff (!rst_n) s3_valid |-> (s3_sum_red < MOD_Q);
    endproperty
    property p_s3_dif_lt_q;
        @(posedge clk) disable iff (!rst_n) s3_valid |-> (s3_dif_red < MOD_Q);
    endproperty
    property p_out_lt_q;
        @(posedge clk) disable iff (!rst_n) out_valid |-> ((coeff_left_o < MOD_Q) && (coeff_right_o < MOD_Q));
    endproperty

    assert property (p_s3_sum_lt_q);
    assert property (p_s3_dif_lt_q);
    assert property (p_out_lt_q);
`endif

endmodule
