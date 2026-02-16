`timescale 1ns/1ps

// Parallel NTT core for recursive zk-SNARK packet attestation.
// Tuned for Kyber-sized polynomials (N=256) and a 1200 ns fold target.
module montgomery_mul_pipe #(
    parameter int WIDTH = 16,
    parameter logic [WIDTH-1:0] MODULUS = 16'd3329,
    parameter logic [WIDTH-1:0] N_PRIME = 16'd62209
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  in_valid,
    input  logic [WIDTH-1:0]      a,
    input  logic [WIDTH-1:0]      b,
    output logic                  out_valid,
    output logic [WIDTH-1:0]      result
);
    logic [2*WIDTH-1:0] t_s0, t_s1, t_s2;
    logic [WIDTH-1:0]   m_s1;
    logic [WIDTH:0]     u_s3;
    logic [3:0]         valid_pipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            t_s0      <= '0;
            t_s1      <= '0;
            t_s2      <= '0;
            m_s1      <= '0;
            u_s3      <= '0;
            result    <= '0;
            valid_pipe<= '0;
            out_valid <= 1'b0;
        end else begin
            valid_pipe <= {valid_pipe[2:0], in_valid};

            if (in_valid) begin
                t_s0 <= a * b;
            end

            if (valid_pipe[0]) begin
                t_s1 <= t_s0;
                m_s1 <= (t_s0[WIDTH-1:0] * N_PRIME);
            end

            if (valid_pipe[1]) begin
                t_s2 <= t_s1 + (m_s1 * MODULUS);
            end

            if (valid_pipe[2]) begin
                u_s3 <= t_s2[2*WIDTH-1:WIDTH];
                if (t_s2[2*WIDTH-1:WIDTH] >= MODULUS) begin
                    result <= t_s2[2*WIDTH-1:WIDTH] - MODULUS;
                end else begin
                    result <= t_s2[2*WIDTH-1:WIDTH];
                end
            end

            out_valid <= valid_pipe[3];
        end
    end
endmodule

module parallel_ntt_core #(
    parameter int LANES = 16,
    parameter int LOGN = 8,               // N = 256 coefficients
    parameter int WIDTH = 16,
    parameter int PIPELINE_DEPTH = 4,
    parameter logic [WIDTH-1:0] MODULUS = 16'd3329,
    parameter int CLOCK_NS = 2,
    parameter int FOLD_TARGET_NS = 1200
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     coeff_valid,
    input  logic [WIDTH-1:0]         coeff_data,
    input  logic [$clog2(1<<LOGN)-1:0] coeff_addr,
    input  logic                     start_ntt,
    output logic                     ntt_done,
    output logic                     busy,
    output logic                     witness_valid,
    output logic [255:0]             witness_256
);
    localparam int N = (1 << LOGN);
    localparam int STAGES = LOGN;
    localparam int BUTTERFLIES_PER_STAGE = N/2;
    localparam int BATCHES_PER_STAGE = (BUTTERFLIES_PER_STAGE + LANES - 1) / LANES;
    localparam int EST_CYCLES = (STAGES * BATCHES_PER_STAGE) + PIPELINE_DEPTH + 12;
    localparam int EST_LATENCY_NS = EST_CYCLES * CLOCK_NS;

    typedef enum logic [2:0] { IDLE, LOAD, NTT_RUN, FOLD, EMIT_WITNESS } ntt_state_t;
    ntt_state_t state;

    logic [WIDTH-1:0] poly_mem [0:N-1];
    logic [WIDTH-1:0] twiddle_rom [0:N-1];

    logic [$clog2(STAGES)-1:0] stage_idx;
    logic [$clog2(N)-1:0]      butterfly_idx;

    // Z-register lane mapping: coefficient k is mapped into lane (k % LANES).
    logic lane_valid   [0:LANES-1];
    logic [WIDTH-1:0] z_lane_a [0:LANES-1];
    logic [WIDTH-1:0] z_lane_b [0:LANES-1];
    logic [WIDTH-1:0] z_lane_twiddle [0:LANES-1];
    logic [WIDTH-1:0] lane_mul [0:LANES-1];
    logic lane_mul_vld [0:LANES-1];

    logic [255:0] witness_acc;

    initial begin
        if (EST_LATENCY_NS > FOLD_TARGET_NS) begin
            $error("NTT fold latency estimate %0d ns exceeds %0d ns target", EST_LATENCY_NS, FOLD_TARGET_NS);
        end
    end

    genvar i;
    generate
        for (i = 0; i < LANES; i++) begin : G_MONT
            montgomery_mul_pipe #(
                .WIDTH(WIDTH),
                .MODULUS(MODULUS)
            ) u_mont (
                .clk      (clk),
                .rst_n    (rst_n),
                .in_valid (lane_valid[i]),
                .a        (z_lane_b[i]),
                .b        (z_lane_twiddle[i]),
                .out_valid(lane_mul_vld[i]),
                .result   (lane_mul[i])
            );
        end
    endgenerate

    integer t;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (t = 0; t < N; t++) begin
                twiddle_rom[t] <= t + 1;
            end
        end
    end

    integer l;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= IDLE;
            stage_idx        <= '0;
            butterfly_idx    <= '0;
            busy             <= 1'b0;
            ntt_done         <= 1'b0;
            witness_valid    <= 1'b0;
            witness_256      <= '0;
            witness_acc      <= 256'h6a09e667f3bcc908bb67ae8584caa73b;
            for (l = 0; l < LANES; l++) begin
                lane_valid[l]      <= 1'b0;
                z_lane_a[l]        <= '0;
                z_lane_b[l]        <= '0;
                z_lane_twiddle[l]  <= '0;
            end
        end else begin
            ntt_done      <= 1'b0;
            witness_valid <= 1'b0;

            if (coeff_valid) begin
                poly_mem[coeff_addr] <= coeff_data;
            end

            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    if (start_ntt) begin
                        busy <= 1'b1;
                        stage_idx <= '0;
                        butterfly_idx <= '0;
                        state <= LOAD;
                    end
                end

                LOAD: state <= NTT_RUN;

                NTT_RUN: begin
                    for (l = 0; l < LANES; l++) begin
                        int lane_base;
                        lane_base = butterfly_idx + l;
                        lane_valid[l] <= (lane_base < BUTTERFLIES_PER_STAGE);
                        z_lane_a[l] <= poly_mem[lane_base % N];
                        z_lane_b[l] <= poly_mem[(lane_base + (1 << stage_idx)) % N];
                        z_lane_twiddle[l] <= twiddle_rom[lane_base % N];
                    end

                    for (l = 0; l < LANES; l++) begin
                        int lane_base;
                        lane_base = butterfly_idx + l;
                        if (lane_mul_vld[l] && (lane_base < BUTTERFLIES_PER_STAGE)) begin
                            poly_mem[lane_base % N] <= (z_lane_a[l] + lane_mul[l]) % MODULUS;
                            poly_mem[(lane_base + (1 << stage_idx)) % N] <=
                                (z_lane_a[l] + MODULUS - lane_mul[l]) % MODULUS;
                        end
                    end

                    butterfly_idx <= butterfly_idx + LANES;
                    if (butterfly_idx >= BUTTERFLIES_PER_STAGE - LANES) begin
                        butterfly_idx <= '0;
                        stage_idx <= stage_idx + 1;
                        if (stage_idx == STAGES-1) begin
                            state <= FOLD;
                        end
                    end
                end

                FOLD: begin
                    witness_acc[63:0]    <= witness_acc[63:0]    ^ poly_mem[0];
                    witness_acc[127:64]  <= witness_acc[127:64]  ^ poly_mem[64];
                    witness_acc[191:128] <= witness_acc[191:128] ^ poly_mem[128];
                    witness_acc[255:192] <= witness_acc[255:192] ^ poly_mem[192];
                    state <= EMIT_WITNESS;
                end

                EMIT_WITNESS: begin
                    witness_256 <= witness_acc;
                    witness_valid <= 1'b1;
                    ntt_done <= 1'b1;
                    busy <= 1'b0;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
