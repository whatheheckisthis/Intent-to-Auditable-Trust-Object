`timescale 1ns/1ps

// Parallel NTT core for zk-SNARK proving backends (Groth16/Plonky2).
// Architecture goals:
//  - Sustain 100 MSPS packet attest stream by decoupling ingest, NTT, and proof aggregation.
//  - Maintain high modular multiply throughput via deeply pipelined Montgomery units.
//  - Provide a 256-bit epoch witness output for recursive prover ingestion.

module montgomery_mul_pipe #(
    parameter int WIDTH = 64,
    parameter logic [WIDTH-1:0] MODULUS = 64'hffffffff00000001,
    parameter logic [WIDTH-1:0] N_PRIME = 64'h00000000ffffffff
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  in_valid,
    input  logic [WIDTH-1:0]      a,
    input  logic [WIDTH-1:0]      b,
    output logic                  out_valid,
    output logic [WIDTH-1:0]      result
);
    // 4-stage Montgomery pipeline:
    // S0: partial multiply
    // S1: reduction factor m = (t * n') mod R
    // S2: t + mN
    // S3: divide by R and conditional subtract

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
    parameter int LANES              = 8,
    parameter int LOGN               = 12,            // N=4096 default
    parameter int WIDTH              = 64,
    parameter int PIPELINE_DEPTH     = 4,
    parameter int EPOCH_PACKET_PROOFS= 1024,
    parameter logic [WIDTH-1:0] MODULUS = 64'hffffffff00000001
) (
    input  logic                     clk,
    input  logic                     rst_n,

    // Ingress interface for polynomial coefficients from packet prover frontend.
    input  logic                     coeff_valid,
    input  logic [WIDTH-1:0]         coeff_data,
    input  logic [$clog2(1<<LOGN)-1:0] coeff_addr,

    // Control/status
    input  logic                     start_ntt,
    output logic                     ntt_done,
    output logic                     busy,

    // Recursive witness output per epoch
    output logic                     witness_valid,
    output logic [255:0]             witness_256
);
    localparam int N = (1 << LOGN);
    localparam int STAGES = LOGN;

    typedef enum logic [2:0] {
        IDLE,
        LOAD,
        NTT_RUN,
        MERKLE_COMPRESS,
        EMIT_WITNESS
    } ntt_state_t;

    ntt_state_t state;

    logic [WIDTH-1:0] poly_mem [0:N-1];
    logic [WIDTH-1:0] twiddle_rom [0:N-1];

    logic [$clog2(STAGES)-1:0] stage_idx;
    logic [$clog2(N)-1:0]      butterfly_idx;
    logic                      lane_valid   [0:LANES-1];
    logic [WIDTH-1:0]          lane_a       [0:LANES-1];
    logic [WIDTH-1:0]          lane_b       [0:LANES-1];
    logic [WIDTH-1:0]          lane_twiddle [0:LANES-1];
    logic [WIDTH-1:0]          lane_mul     [0:LANES-1];
    logic                      lane_mul_vld [0:LANES-1];

    logic [31:0] packet_proof_count;
    logic [255:0] witness_acc;

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
                .a        (lane_b[i]),
                .b        (lane_twiddle[i]),
                .out_valid(lane_mul_vld[i]),
                .result   (lane_mul[i])
            );
        end
    endgenerate

    // Simple placeholder twiddle init (real design loads from precomputed ROM/BRAM).
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
            packet_proof_count <= '0;
            busy             <= 1'b0;
            ntt_done         <= 1'b0;
            witness_valid    <= 1'b0;
            witness_256      <= '0;
            witness_acc      <= 256'h6a09e667f3bcc908bb67ae8584caa73b;
            for (l = 0; l < LANES; l++) begin
                lane_valid[l]   <= 1'b0;
                lane_a[l]       <= '0;
                lane_b[l]       <= '0;
                lane_twiddle[l] <= '0;
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
                        busy          <= 1'b1;
                        stage_idx     <= '0;
                        butterfly_idx <= '0;
                        state         <= LOAD;
                    end
                end

                LOAD: begin
                    packet_proof_count <= packet_proof_count + 1;
                    state <= NTT_RUN;
                end

                NTT_RUN: begin
                    // Lane scheduler: each lane performs one butterfly multiplication stream.
                    for (l = 0; l < LANES; l++) begin
                        lane_valid[l]   <= 1'b1;
                        lane_a[l]       <= poly_mem[(butterfly_idx + l) % N];
                        lane_b[l]       <= poly_mem[(butterfly_idx + l + (1 << stage_idx)) % N];
                        lane_twiddle[l] <= twiddle_rom[(butterfly_idx + l) % N];
                    end

                    // Commit butterfly outputs when multiplier results are valid.
                    for (l = 0; l < LANES; l++) begin
                        if (lane_mul_vld[l]) begin
                            poly_mem[(butterfly_idx + l) % N] <= (lane_a[l] + lane_mul[l]) % MODULUS;
                            poly_mem[(butterfly_idx + l + (1 << stage_idx)) % N] <=
                                (lane_a[l] + MODULUS - lane_mul[l]) % MODULUS;
                        end
                    end

                    butterfly_idx <= butterfly_idx + LANES;
                    if (butterfly_idx >= N - LANES) begin
                        butterfly_idx <= '0;
                        stage_idx <= stage_idx + 1;
                        if (stage_idx == STAGES-1) begin
                            state <= MERKLE_COMPRESS;
                        end
                    end
                end

                MERKLE_COMPRESS: begin
                    // Epoch compression into 256-bit witness digest (placeholder sponge schedule).
                    witness_acc[63:0]    <= witness_acc[63:0]    ^ poly_mem[0];
                    witness_acc[127:64]  <= witness_acc[127:64]  ^ poly_mem[N/4];
                    witness_acc[191:128] <= witness_acc[191:128] ^ poly_mem[N/2];
                    witness_acc[255:192] <= witness_acc[255:192] ^ poly_mem[3*N/4];
                    state <= EMIT_WITNESS;
                end

                EMIT_WITNESS: begin
                    witness_256   <= witness_acc;
                    witness_valid <= 1'b1;
                    ntt_done      <= 1'b1;
                    busy          <= 1'b0;
                    state         <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
