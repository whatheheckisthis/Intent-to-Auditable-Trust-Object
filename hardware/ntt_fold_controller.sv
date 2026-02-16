`timescale 1ns/1ps

// Fixed-pattern controller for an 8-stage radix-2 NTT (N=256).
// The schedule is deterministic to reduce side-channel leakage from control flow.
module ntt_fold_controller #(
    parameter int N = 256,
    parameter int LOGN = 8,
    parameter int LANES = 2,
    parameter int ADDR_W = $clog2(N)
) (
    input  logic                clk_core,
    input  logic                rst_n,
    input  logic                start,
    output logic                busy,
    output logic                done,
    output logic                issue,
    output logic [LOGN-1:0]     stage,
    output logic [ADDR_W-1:0]   tw_addr_a,
    output logic [ADDR_W-1:0]   tw_addr_b,
    output logic [ADDR_W-1:0]   coeff_a_addr,
    output logic [ADDR_W-1:0]   coeff_b_addr
);
    localparam int HALF_N = N/2;

    typedef enum logic [1:0] {S_IDLE, S_STAGE_INIT, S_STAGE_RUN, S_DONE} ctrl_state_t;
    ctrl_state_t state;

    logic [ADDR_W-1:0] bf_idx;
    logic [ADDR_W-1:0] stride;

    always_comb begin
        stride = (1 << stage);

        // Deterministic addressing pattern:
        // two butterflies issued per cycle to feed two parallel butterfly units.
        coeff_a_addr = bf_idx;
        coeff_b_addr = bf_idx + stride;
        tw_addr_a    = bf_idx;
        tw_addr_b    = bf_idx + 1'b1;

        issue = (state == S_STAGE_RUN);
    end

    always_ff @(posedge clk_core or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            busy   <= 1'b0;
            done   <= 1'b0;
            stage  <= '0;
            bf_idx <= '0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy   <= 1'b0;
                    stage  <= '0;
                    bf_idx <= '0;
                    if (start) begin
                        busy  <= 1'b1;
                        state <= S_STAGE_INIT;
                    end
                end

                S_STAGE_INIT: begin
                    bf_idx <= '0;
                    state  <= S_STAGE_RUN;
                end

                S_STAGE_RUN: begin
                    bf_idx <= bf_idx + LANES;
                    if (bf_idx >= (HALF_N - LANES)) begin
                        bf_idx <= '0;
                        if (stage == LOGN-1) begin
                            state <= S_DONE;
                        end else begin
                            stage <= stage + 1'b1;
                            state <= S_STAGE_INIT;
                        end
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Safety check: twiddle addresses are always in [0, N-1] when issuing work.
    property p_twiddle_addr_bounds;
        @(posedge clk_core) disable iff(!rst_n)
        issue |-> ((tw_addr_a < N) && (tw_addr_b < N));
    endproperty

    assert property (p_twiddle_addr_bounds)
        else $fatal(1, "Twiddle address out of bounds during 8-stage fold");
endmodule
