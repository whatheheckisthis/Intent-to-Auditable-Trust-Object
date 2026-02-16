module ntt_butterfly #(
    parameter int W = 32,
    parameter logic [W-1:0] Q = 32'd2013265921,
    parameter logic [W-1:0] Q_INV_NEG = 32'd2013265919
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         in_valid,
    input  logic [W-1:0] coeff_a_i,
    input  logic [W-1:0] coeff_b_i,
    input  logic [W-1:0] twiddle_i,
    output logic         out_valid,
    output logic [W-1:0] coeff_a_o,
    output logic [W-1:0] coeff_b_o
);

    logic [W-1:0] z_lane_a;
    logic [W-1:0] z_lane_b;
    logic [W-1:0] z_lane_tw;

    logic [2*W-1:0] mul_s1;
    logic [W-1:0]   a_s2;
    logic [W-1:0]   mul_red_s2;

    logic [2:0] valid_pipe;

    function automatic logic [W-1:0] montgomery_asr_reduce(input logic [2*W-1:0] t);
        logic [W-1:0] m;
        logic [2*W:0] u_full;
        logic [W:0]   u_shift;
        logic [W:0]   u_sub;
        begin
            m = t[W-1:0] * Q_INV_NEG;
            u_full = t + (m * Q);
            u_shift = u_full[2*W:W];
            u_sub = u_shift - {1'b0, Q};
            montgomery_asr_reduce = u_sub[W] ? u_shift[W-1:0] : u_sub[W-1:0];
        end
    endfunction

    function automatic logic [W-1:0] add_mod_q(input logic [W-1:0] x, input logic [W-1:0] y);
        logic [W:0] s;
        begin
            s = {1'b0, x} + {1'b0, y};
            add_mod_q = (s >= {1'b0, Q}) ? (s - {1'b0, Q})[W-1:0] : s[W-1:0];
        end
    endfunction

    function automatic logic [W-1:0] sub_mod_q(input logic [W-1:0] x, input logic [W-1:0] y);
        begin
            sub_mod_q = (x >= y) ? (x - y) : (x + Q - y);
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
                z_lane_a  <= coeff_a_i;
                z_lane_b  <= coeff_b_i;
                z_lane_tw <= twiddle_i;
            end

            // Stage 1: multiply b*twiddle.
            mul_s1 <= z_lane_b * z_lane_tw;

            // Stage 2: Montgomery-ASR reduction.
            a_s2       <= z_lane_a;
            mul_red_s2 <= montgomery_asr_reduce(mul_s1);

            // Stage 3: radix-2 butterfly outputs.
            coeff_a_o <= add_mod_q(a_s2, mul_red_s2);
            coeff_b_o <= sub_mod_q(a_s2, mul_red_s2);
        end
    end

endmodule
