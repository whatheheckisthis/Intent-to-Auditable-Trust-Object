`timescale 1ns/1ps

// Dual-clock bridge from clk_bus (100 MHz) to clk_core (400 MHz).
// Implements an asynchronous FIFO with Gray-coded pointers.
module ntt_clock_domain_bridge #(
    parameter int WIDTH = 16,
    parameter int DEPTH = 16,
    parameter int ADDR_W = $clog2(DEPTH)
) (
    input  logic               rst_n,
    input  logic               clk_bus,
    input  logic               clk_core,
    input  logic               bus_valid,
    input  logic [WIDTH-1:0]   bus_data,
    output logic               bus_ready,
    output logic               core_valid,
    output logic [WIDTH-1:0]   core_data,
    input  logic               core_ready
);
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    logic [ADDR_W:0] wr_ptr_bin, wr_ptr_bin_n;
    logic [ADDR_W:0] wr_ptr_gray, wr_ptr_gray_n;
    logic [ADDR_W:0] rd_ptr_bin, rd_ptr_bin_n;
    logic [ADDR_W:0] rd_ptr_gray, rd_ptr_gray_n;

    logic [ADDR_W:0] rd_ptr_gray_sync_bus_ff1, rd_ptr_gray_sync_bus_ff2;
    logic [ADDR_W:0] wr_ptr_gray_sync_core_ff1, wr_ptr_gray_sync_core_ff2;

    logic fifo_full;
    logic fifo_empty;

    function automatic logic [ADDR_W:0] bin2gray(input logic [ADDR_W:0] b);
        return (b >> 1) ^ b;
    endfunction

    always_comb begin
        wr_ptr_bin_n  = wr_ptr_bin + ((bus_valid && bus_ready) ? 1'b1 : 1'b0);
        wr_ptr_gray_n = bin2gray(wr_ptr_bin_n);
        rd_ptr_bin_n  = rd_ptr_bin + ((core_valid && core_ready) ? 1'b1 : 1'b0);
        rd_ptr_gray_n = bin2gray(rd_ptr_bin_n);

        fifo_empty = (rd_ptr_gray == wr_ptr_gray_sync_core_ff2);
        fifo_full  = (wr_ptr_gray_n == {~rd_ptr_gray_sync_bus_ff2[ADDR_W:ADDR_W-1], rd_ptr_gray_sync_bus_ff2[ADDR_W-2:0]});
    end

    assign bus_ready  = !fifo_full;
    assign core_valid = !fifo_empty;
    assign core_data  = mem[rd_ptr_bin[ADDR_W-1:0]];

    always_ff @(posedge clk_bus or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr_bin <= '0;
            wr_ptr_gray <= '0;
            rd_ptr_gray_sync_bus_ff1 <= '0;
            rd_ptr_gray_sync_bus_ff2 <= '0;
        end else begin
            rd_ptr_gray_sync_bus_ff1 <= rd_ptr_gray;
            rd_ptr_gray_sync_bus_ff2 <= rd_ptr_gray_sync_bus_ff1;

            if (bus_valid && bus_ready) begin
                mem[wr_ptr_bin[ADDR_W-1:0]] <= bus_data;
                wr_ptr_bin  <= wr_ptr_bin_n;
                wr_ptr_gray <= wr_ptr_gray_n;
            end
        end
    end

    always_ff @(posedge clk_core or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr_bin <= '0;
            rd_ptr_gray <= '0;
            wr_ptr_gray_sync_core_ff1 <= '0;
            wr_ptr_gray_sync_core_ff2 <= '0;
        end else begin
            wr_ptr_gray_sync_core_ff1 <= wr_ptr_gray;
            wr_ptr_gray_sync_core_ff2 <= wr_ptr_gray_sync_core_ff1;

            if (core_valid && core_ready) begin
                rd_ptr_bin  <= rd_ptr_bin_n;
                rd_ptr_gray <= rd_ptr_gray_n;
            end
        end
    end
endmodule
