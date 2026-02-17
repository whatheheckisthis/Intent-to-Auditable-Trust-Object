interface nic_bus_if;
    logic        clk;
    logic        reset_n;
    logic        bus_valid;
    logic [31:0] data_bus;

    modport monitor(
        input clk,
        input reset_n,
        input bus_valid,
        input data_bus
    );

    modport driver(
        output clk,
        output reset_n,
        output bus_valid,
        output data_bus
    );
endinterface

interface axi_lite_audit_if;
    logic        aclk;
    logic        aresetn;
    logic [31:0] araddr;
    logic        arvalid;
    logic        arready;
    logic [31:0] rdata;
    logic [1:0]  rresp;
    logic        rvalid;
    logic        rready;

    modport slave(
        input  aclk,
        input  aresetn,
        input  araddr,
        input  arvalid,
        input  rready,
        output arready,
        output rdata,
        output rresp,
        output rvalid
    );

    modport master(
        output aclk,
        output aresetn,
        output araddr,
        output arvalid,
        output rready,
        input  arready,
        input  rdata,
        input  rresp,
        input  rvalid
    );
endinterface

module nic_bus_audit_hook #(
    parameter int unsigned TOGGLE_THRESHOLD = 12,
    parameter int unsigned BURST_LIMIT      = 4,
    parameter logic [15:0] ALLOWED_ADDR_MIN = 16'h1000,
    parameter logic [15:0] ALLOWED_ADDR_MAX = 16'h10FF
) (
    input  logic        clk,
    input  logic        reset_n,
    input  logic        bus_valid,
    input  logic [31:0] data_bus,
    input  logic [31:0] s_axil_araddr,
    input  logic        s_axil_arvalid,
    input  logic        s_axil_rready,
    output logic        s_axil_arready,
    output logic [31:0] s_axil_rdata,
    output logic [1:0]  s_axil_rresp,
    output logic        s_axil_rvalid
);
    logic [31:0] last_data_bus;
    logic [31:0] toggled_bits;
    logic [5:0]  hamming_distance;
    logic [2:0]  toggle_burst_count;
    logic        suspicious_toggle;
    logic        unauthorized_access;

    reg [63:0] audit_event_counter;

    function automatic [5:0] popcount32(input logic [31:0] value);
        int idx;
        begin
            popcount32 = '0;
            for (idx = 0; idx < 32; idx++) begin
                popcount32 = popcount32 + value[idx];
            end
        end
    endfunction

    assign toggled_bits      = data_bus ^ last_data_bus;
    assign hamming_distance  = popcount32(toggled_bits);
    assign suspicious_toggle = bus_valid
                            && (hamming_distance >= TOGGLE_THRESHOLD)
                            && (toggle_burst_count >= (BURST_LIMIT - 1));

    assign unauthorized_access = bus_valid
                              && ((data_bus[31:16] < ALLOWED_ADDR_MIN)
                               || (data_bus[31:16] > ALLOWED_ADDR_MAX));

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            last_data_bus      <= '0;
            toggle_burst_count <= '0;
            audit_event_counter <= '0;
        end else begin
            if (bus_valid) begin
                if (hamming_distance >= TOGGLE_THRESHOLD) begin
                    if (toggle_burst_count < BURST_LIMIT) begin
                        toggle_burst_count <= toggle_burst_count + 1'b1;
                    end
                end else begin
                    toggle_burst_count <= '0;
                end

                if (suspicious_toggle || unauthorized_access) begin
                    audit_event_counter <= audit_event_counter + 64'd1;
                end

                last_data_bus <= data_bus;
            end
        end
    end

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            s_axil_arready <= 1'b0;
            s_axil_rvalid  <= 1'b0;
            s_axil_rresp   <= 2'b00;
            s_axil_rdata   <= 32'h0;
        end else begin
            if (!s_axil_rvalid) begin
                s_axil_arready <= 1'b1;
                if (s_axil_arvalid) begin
                    s_axil_arready <= 1'b0;
                    s_axil_rvalid  <= 1'b1;
                    s_axil_rresp   <= 2'b00;

                    unique case (s_axil_araddr[3:2])
                        2'b00: s_axil_rdata <= audit_event_counter[31:0];
                        2'b01: s_axil_rdata <= audit_event_counter[63:32];
                        default: begin
                            s_axil_rdata <= 32'hDEAD_BEEF;
                            s_axil_rresp <= 2'b10;
                        end
                    endcase
                end
            end else if (s_axil_rvalid && s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end
endmodule

module nic_bus_audit_hook_with_if #(
    parameter int unsigned TOGGLE_THRESHOLD = 12,
    parameter int unsigned BURST_LIMIT      = 4,
    parameter logic [15:0] ALLOWED_ADDR_MIN = 16'h1000,
    parameter logic [15:0] ALLOWED_ADDR_MAX = 16'h10FF
) (
    nic_bus_if.monitor nic,
    axi_lite_audit_if.slave axil
);
    nic_bus_audit_hook #(
        .TOGGLE_THRESHOLD(TOGGLE_THRESHOLD),
        .BURST_LIMIT(BURST_LIMIT),
        .ALLOWED_ADDR_MIN(ALLOWED_ADDR_MIN),
        .ALLOWED_ADDR_MAX(ALLOWED_ADDR_MAX)
    ) u_core (
        .clk(nic.clk),
        .reset_n(nic.reset_n),
        .bus_valid(nic.bus_valid),
        .data_bus(nic.data_bus),
        .s_axil_araddr(axil.araddr),
        .s_axil_arvalid(axil.arvalid),
        .s_axil_rready(axil.rready),
        .s_axil_arready(axil.arready),
        .s_axil_rdata(axil.rdata),
        .s_axil_rresp(axil.rresp),
        .s_axil_rvalid(axil.rvalid)
    );
endmodule
