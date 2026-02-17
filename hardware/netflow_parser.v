module netflow_parser (
    input  logic         clk,
    input  logic         rst_n,
    input  logic [511:0] s_axis_tdata,
    input  logic [63:0]  s_axis_tkeep,
    input  logic         s_axis_tvalid,
    input  logic         s_axis_tlast,
    output logic         s_axis_tready,
    output logic         tuple_valid,
    output logic [127:0] src_ip,
    output logic [127:0] dst_ip,
    output logic [15:0]  src_port,
    output logic [15:0]  dst_port,
    output logic [7:0]   l4_proto,
    output logic [7:0]   ip_tos,
    output logic [7:0]   tcp_flags,
    output logic [63:0]  hw_ts_ns,
    output logic [31:0]  parser_meta
);

// 100Gbps AXI-stream parser targeting one tuple extraction per cycle.
// Exposes a V7 tuple: src_ip, dst_ip, src_port, dst_port, proto, ToS, TCP flags.
// hw_ts_ns is expected to be stamped by the upstream NIC pipeline and packed into
// the high dword lane of beat-0 metadata (bits [511:448]).

localparam logic [15:0] ETHERTYPE_IPV4 = 16'h0800;
localparam logic [15:0] ETHERTYPE_IPV6 = 16'h86DD;

logic [15:0] eth_type;
logic [3:0]  ipv4_ihl;
logic [7:0]  ipv4_proto;
logic [7:0]  ipv6_next_hdr;
logic [7:0]  tos_q;
logic [7:0]  flags_q;
logic [127:0] src_ip_q;
logic [127:0] dst_ip_q;
logic [15:0] src_port_q;
logic [15:0] dst_port_q;
logic tuple_hit;

assign s_axis_tready = 1'b1;

always_comb begin
    eth_type     = s_axis_tdata[111:96];
    ipv4_ihl     = s_axis_tdata[115:112];
    ipv4_proto   = s_axis_tdata[191:184];
    ipv6_next_hdr= s_axis_tdata[167:160];
    tuple_hit    = 1'b0;

    src_ip_q     = '0;
    dst_ip_q     = '0;
    src_port_q   = '0;
    dst_port_q   = '0;
    tos_q        = '0;
    flags_q      = '0;

    if (s_axis_tvalid && s_axis_tkeep[53:0] != 64'b0) begin
        if (eth_type == ETHERTYPE_IPV4) begin
            tuple_hit  = (ipv4_proto == 8'd6) || (ipv4_proto == 8'd17);
            src_ip_q[31:0] = s_axis_tdata[239:208];
            dst_ip_q[31:0] = s_axis_tdata[271:240];
            tos_q = s_axis_tdata[127:120];

            // Parse source/destination ports for common IHL=5 fast path.
            // Variable IHL fallback is bounded to first 64B beat.
            if (ipv4_ihl == 4'd5) begin
                src_port_q = s_axis_tdata[287:272];
                dst_port_q = s_axis_tdata[303:288];
                if (ipv4_proto == 8'd6) begin
                    flags_q = s_axis_tdata[391:384];
                end
            end
        end else if (eth_type == ETHERTYPE_IPV6) begin
            tuple_hit  = (ipv6_next_hdr == 8'd6) || (ipv6_next_hdr == 8'd17);
            src_ip_q   = s_axis_tdata[319:192];
            dst_ip_q   = s_axis_tdata[447:320];
            src_port_q = s_axis_tdata[463:448];
            dst_port_q = s_axis_tdata[479:464];
            tos_q      = {s_axis_tdata[123:120], s_axis_tdata[127:124]};
            if (ipv6_next_hdr == 8'd6) begin
                flags_q = s_axis_tdata[503:496];
            end
        end
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        tuple_valid <= 1'b0;
        src_ip      <= '0;
        dst_ip      <= '0;
        src_port    <= '0;
        dst_port    <= '0;
        l4_proto    <= '0;
        ip_tos      <= '0;
        tcp_flags   <= '0;
        hw_ts_ns    <= '0;
        parser_meta <= '0;
    end else begin
        tuple_valid <= tuple_hit;
        src_ip      <= src_ip_q;
        dst_ip      <= dst_ip_q;
        src_port    <= src_port_q;
        dst_port    <= dst_port_q;
        l4_proto    <= (eth_type == ETHERTYPE_IPV4) ? ipv4_proto : ipv6_next_hdr;
        ip_tos      <= tos_q;
        tcp_flags   <= flags_q;
        hw_ts_ns    <= s_axis_tdata[511:448];
        parser_meta <= {8'(eth_type == ETHERTYPE_IPV6), 7'd0, s_axis_tlast, s_axis_tkeep[15:0]};
    end
end

endmodule
