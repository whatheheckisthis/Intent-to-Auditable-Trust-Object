#!/usr/bin/env python3
import ctypes
import os
import signal
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from bcc import BPF


class AuditEvent(ctypes.Structure):
    _fields_ = [
        ("ts_ns", ctypes.c_ulonglong),
        ("src_ip", ctypes.c_uint),
        ("dst_ip", ctypes.c_uint),
        ("src_port", ctypes.c_ushort),
        ("dst_port", ctypes.c_ushort),
        ("proto", ctypes.c_ubyte),
        ("pkt_len", ctypes.c_ushort),
        ("delta_ns", ctypes.c_ulonglong),
        ("mean_ns", ctypes.c_ulonglong),
        ("variance_ns", ctypes.c_ulonglong),
        ("tvla_tscore", ctypes.c_uint),
        ("leak_flag", ctypes.c_ubyte),
    ]


METRICS = {
    "events_total": 0,
    "leaks_total": 0,
    "last_tscore": 0,
    "last_delta_ns": 0,
    "last_pkt_len": 0,
    "last_seen_unix": 0,
}
METRICS_LOCK = threading.Lock()


def _ip(n):
    return socket.inet_ntoa(struct.pack("I", n))


def handle_event(cpu, data, size):
    event = ctypes.cast(data, ctypes.POINTER(AuditEvent)).contents
    with METRICS_LOCK:
        METRICS["events_total"] += 1
        METRICS["leaks_total"] += int(event.leak_flag)
        METRICS["last_tscore"] = int(event.tvla_tscore)
        METRICS["last_delta_ns"] = int(event.delta_ns)
        METRICS["last_pkt_len"] = int(event.pkt_len)
        METRICS["last_seen_unix"] = int(time.time())

    print(
        "[audit] leak=%d src=%s:%d dst=%s:%d proto=%d len=%d delta_ns=%d t=%d"
        % (
            event.leak_flag,
            _ip(event.src_ip),
            socket.ntohs(event.src_port),
            _ip(event.dst_ip),
            socket.ntohs(event.dst_port),
            event.proto,
            event.pkt_len,
            event.delta_ns,
            event.tvla_tscore,
        ),
        flush=True,
    )


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        with METRICS_LOCK:
            body = "\n".join(
                [
                    "# HELP xdp_audit_events_total Total anomalous timing events emitted by XDP",
                    "# TYPE xdp_audit_events_total counter",
                    f"xdp_audit_events_total {METRICS['events_total']}",
                    "# HELP xdp_audit_leak_flags_total Total TVLA leak alerts generated",
                    "# TYPE xdp_audit_leak_flags_total counter",
                    f"xdp_audit_leak_flags_total {METRICS['leaks_total']}",
                    "# HELP xdp_audit_last_tscore Last computed TVLA |t|-score",
                    "# TYPE xdp_audit_last_tscore gauge",
                    f"xdp_audit_last_tscore {METRICS['last_tscore']}",
                    "# HELP xdp_audit_last_delta_ns Last packet timing delta in nanoseconds",
                    "# TYPE xdp_audit_last_delta_ns gauge",
                    f"xdp_audit_last_delta_ns {METRICS['last_delta_ns']}",
                    "# HELP xdp_audit_last_packet_len_bytes Last packet size considered in TVLA check",
                    "# TYPE xdp_audit_last_packet_len_bytes gauge",
                    f"xdp_audit_last_packet_len_bytes {METRICS['last_pkt_len']}",
                    "# HELP xdp_audit_last_seen_unix_seconds Last time an event was observed",
                    "# TYPE xdp_audit_last_seen_unix_seconds gauge",
                    f"xdp_audit_last_seen_unix_seconds {METRICS['last_seen_unix']}",
                ]
            ) + "\n"

        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def serve_metrics(listen_addr):
    host, port = listen_addr.split(":")
    server = HTTPServer((host, int(port)), MetricsHandler)
    server.serve_forever()


def main():
    iface = os.getenv("XDP_INTERFACE", "eth0")
    xdp_mode = os.getenv("XDP_MODE", "drv").lower()
    metrics_addr = os.getenv("METRICS_ADDR", "0.0.0.0:9400")

    flags = 0
    if xdp_mode == "hw":
        flags = BPF.XDP_FLAGS_HW_MODE
    elif xdp_mode == "skb":
        flags = BPF.XDP_FLAGS_SKB_MODE
    else:
        flags = BPF.XDP_FLAGS_DRV_MODE

    b = BPF(src_file="/opt/xdp/xdp_audit.c")
    fn = b.load_func("xdp_audit", BPF.XDP)
    b.attach_xdp(iface, fn, flags)
    print(f"[startup] XDP audit hook attached on {iface} with mode={xdp_mode}", flush=True)

    b["audit_events"].open_perf_buffer(handle_event)

    t = threading.Thread(target=serve_metrics, args=(metrics_addr,), daemon=True)
    t.start()

    running = True

    def shutdown_handler(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    while running:
        b.perf_buffer_poll(timeout=1000)

    b.remove_xdp(iface, flags)
    print("[shutdown] XDP audit hook detached", flush=True)


if __name__ == "__main__":
    main()
