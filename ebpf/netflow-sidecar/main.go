package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	bpf "github.com/aquasecurity/libbpfgo"
)

type flowKey struct {
	SrcIP   uint32
	DstIP   uint32
	SrcPort uint16
	DstPort uint16
	Proto   uint8
	Pad     [3]byte
}

type flowMetrics struct {
	Packets    uint64
	Bytes      uint64
	LastSeenNS uint64
	FirstSeen  uint64
}

type auditEvent struct {
	TSNS          uint64
	WindowPackets uint64
	PktDeltaNS    uint64
	SrcIP         uint32
	DstIP         uint32
	SrcPort       uint16
	DstPort       uint16
	Proto         uint8
	Reason        uint8
}

type flowSnapshot struct {
	Key   flowKey
	Value flowMetrics
}

type collectorState struct {
	mu           sync.RWMutex
	activeFlows  int
	totalPackets uint64
	totalBytes   uint64
	lastPollUnix int64
	topFlows     []flowSnapshot
}

func main() {
	iface := getenv("XDP_INTERFACE", "eth0")
	objPath := getenv("BPF_OBJECT", "/opt/netflow/xdp_audit.bpf.o")
	metricsAddr := getenv("METRICS_ADDR", "0.0.0.0:9400")
	topK := atoiOr(getenv("NETFLOW_TOPK", "50"), 50)
	pollEvery := durationOr(getenv("NETFLOW_POLL_INTERVAL", "10s"), 10*time.Second)

	module, err := bpf.NewModuleFromFile(objPath)
	if err != nil {
		log.Fatalf("open BPF object: %v", err)
	}
	defer module.Close()

	if err := module.BPFLoadObject(); err != nil {
		log.Fatalf("load BPF object: %v", err)
	}

	prog, err := module.GetProgram("netflow_filter")
	if err != nil {
		log.Fatalf("lookup program: %v", err)
	}

	link, err := prog.AttachXDP(iface)
	if err != nil {
		log.Fatalf("attach XDP to %s: %v", iface, err)
	}
	defer link.Destroy()
	log.Printf("netflow_filter attached via libbpf on %s", iface)

	state := &collectorState{}

	flowMap, err := module.GetMap("flow_map")
	if err != nil {
		log.Fatalf("lookup flow_map: %v", err)
	}

	events := make(chan []byte, 1024)
	rb, err := module.InitRingBuf("events", events)
	if err != nil {
		log.Fatalf("init ringbuf: %v", err)
	}
	defer rb.Stop()
	rb.Start()

	go consumeAuditEvents(events)
	go serveMetrics(metricsAddr, state)

	ticker := time.NewTicker(pollEvery)
	defer ticker.Stop()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)

	for {
		select {
		case <-ticker.C:
			if err := snapshotFlows(flowMap, topK, state); err != nil {
				log.Printf("flow map snapshot failed: %v", err)
			}
		case s := <-sig:
			log.Printf("received %s, shutting down", s)
			return
		}
	}
}

func consumeAuditEvents(events <-chan []byte) {
	for raw := range events {
		if len(raw) < 40 {
			continue
		}
		var evt auditEvent
		if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, &evt); err != nil {
			continue
		}
		log.Printf("Trust Violation reason=%d src=%s:%d dst=%s:%d proto=%d delta_ns=%d window_pkts=%d ts_ns=%d",
			evt.Reason,
			ipString(evt.SrcIP), ntohs(evt.SrcPort),
			ipString(evt.DstIP), ntohs(evt.DstPort),
			evt.Proto,
			evt.PktDeltaNS,
			evt.WindowPackets,
			evt.TSNS,
		)
	}
}

func snapshotFlows(flowMap *bpf.BPFMap, topK int, st *collectorState) error {
	iter := flowMap.Iterator()
	flows := []flowSnapshot{}
	var totalPkts, totalBytes uint64

	for iter.Next() {
		key := iter.Key()
		if len(key) < 16 {
			continue
		}
		perCPU, err := flowMap.GetValue(key)
		if err != nil {
			continue
		}

		k := decodeFlowKey(key)
		v := aggregatePerCPUValue(perCPU)
		totalPkts += v.Packets
		totalBytes += v.Bytes
		flows = append(flows, flowSnapshot{Key: k, Value: v})
	}

	sort.Slice(flows, func(i, j int) bool {
		return flows[i].Value.Packets > flows[j].Value.Packets
	})
	if len(flows) > topK {
		flows = flows[:topK]
	}

	st.mu.Lock()
	st.activeFlows = len(flows)
	st.totalPackets = totalPkts
	st.totalBytes = totalBytes
	st.lastPollUnix = time.Now().Unix()
	st.topFlows = flows
	st.mu.Unlock()

	return nil
}

func aggregatePerCPUValue(raw []byte) flowMetrics {
	const oneCPUSize = 32
	var out flowMetrics
	for off := 0; off+oneCPUSize <= len(raw); off += oneCPUSize {
		chunk := raw[off : off+oneCPUSize]
		out.Packets += binary.LittleEndian.Uint64(chunk[0:8])
		out.Bytes += binary.LittleEndian.Uint64(chunk[8:16])
		last := binary.LittleEndian.Uint64(chunk[16:24])
		if last > out.LastSeenNS {
			out.LastSeenNS = last
		}
		first := binary.LittleEndian.Uint64(chunk[24:32])
		if out.FirstSeen == 0 || (first > 0 && first < out.FirstSeen) {
			out.FirstSeen = first
		}
	}
	return out
}

func decodeFlowKey(raw []byte) flowKey {
	return flowKey{
		SrcIP:   binary.LittleEndian.Uint32(raw[0:4]),
		DstIP:   binary.LittleEndian.Uint32(raw[4:8]),
		SrcPort: binary.LittleEndian.Uint16(raw[8:10]),
		DstPort: binary.LittleEndian.Uint16(raw[10:12]),
		Proto:   raw[12],
	}
}

func serveMetrics(addr string, st *collectorState) {
	http.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		st.mu.RLock()
		defer st.mu.RUnlock()

		var b strings.Builder
		b.WriteString("# TYPE xdp_netflow_active_flows gauge\n")
		b.WriteString(fmt.Sprintf("xdp_netflow_active_flows %d\n", st.activeFlows))
		b.WriteString("# TYPE xdp_netflow_packets_total_snapshot gauge\n")
		b.WriteString(fmt.Sprintf("xdp_netflow_packets_total_snapshot %d\n", st.totalPackets))
		b.WriteString("# TYPE xdp_netflow_bytes_total_snapshot gauge\n")
		b.WriteString(fmt.Sprintf("xdp_netflow_bytes_total_snapshot %d\n", st.totalBytes))
		b.WriteString("# TYPE xdp_netflow_last_poll_unix_seconds gauge\n")
		b.WriteString(fmt.Sprintf("xdp_netflow_last_poll_unix_seconds %d\n", st.lastPollUnix))
		b.WriteString("# TYPE xdp_netflow_flow_packets gauge\n")
		b.WriteString("# TYPE xdp_netflow_flow_bytes gauge\n")

		for _, f := range st.topFlows {
			labels := fmt.Sprintf("src=\"%s\",dst=\"%s\",proto=\"%d\",sport=\"%d\",dport=\"%d\"",
				ipString(f.Key.SrcIP), ipString(f.Key.DstIP), f.Key.Proto, ntohs(f.Key.SrcPort), ntohs(f.Key.DstPort))
			b.WriteString(fmt.Sprintf("xdp_netflow_flow_packets{%s} %d\n", labels, f.Value.Packets))
			b.WriteString(fmt.Sprintf("xdp_netflow_flow_bytes{%s} %d\n", labels, f.Value.Bytes))
		}

		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		_, _ = w.Write([]byte(b.String()))
	})

	log.Printf("metrics endpoint listening at %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("metrics server error: %v", err)
	}
}

func getenv(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}

func atoiOr(v string, d int) int {
	n, err := strconv.Atoi(v)
	if err != nil {
		return d
	}
	return n
}

func durationOr(v string, d time.Duration) time.Duration {
	t, err := time.ParseDuration(v)
	if err != nil {
		return d
	}
	return t
}

func ipString(raw uint32) string {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, raw)
	return net.IPv4(b[0], b[1], b[2], b[3]).String()
}

func ntohs(x uint16) uint16 { return (x<<8)&0xff00 | x>>8 }
