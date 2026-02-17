package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math/bits"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

type flowID struct {
	SrcIP   uint32
	DstIP   uint32
	SrcPort uint16
	DstPort uint16
	Proto   uint8
}

type flowMetrics struct {
	Packets    uint64
	Bytes      uint64
	LastSeenNS uint64
}

type flowSnapshot struct {
	Key   flowID
	Value flowMetrics
}

type state struct {
	mu        sync.RWMutex
	active    int
	totalPkts uint64
	totalByts uint64
	lastPoll  int64
	topFlows  []flowSnapshot
}

type mapEntry struct {
	Key    []string         `json:"key"`
	Value  []string         `json:"value"`
	Values []perCPUMapValue `json:"values"`
}

type perCPUMapValue struct {
	Value []string `json:"value"`
}

type mapSummary struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}

func main() {
	iface := getenv("XDP_INTERFACE", "eth0")
	metricsAddr := getenv("METRICS_ADDR", "0.0.0.0:9400")
	xdpMode := getenv("XDP_MODE", "drv")
	objPath := getenv("BPF_OBJECT", "/opt/netflow/netflow_filter.bpf.o")
	topK := atoiOr(getenv("NETFLOW_TOPK", "50"), 50)
	pollEvery := durationOr(getenv("NETFLOW_POLL_INTERVAL", "10s"), 10*time.Second)

	if topK < 1 {
		topK = 1
	}

	if err := attachXDP(iface, xdpMode, objPath); err != nil {
		log.Fatalf("attach xdp: %v", err)
	}
	defer func() {
		if err := detachXDP(iface, xdpMode); err != nil {
			log.Printf("detach xdp warning: %v", err)
		}
	}()
	log.Printf("netflow_filter attached to %s (mode=%s)", iface, xdpMode)

	mapID, err := findFlowMapID()
	if err != nil {
		log.Fatalf("locate flow_map: %v", err)
	}
	log.Printf("flow_map id=%d", mapID)

	st := &state{}

	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		serveMetrics(w, st)
	})

	go func() {
		if err := http.ListenAndServe(metricsAddr, nil); err != nil {
			log.Fatalf("metrics server failed: %v", err)
		}
	}()
	log.Printf("prometheus metrics listening at %s", metricsAddr)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	ticker := time.NewTicker(pollEvery)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("shutdown requested")
			return
		case <-ticker.C:
			if err := pollFlowMap(mapID, topK, st); err != nil {
				log.Printf("poll flow_map failed: %v", err)
			}
		}
	}
}

func attachXDP(iface, mode, objPath string) error {
	args := []string{"link", "set", "dev", iface}
	switch strings.ToLower(mode) {
	case "hw":
		args = append(args, "xdpoffload", "obj", objPath, "sec", "xdp")
	case "skb":
		args = append(args, "xdpgeneric", "obj", objPath, "sec", "xdp")
	default:
		args = append(args, "xdpdrv", "obj", objPath, "sec", "xdp")
	}
	return run("ip", args...)
}

func detachXDP(iface, mode string) error {
	args := []string{"link", "set", "dev", iface}
	switch strings.ToLower(mode) {
	case "hw":
		args = append(args, "xdpoffload", "off")
	case "skb":
		args = append(args, "xdpgeneric", "off")
	default:
		args = append(args, "xdpdrv", "off")
	}
	return run("ip", args...)
}

func findFlowMapID() (int, error) {
	out, err := exec.Command("bpftool", "-j", "map", "show").Output()
	if err != nil {
		return 0, err
	}
	var maps []mapSummary
	if err := json.Unmarshal(out, &maps); err != nil {
		return 0, err
	}
	for _, m := range maps {
		if m.Name == "flow_map" {
			return m.ID, nil
		}
	}
	return 0, fmt.Errorf("flow_map not found")
}

func pollFlowMap(mapID, topK int, st *state) error {
	out, err := exec.Command("bpftool", "-j", "map", "dump", "id", strconv.Itoa(mapID)).Output()
	if err != nil {
		return err
	}
	var entries []mapEntry
	if len(bytes.TrimSpace(out)) > 0 {
		if err := json.Unmarshal(out, &entries); err != nil {
			return err
		}
	}

	flows := make([]flowSnapshot, 0, len(entries))
	var pkts, byts uint64
	for _, e := range entries {
		k, err := parseFlowKey(e.Key)
		if err != nil {
			continue
		}
		v, err := parseFlowValues(e)
		if err != nil {
			continue
		}
		flows = append(flows, flowSnapshot{Key: k, Value: v})
		pkts += v.Packets
		byts += v.Bytes
	}

	sort.Slice(flows, func(i, j int) bool { return flows[i].Value.Packets > flows[j].Value.Packets })
	if len(flows) > topK {
		flows = flows[:topK]
	}

	st.mu.Lock()
	st.active = len(entries)
	st.totalPkts = pkts
	st.totalByts = byts
	st.lastPoll = time.Now().Unix()
	st.topFlows = flows
	st.mu.Unlock()
	return nil
}

func parseFlowKey(xs []string) (flowID, error) {
	b, err := hexBytes(xs)
	if err != nil || len(b) < 16 {
		return flowID{}, fmt.Errorf("bad key")
	}
	return flowID{
		SrcIP:   binary.LittleEndian.Uint32(b[0:4]),
		DstIP:   binary.LittleEndian.Uint32(b[4:8]),
		SrcPort: binary.LittleEndian.Uint16(b[8:10]),
		DstPort: binary.LittleEndian.Uint16(b[10:12]),
		Proto:   b[12],
	}, nil
}

func parseFlowValue(xs []string) (flowMetrics, error) {
	b, err := hexBytes(xs)
	if err != nil || len(b) < 24 {
		return flowMetrics{}, fmt.Errorf("bad value")
	}
	return flowMetrics{
		Packets:    binary.LittleEndian.Uint64(b[0:8]),
		Bytes:      binary.LittleEndian.Uint64(b[8:16]),
		LastSeenNS: binary.LittleEndian.Uint64(b[16:24]),
	}, nil
}

func parseFlowValues(e mapEntry) (flowMetrics, error) {
	if len(e.Value) > 0 {
		return parseFlowValue(e.Value)
	}

	if len(e.Values) == 0 {
		return flowMetrics{}, fmt.Errorf("missing value")
	}

	var out flowMetrics
	for _, v := range e.Values {
		m, err := parseFlowValue(v.Value)
		if err != nil {
			return flowMetrics{}, err
		}
		out.Packets += m.Packets
		out.Bytes += m.Bytes
		if m.LastSeenNS > out.LastSeenNS {
			out.LastSeenNS = m.LastSeenNS
		}
	}

	return out, nil
}

func hexBytes(xs []string) ([]byte, error) {
	b := make([]byte, 0, len(xs))
	for _, x := range xs {
		x = strings.TrimPrefix(x, "0x")
		v, err := strconv.ParseUint(x, 16, 8)
		if err != nil {
			return nil, err
		}
		b = append(b, byte(v))
	}
	return b, nil
}

func serveMetrics(w http.ResponseWriter, st *state) {
	st.mu.RLock()
	defer st.mu.RUnlock()

	var b strings.Builder
	b.WriteString("# HELP xdp_netflow_active_flows Current number of active flow entries in flow_map\n")
	b.WriteString("# TYPE xdp_netflow_active_flows gauge\n")
	b.WriteString(fmt.Sprintf("xdp_netflow_active_flows %d\n", st.active))
	b.WriteString("# HELP xdp_netflow_packets_total_snapshot Snapshot sum of packets across tracked flows\n")
	b.WriteString("# TYPE xdp_netflow_packets_total_snapshot gauge\n")
	b.WriteString(fmt.Sprintf("xdp_netflow_packets_total_snapshot %d\n", st.totalPkts))
	b.WriteString("# HELP xdp_netflow_bytes_total_snapshot Snapshot sum of bytes across tracked flows\n")
	b.WriteString("# TYPE xdp_netflow_bytes_total_snapshot gauge\n")
	b.WriteString(fmt.Sprintf("xdp_netflow_bytes_total_snapshot %d\n", st.totalByts))
	b.WriteString("# HELP xdp_netflow_last_poll_unix_seconds Unix timestamp of last flow_map poll\n")
	b.WriteString("# TYPE xdp_netflow_last_poll_unix_seconds gauge\n")
	b.WriteString(fmt.Sprintf("xdp_netflow_last_poll_unix_seconds %d\n", st.lastPoll))
	b.WriteString("# HELP xdp_netflow_flow_packets Per-flow packet counters for top-N flows\n")
	b.WriteString("# TYPE xdp_netflow_flow_packets gauge\n")
	b.WriteString("# HELP xdp_netflow_flow_bytes Per-flow byte counters for top-N flows\n")
	b.WriteString("# TYPE xdp_netflow_flow_bytes gauge\n")

	for _, f := range st.topFlows {
		lbl := fmt.Sprintf("src=\"%s\",dst=\"%s\",proto=\"%d\",sport=\"%d\",dport=\"%d\"",
			ipString(f.Key.SrcIP), ipString(f.Key.DstIP), f.Key.Proto,
			bits.ReverseBytes16(f.Key.SrcPort), bits.ReverseBytes16(f.Key.DstPort))
		b.WriteString(fmt.Sprintf("xdp_netflow_flow_packets{%s} %d\n", lbl, f.Value.Packets))
		b.WriteString(fmt.Sprintf("xdp_netflow_flow_bytes{%s} %d\n", lbl, f.Value.Bytes))
	}

	payload := b.String()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
	_, _ = w.Write([]byte(payload))
}

func ipString(raw uint32) string {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, raw)
	return net.IPv4(b[0], b[1], b[2], b[3]).String()
}

func run(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
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

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
