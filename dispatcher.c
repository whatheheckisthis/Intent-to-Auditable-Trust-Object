#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "pkcs11_signer.h"

typedef struct {
    uint8_t family;
    uint8_t proto;
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t pad;
    uint8_t src_addr[16];
    uint8_t dst_addr[16];
} flow_key_t;

typedef struct {
    uint64_t packets;
    uint64_t bytes;
    uint64_t first_seen_ns;
    uint64_t last_seen_ns;
} flow_metrics_t;

typedef struct {
    flow_key_t key;
    uint64_t packets;
    uint64_t bytes;
    uint64_t first_seen_ns;
    uint64_t last_seen_ns;
} aggregated_flow_t;

typedef struct {
    uint64_t unix_ts;
    flow_key_t masked_key;
    uint64_t packets;
    uint64_t bytes;
    uint64_t first_seen_ns;
    uint64_t last_seen_ns;
    uint8_t sha256[SHA256_DIGEST_LENGTH];
} evidence_t;

extern void fast_mask_ip_pair(void *src16, void *dst16, uint8_t family);

static volatile sig_atomic_t keep_running = 1;
static uint64_t verified_flows_total = 0;
static uint64_t unverified_flows_total = 0;
static uint64_t signed_evidence_total = 0;

static void on_signal(int signo) {
    (void)signo;
    keep_running = 0;
}

static int sign_evidence(const evidence_t *evidence,
                         unsigned char *signature,
                         size_t *signature_len) {
    if (sign_log_hsm((unsigned char *)evidence->sha256, SHA256_DIGEST_LENGTH) != 0) {
        return -1;
    }

    size_t out_len = 0;
    const unsigned char *raw_sig = pkcs11_last_signature(&out_len);
    if (raw_sig == NULL || out_len == 0 || out_len > *signature_len) {
        return -1;
    }

    memcpy(signature, raw_sig, out_len);
    *signature_len = out_len;
    return 0;
}

static int aggregate_flow_map(int map_fd, aggregated_flow_t *out, size_t out_cap, size_t *out_len) {
    flow_key_t prev = {0};
    flow_key_t next = {0};
    bool first = true;
    size_t count = 0;

    int ncpu = libbpf_num_possible_cpus();
    if (ncpu <= 0) {
        fprintf(stderr, "Unable to detect possible CPUs.\n");
        return -1;
    }

    flow_metrics_t *percpu = calloc((size_t)ncpu, sizeof(flow_metrics_t));
    if (percpu == NULL) {
        fprintf(stderr, "Out of memory allocating per-cpu read buffer.\n");
        return -1;
    }

    while (count < out_cap) {
        int rc = bpf_map_get_next_key(map_fd, first ? NULL : &prev, &next);
        if (rc != 0) {
            if (errno == ENOENT) {
                break;
            }
            perror("bpf_map_get_next_key");
            free(percpu);
            return -1;
        }

        memset(percpu, 0, sizeof(flow_metrics_t) * (size_t)ncpu);
        if (bpf_map_lookup_elem(map_fd, &next, percpu) != 0) {
            perror("bpf_map_lookup_elem");
            prev = next;
            first = false;
            continue;
        }

        uint64_t packets = 0;
        uint64_t bytes = 0;
        uint64_t first_seen_ns = 0;
        uint64_t last_seen_ns = 0;
        for (int i = 0; i < ncpu; ++i) {
            packets += percpu[i].packets;
            bytes += percpu[i].bytes;
            if (percpu[i].first_seen_ns != 0 && (first_seen_ns == 0 || percpu[i].first_seen_ns < first_seen_ns)) {
                first_seen_ns = percpu[i].first_seen_ns;
            }
            if (percpu[i].last_seen_ns > last_seen_ns) {
                last_seen_ns = percpu[i].last_seen_ns;
            }
        }

        out[count].key = next;
        out[count].packets = packets;
        out[count].bytes = bytes;
        out[count].first_seen_ns = first_seen_ns;
        out[count].last_seen_ns = last_seen_ns;
        count++;

        prev = next;
        first = false;
    }

    free(percpu);
    *out_len = count;
    return 0;
}

static int write_log_line(const char *path, const char *line) {
    FILE *f = fopen(path, "a");
    if (f == NULL) {
        perror("fopen");
        return -1;
    }

    if (fputs(line, f) == EOF || fputc('\n', f) == EOF) {
        perror("fputs");
        fclose(f);
        return -1;
    }

    if (fflush(f) != 0 || fsync(fileno(f)) != 0) {
        perror("flush/fsync");
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

static void format_addr(uint8_t family, const uint8_t addr[16], char *out, size_t out_len) {
    if (family == AF_INET) {
        inet_ntop(AF_INET, addr, out, (socklen_t)out_len);
    } else if (family == AF_INET6) {
        inet_ntop(AF_INET6, addr, out, (socklen_t)out_len);
    } else {
        snprintf(out, out_len, "unknown");
    }
}

static int export_metrics_http(uint16_t port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return -1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind");
        close(server_fd);
        return -1;
    }

    if (listen(server_fd, 16) != 0) {
        perror("listen");
        close(server_fd);
        return -1;
    }

    while (keep_running) {
        int client = accept(server_fd, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("accept");
            break;
        }

        double verification_success = 1.0;
        const uint64_t total = verified_flows_total + unverified_flows_total;
        if (total > 0) {
            verification_success = (double)verified_flows_total / (double)total;
        }

        char body[768];
        int body_len = snprintf(
            body,
            sizeof(body),
            "# HELP ato_verified_flows_total Number of HSM-verified flows.\n"
            "# TYPE ato_verified_flows_total counter\n"
            "ato_verified_flows_total %llu\n"
            "# HELP ato_unverified_flows_total Number of unverified flows.\n"
            "# TYPE ato_unverified_flows_total counter\n"
            "ato_unverified_flows_total %llu\n"
            "# HELP ato_signed_evidence_total Number of signed evidence records emitted.\n"
            "# TYPE ato_signed_evidence_total counter\n"
            "ato_signed_evidence_total %llu\n"
            "# HELP ato_signature_verification_success Signature verification success ratio.\n"
            "# TYPE ato_signature_verification_success gauge\n"
            "ato_signature_verification_success %.6f\n",
            (unsigned long long)verified_flows_total,
            (unsigned long long)unverified_flows_total,
            (unsigned long long)signed_evidence_total,
            verification_success);

        char response[1400];
        int n = snprintf(response,
                         sizeof(response),
                         "HTTP/1.1 200 OK\r\n"
                         "Content-Type: text/plain; version=0.0.4\r\n"
                         "Content-Length: %d\r\n"
                         "Connection: close\r\n\r\n%.*s",
                         body_len,
                         body_len,
                         body);
        if (n > 0) {
            (void)write(client, response, (size_t)n);
        }
        close(client);
    }

    close(server_fd);
    return 0;
}

static int run_dispatch_loop(const char *map_path, const char *log_path, unsigned interval_s) {
    int map_fd = bpf_obj_get(map_path);
    if (map_fd < 0) {
        perror("bpf_obj_get flow_map");
        return -1;
    }

    if (pkcs11_initialize_from_env() != 0) {
        close(map_fd);
        return -1;
    }

    const size_t cap = 4096;
    aggregated_flow_t *flows = calloc(cap, sizeof(*flows));
    if (flows == NULL) {
        close(map_fd);
        pkcs11_cleanup();
        return -1;
    }

    while (keep_running) {
        size_t len = 0;
        if (aggregate_flow_map(map_fd, flows, cap, &len) != 0) {
            sleep(interval_s);
            continue;
        }

        for (size_t i = 0; i < len; ++i) {
            evidence_t evidence = {
                .unix_ts = (uint64_t)time(NULL),
                .masked_key = flows[i].key,
                .packets = flows[i].packets,
                .bytes = flows[i].bytes,
                .first_seen_ns = flows[i].first_seen_ns,
                .last_seen_ns = flows[i].last_seen_ns,
            };

            fast_mask_ip_pair(evidence.masked_key.src_addr,
                              evidence.masked_key.dst_addr,
                              evidence.masked_key.family == AF_INET ? 4 : 6);

            SHA256((const unsigned char *)&evidence,
                   sizeof(evidence) - SHA256_DIGEST_LENGTH,
                   evidence.sha256);

            unsigned char signature[512];
            size_t sig_len = sizeof(signature);
            int verified = sign_evidence(&evidence, signature, &sig_len) == 0;
            if (verified) {
                verified_flows_total += 1;
            } else {
                unverified_flows_total += 1;
            }

            char encoded[1024] = {0};
            if (verified) {
                EVP_EncodeBlock((unsigned char *)encoded, signature, (int)sig_len);
            } else {
                strcpy(encoded, "UNSIGNED");
            }

            char digest_hex[SHA256_DIGEST_LENGTH * 2 + 1] = {0};
            for (size_t j = 0; j < SHA256_DIGEST_LENGTH; ++j) {
                snprintf(digest_hex + (j * 2), 3, "%02x", evidence.sha256[j]);
            }

            char src_txt[INET6_ADDRSTRLEN] = {0};
            char dst_txt[INET6_ADDRSTRLEN] = {0};
            format_addr(evidence.masked_key.family, evidence.masked_key.src_addr, src_txt, sizeof(src_txt));
            format_addr(evidence.masked_key.family, evidence.masked_key.dst_addr, dst_txt, sizeof(dst_txt));

            char line[2200];
            snprintf(line,
                     sizeof(line),
                     "ts=%llu family=%u src=%s dst=%s proto=%u packets=%llu bytes=%llu first_seen_ns=%llu last_seen_ns=%llu digest_sha256=%s verified=%d signature=%s",
                     (unsigned long long)evidence.unix_ts,
                     evidence.masked_key.family,
                     src_txt,
                     dst_txt,
                     evidence.masked_key.proto,
                     (unsigned long long)evidence.packets,
                     (unsigned long long)evidence.bytes,
                     (unsigned long long)evidence.first_seen_ns,
                     (unsigned long long)evidence.last_seen_ns,
                     digest_hex,
                     verified,
                     encoded);

            if (write_log_line(log_path, line) == 0) {
                signed_evidence_total += 1;
            }
        }

        sleep(interval_s);
    }

    free(flows);
    close(map_fd);
    pkcs11_cleanup();
    return 0;
}

int main(void) {
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    const char *map_path = getenv("FLOW_MAP_PATH");
    if (map_path == NULL) {
        map_path = "/sys/fs/bpf/flow_map";
    }

    const char *log_path = getenv("SIGNED_EVIDENCE_LOG");
    if (log_path == NULL) {
        log_path = "/var/log/audit/signed_evidence.log";
    }

    unsigned interval_s = 5;
    const char *interval_env = getenv("DISPATCH_INTERVAL_S");
    if (interval_env != NULL) {
        interval_s = (unsigned)strtoul(interval_env, NULL, 10);
        if (interval_s == 0) {
            interval_s = 5;
        }
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }

    if (pid == 0) {
        return run_dispatch_loop(map_path, log_path, interval_s) == 0 ? 0 : 1;
    }

    return export_metrics_http(9400) == 0 ? 0 : 1;
}
