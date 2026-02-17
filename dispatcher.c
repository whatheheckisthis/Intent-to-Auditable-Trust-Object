#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
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

#include "pkcs11_signer.h"

typedef struct {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t proto;
    uint8_t pad[3];
} flow_key_t;

typedef struct {
    uint64_t packets;
    uint64_t bytes;
    uint64_t last_seen_ns;
    uint64_t first_seen_ns;
} flow_metrics_t;

typedef struct {
    flow_key_t key;
    uint64_t packets;
    uint64_t bytes;
} aggregated_flow_t;

static volatile sig_atomic_t keep_running = 1;
static uint64_t verified_flows_total = 0;
static uint64_t unverified_flows_total = 0;
static uint64_t signed_evidence_total = 0;

static void on_signal(int signo) {
    (void)signo;
    keep_running = 0;
}

static uint64_t to_be64(uint64_t v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap64(v);
#else
    return v;
#endif
}

static void mask_sensitive_fields(flow_key_t *key) {
    key->src_ip &= htonl(0xFFFFFF00);
    key->dst_ip &= htonl(0xFFFFFF00);
    key->src_port = 0;
    key->dst_port = 0;
}

static int sign_evidence(const unsigned char *digest,
                         size_t digest_len,
                         unsigned char *signature,
                         size_t *signature_len) {
    const int max_attempts = 3;
    for (int i = 0; i < max_attempts; ++i) {
        if (sign_log_hsm(digest, digest_len) == 0) {
            size_t out_len = 0;
            const unsigned char *raw_sig = pkcs11_last_signature(&out_len);
            if (raw_sig == NULL || out_len == 0 || out_len > *signature_len) {
                return -1;
            }
            memcpy(signature, raw_sig, out_len);
            *signature_len = out_len;
            return 0;
        }
        usleep(150000);
    }
    return -1;
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
        for (int i = 0; i < ncpu; ++i) {
            packets += percpu[i].packets;
            bytes += percpu[i].bytes;
        }

        out[count].key = next;
        out[count].packets = packets;
        out[count].bytes = bytes;
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

        char body[512];
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
            "ato_signed_evidence_total %llu\n",
            (unsigned long long)verified_flows_total,
            (unsigned long long)unverified_flows_total,
            (unsigned long long)signed_evidence_total);

        char response[1024];
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
            flow_key_t masked = flows[i].key;
            mask_sensitive_fields(&masked);

            uint64_t payload[6] = {
                to_be64((uint64_t)masked.src_ip),
                to_be64((uint64_t)masked.dst_ip),
                to_be64((uint64_t)masked.src_port),
                to_be64((uint64_t)masked.dst_port),
                to_be64(flows[i].packets),
                to_be64(flows[i].bytes),
            };

            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256((const unsigned char *)payload, sizeof(payload), digest);

            unsigned char signature[512];
            size_t sig_len = sizeof(signature);
            int verified = sign_evidence(digest, sizeof(digest), signature, &sig_len) == 0;
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

            char line[1600];
            snprintf(line,
                     sizeof(line),
                     "ts=%ld src_ip=%u dst_ip=%u proto=%u packets=%llu bytes=%llu verified=%d signature=%s",
                     time(NULL),
                     ntohl(masked.src_ip),
                     ntohl(masked.dst_ip),
                     masked.proto,
                     (unsigned long long)flows[i].packets,
                     (unsigned long long)flows[i].bytes,
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
