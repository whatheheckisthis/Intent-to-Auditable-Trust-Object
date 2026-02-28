#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
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
#include "osint_audit_log.h"
#include "tinycbor.h"

#define FLOW_CAP 4096U
#define SIGN_BATCH_SIZE 1000U
#define MAX_SIGNATURE_LEN 128U

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
    uint64_t bpf_map_checksum;
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

static int exfiltrate_to_audit_vault(const char *vault_path,
                                     const uint8_t *signed_blob,
                                     size_t signed_blob_len,
                                     const uint8_t *signature,
                                     size_t signature_len) {
    FILE *vault = fopen(vault_path, "ab");
    if (vault == NULL) {
        return -1;
    }

    uint32_t cbor_len = (uint32_t)signed_blob_len;
    uint16_t sig_len = (uint16_t)signature_len;
    int ok = fwrite(&cbor_len, sizeof(cbor_len), 1, vault) == 1 &&
             fwrite(&sig_len, sizeof(sig_len), 1, vault) == 1 &&
             fwrite(signed_blob, signed_blob_len, 1, vault) == 1 &&
             fwrite(signature, signature_len, 1, vault) == 1;

    fclose(vault);
    return ok ? 0 : -1;
}

static int cbor_wrap_ipld_record(const uint8_t *data,
                                size_t data_len,
                                const uint8_t *signature,
                                size_t signature_len,
                                const char *pubkey,
                                uint8_t **out,
                                size_t *out_len) {
    if (data == NULL || signature == NULL || pubkey == NULL || out == NULL || out_len == NULL) {
        return -1;
    }

    uint8_t *buf = calloc(1, data_len + signature_len + strlen(pubkey) + 256U);
    if (buf == NULL) {
        return -1;
    }

    CborEncoder enc;
    CborEncoder map;
    cbor_encoder_init(&enc, buf, data_len + signature_len + strlen(pubkey) + 256U, 0);
    if (cbor_encoder_create_map(&enc, &map, 3) != CborNoError) {
        free(buf);
        return -1;
    }

    /* Canonical map key order: data, pubkey, signature */
    if (cbor_encode_text_stringz(&map, "data") != CborNoError ||
        cbor_encode_byte_string(&map, data, data_len) != CborNoError ||
        cbor_encode_text_stringz(&map, "pubkey") != CborNoError ||
        cbor_encode_text_stringz(&map, pubkey) != CborNoError ||
        cbor_encode_text_stringz(&map, "signature") != CborNoError ||
        cbor_encode_byte_string(&map, signature, signature_len) != CborNoError ||
        cbor_encoder_close_container(&enc, &map) != CborNoError) {
        free(buf);
        return -1;
    }

    *out_len = cbor_encoder_get_buffer_size(&enc, buf);
    *out = buf;
    return 0;
}

static int extract_ipfs_hash(const char *response, char *cid, size_t cid_len) {
    const char *tag = "\"Hash\":\"";
    const char *start = strstr(response, tag);
    if (start == NULL) {
        return -1;
    }

    start += strlen(tag);
    const char *end = strchr(start, '"');
    if (end == NULL) {
        return -1;
    }

    size_t n = (size_t)(end - start);
    if (n == 0 || n >= cid_len) {
        return -1;
    }

    memcpy(cid, start, n);
    cid[n] = '\0';
    return 0;
}

static int upload_blob_to_ipfs(const char *host,
                               const char *port,
                               const uint8_t *blob,
                               size_t blob_len,
                               char *cid,
                               size_t cid_len) {
    if (host == NULL || port == NULL || blob == NULL || cid == NULL) {
        return -1;
    }

    const char *boundary = "------------------------ato-ipfs-boundary";
    char header[512];
    int header_len = snprintf(header,
                              sizeof(header),
                              "--%s\r\n"
                              "Content-Disposition: form-data; name=\"file\"; filename=\"evidence.cbor\"\r\n"
                              "Content-Type: application/cbor\r\n\r\n",
                              boundary);
    if (header_len <= 0 || (size_t)header_len >= sizeof(header)) {
        return -1;
    }

    char trailer[128];
    int trailer_len = snprintf(trailer, sizeof(trailer), "\r\n--%s--\r\n", boundary);
    if (trailer_len <= 0 || (size_t)trailer_len >= sizeof(trailer)) {
        return -1;
    }

    size_t body_len = (size_t)header_len + blob_len + (size_t)trailer_len;

    char request_head[1024];
    int request_head_len = snprintf(request_head,
                                    sizeof(request_head),
                                    "POST /api/v0/add?pin=true HTTP/1.1\r\n"
                                    "Host: %s:%s\r\n"
                                    "Content-Type: multipart/form-data; boundary=%s\r\n"
                                    "Content-Length: %zu\r\n"
                                    "Connection: close\r\n\r\n",
                                    host,
                                    port,
                                    boundary,
                                    body_len);
    if (request_head_len <= 0 || (size_t)request_head_len >= sizeof(request_head)) {
        return -1;
    }

    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *res = NULL;
    if (getaddrinfo(host, port, &hints, &res) != 0) {
        return -1;
    }

    int sock = -1;
    for (struct addrinfo *rp = res; rp != NULL; rp = rp->ai_next) {
        sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock < 0) continue;
        if (connect(sock, rp->ai_addr, rp->ai_addrlen) == 0) break;
        close(sock);
        sock = -1;
    }
    freeaddrinfo(res);

    if (sock < 0) {
        return -1;
    }

    int ok = write(sock, request_head, (size_t)request_head_len) == request_head_len &&
             write(sock, header, (size_t)header_len) == header_len &&
             write(sock, blob, blob_len) == (ssize_t)blob_len &&
             write(sock, trailer, (size_t)trailer_len) == trailer_len;

    char response[4096] = {0};
    ssize_t nread = read(sock, response, sizeof(response) - 1);
    close(sock);
    if (!ok || nread <= 0) {
        return -1;
    }

    response[nread] = '\0';
    return extract_ipfs_hash(response, cid, cid_len);
}

static void on_signal(int signo) {
    (void)signo;
    keep_running = 0;
}

static uint64_t fold_checksum64(uint64_t seed, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    for (size_t i = 0; i < len; ++i) {
        seed ^= (uint64_t)p[i];
        seed *= 1099511628211ULL;
    }
    return seed;
}

static int aggregate_flow_map(int map_fd, aggregated_flow_t *out, size_t out_cap, size_t *out_len, uint64_t *map_checksum) {
    flow_key_t prev = {0};
    flow_key_t next = {0};
    bool first = true;
    size_t count = 0;
    uint64_t checksum = 1469598103934665603ULL;

    int ncpu = libbpf_num_possible_cpus();
    if (ncpu <= 0) {
        return -1;
    }

    flow_metrics_t *percpu = calloc((size_t)ncpu, sizeof(flow_metrics_t));
    if (percpu == NULL) {
        return -1;
    }

    while (count < out_cap) {
        int rc = bpf_map_get_next_key(map_fd, first ? NULL : &prev, &next);
        if (rc != 0) {
            if (errno == ENOENT) {
                break;
            }
            free(percpu);
            return -1;
        }

        memset(percpu, 0, sizeof(flow_metrics_t) * (size_t)ncpu);
        if (bpf_map_lookup_elem(map_fd, &next, percpu) != 0) {
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

        checksum = fold_checksum64(checksum, &out[count], sizeof(out[count]));

        count++;
        prev = next;
        first = false;
    }

    free(percpu);
    *map_checksum = checksum;
    *out_len = count;
    return 0;
}

static int write_log_line(const char *path, const char *line) {
    FILE *f = fopen(path, "a");
    if (f == NULL) {
        return -1;
    }

    if (fputs(line, f) == EOF || fputc('\n', f) == EOF) {
        fclose(f);
        return -1;
    }

    if (fflush(f) != 0 || fsync(fileno(f)) != 0) {
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
        return -1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(server_fd);
        return -1;
    }

    if (listen(server_fd, 16) != 0) {
        close(server_fd);
        return -1;
    }

    while (keep_running) {
        int client = accept(server_fd, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) {
                continue;
            }
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
        return -1;
    }

    if (pkcs11_initialize_from_env() != 0) {
        close(map_fd);
        return -1;
    }

    aggregated_flow_t *flows = calloc(FLOW_CAP, sizeof(*flows));
    evidence_t *batch = calloc(SIGN_BATCH_SIZE, sizeof(*batch));
    unsigned char *batch_sigs = calloc(SIGN_BATCH_SIZE, MAX_SIGNATURE_LEN);
    size_t *batch_sig_lens = calloc(SIGN_BATCH_SIZE, sizeof(size_t));
    if (flows == NULL || batch == NULL || batch_sigs == NULL || batch_sig_lens == NULL) {
        free(flows);
        free(batch);
        free(batch_sigs);
        free(batch_sig_lens);
        close(map_fd);
        pkcs11_cleanup();
        return -1;
    }

    const char *vault_path = getenv("AUDIT_VAULT_PATH");
    if (vault_path == NULL) {
        vault_path = "/var/log/audit/audit_vault.cbor";
    }

    const char *ipfs_host = getenv("IPFS_API_HOST");
    if (ipfs_host == NULL) ipfs_host = "127.0.0.1";
    const char *ipfs_port = getenv("IPFS_API_PORT");
    if (ipfs_port == NULL) ipfs_port = "5001";
    const char *hsm_key_id = getenv("HSM_KEY_ID");
    if (hsm_key_id == NULL) hsm_key_id = getenv("HSM_KEY_LABEL");
    if (hsm_key_id == NULL) hsm_key_id = "unknown-hsm-key";

    while (keep_running) {
        size_t len = 0;
        uint64_t map_checksum = 0;
        if (aggregate_flow_map(map_fd, flows, FLOW_CAP, &len, &map_checksum) != 0) {
            sleep(interval_s);
            continue;
        }

        for (size_t cursor = 0; cursor < len;) {
            size_t chunk = len - cursor;
            if (chunk > SIGN_BATCH_SIZE) {
                chunk = SIGN_BATCH_SIZE;
            }

            unsigned char digest_buf[SIGN_BATCH_SIZE * SHA256_DIGEST_LENGTH];
            memset(digest_buf, 0, sizeof(digest_buf));

            for (size_t i = 0; i < chunk; ++i) {
                batch[i] = (evidence_t){
                    .unix_ts = (uint64_t)time(NULL),
                    .bpf_map_checksum = map_checksum,
                    .masked_key = flows[cursor + i].key,
                    .packets = flows[cursor + i].packets,
                    .bytes = flows[cursor + i].bytes,
                    .first_seen_ns = flows[cursor + i].first_seen_ns,
                    .last_seen_ns = flows[cursor + i].last_seen_ns,
                };

                fast_mask_ip_pair(batch[i].masked_key.src_addr,
                                  batch[i].masked_key.dst_addr,
                                  batch[i].masked_key.family == AF_INET ? 4 : 6);

                SHA256((const unsigned char *)&batch[i],
                       sizeof(batch[i]) - SHA256_DIGEST_LENGTH,
                       batch[i].sha256);
                memcpy(digest_buf + (i * SHA256_DIGEST_LENGTH), batch[i].sha256, SHA256_DIGEST_LENGTH);
            }

            int batch_ok = sign_log_hsm_batch(digest_buf,
                                              SHA256_DIGEST_LENGTH,
                                              chunk,
                                              batch_sigs,
                                              batch_sig_lens,
                                              MAX_SIGNATURE_LEN) == 0;

            for (size_t i = 0; i < chunk; ++i) {
                int verified = batch_ok ? 1 : 0;

                osint_sanitized_payload_t osint_packet = {
                    .masked_first_seen_ns = batch[i].first_seen_ns,
                    .masked_last_seen_ns = batch[i].last_seen_ns,
                };
                memcpy(osint_packet.source_digest, batch[i].sha256, SHA256_DIGEST_LENGTH);
                memcpy(&osint_packet.masked_src_ipv4, batch[i].masked_key.src_addr, sizeof(uint32_t));
                memcpy(&osint_packet.masked_dst_ipv4, batch[i].masked_key.dst_addr, sizeof(uint32_t));
                osint_packet.masked_src_ipv4 = ntohl(osint_packet.masked_src_ipv4);
                osint_packet.masked_dst_ipv4 = ntohl(osint_packet.masked_dst_ipv4);
                (void)mask_v7_sve2(&osint_packet);

                char messy_json_like[512];
                snprintf(messy_json_like,
                         sizeof(messy_json_like),
                         "{'lead_family':%u,'proto':%u,'packets':%llu,'bytes':%llu,'notes':'messy-forensic-input'}",
                         batch[i].masked_key.family,
                         batch[i].masked_key.proto,
                         (unsigned long long)batch[i].packets,
                         (unsigned long long)batch[i].bytes);

                uint8_t *cbor_blob = NULL;
                size_t cbor_blob_len = 0;
                char primary_cid[128] = "UNPUBLISHED";
                if (cbor_wrap_osint_data(messy_json_like, &osint_packet, &cbor_blob, &cbor_blob_len) == 0 && verified) {
                    uint8_t *ipld_blob = NULL;
                    size_t ipld_blob_len = 0;
                    if (cbor_wrap_ipld_record(cbor_blob,
                                              cbor_blob_len,
                                              batch_sigs + (i * MAX_SIGNATURE_LEN),
                                              batch_sig_lens[i],
                                              hsm_key_id,
                                              &ipld_blob,
                                              &ipld_blob_len) == 0) {
                        (void)exfiltrate_to_audit_vault(vault_path,
                                                        ipld_blob,
                                                        ipld_blob_len,
                                                        batch_sigs + (i * MAX_SIGNATURE_LEN),
                                                        batch_sig_lens[i]);
                        if (upload_blob_to_ipfs(ipfs_host,
                                                ipfs_port,
                                                ipld_blob,
                                                ipld_blob_len,
                                                primary_cid,
                                                sizeof(primary_cid)) != 0) {
                            strcpy(primary_cid, "IPFS_UPLOAD_FAILED");
                        }
                    }
                    free(ipld_blob);
                }
                free(cbor_blob);

                if (verified) {
                    verified_flows_total += 1;
                } else {
                    unverified_flows_total += 1;
                }

                char encoded[512] = {0};
                if (verified) {
                    EVP_EncodeBlock((unsigned char *)encoded,
                                    batch_sigs + (i * MAX_SIGNATURE_LEN),
                                    (int)batch_sig_lens[i]);
                } else {
                    strcpy(encoded, "UNSIGNED");
                }

                char digest_hex[SHA256_DIGEST_LENGTH * 2 + 1] = {0};
                for (size_t j = 0; j < SHA256_DIGEST_LENGTH; ++j) {
                    snprintf(digest_hex + (j * 2), 3, "%02x", batch[i].sha256[j]);
                }

                char src_txt[INET6_ADDRSTRLEN] = {0};
                char dst_txt[INET6_ADDRSTRLEN] = {0};
                format_addr(batch[i].masked_key.family, batch[i].masked_key.src_addr, src_txt, sizeof(src_txt));
                format_addr(batch[i].masked_key.family, batch[i].masked_key.dst_addr, dst_txt, sizeof(dst_txt));

                char line[2200];
                snprintf(line,
                         sizeof(line),
                         "primary_cid=%s ts=%llu family=%u src=%s dst=%s proto=%u packets=%llu bytes=%llu first_seen_ns=%llu last_seen_ns=%llu bpf_map_checksum=%llu digest_sha256=%s verified=%d signature=%s",
                         primary_cid,
                         (unsigned long long)batch[i].unix_ts,
                         batch[i].masked_key.family,
                         src_txt,
                         dst_txt,
                         batch[i].masked_key.proto,
                         (unsigned long long)batch[i].packets,
                         (unsigned long long)batch[i].bytes,
                         (unsigned long long)batch[i].first_seen_ns,
                         (unsigned long long)batch[i].last_seen_ns,
                         (unsigned long long)batch[i].bpf_map_checksum,
                         digest_hex,
                         verified,
                         encoded);

                if (write_log_line(log_path, line) == 0) {
                    signed_evidence_total += 1;
                }
            }

            cursor += chunk;
        }

        sleep(interval_s);
    }

    free(flows);
    free(batch);
    free(batch_sigs);
    free(batch_sig_lens);
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
        return 1;
    }

    if (pid == 0) {
        return run_dispatch_loop(map_path, log_path, interval_s) == 0 ? 0 : 1;
    }

    return export_metrics_http(9400) == 0 ? 0 : 1;
}
