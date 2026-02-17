#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "pkcs11_signer.h"

#define EVIDENCE_MAX 8192

extern void fast_mask_ip_pairs(unsigned char *buf, const unsigned char *mask, size_t blocks16);

struct metrics_snapshot {
    unsigned long flow_records;
    unsigned long long packets_total;
    unsigned long long bytes_total;
    unsigned long long signed_batches;
    unsigned long long signing_failures;
};

static volatile sig_atomic_t g_running = 1;
static pthread_mutex_t g_metrics_mu = PTHREAD_MUTEX_INITIALIZER;
static struct metrics_snapshot g_metrics = {0};

static void stop_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

static int read_command_output(const char *cmd, char *buf, size_t cap)
{
    FILE *fp = popen(cmd, "r");
    if (!fp)
        return -1;

    size_t used = fread(buf, 1, cap - 1, fp);
    buf[used] = '\0';
    int status = pclose(fp);
    return (status == 0) ? 0 : -1;
}

static void mask_ip_pair(unsigned char src[16], unsigned char dst[16])
{
    static const unsigned char ipv4_ipv6_mask[32] = {
        0xff, 0xff, 0xff, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0xff, 0xff, 0xff, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    unsigned char pair[32];
    memcpy(pair, src, 16);
    memcpy(pair + 16, dst, 16);
    fast_mask_ip_pairs(pair, ipv4_ipv6_mask, 2);
    memcpy(src, pair, 16);
    memcpy(dst, pair + 16, 16);
}

static void extract_totals_from_json(const char *json, unsigned long *flows, unsigned long long *packets, unsigned long long *bytes)
{
    const char *p = json;
    *flows = 0;
    *packets = 0;
    *bytes = 0;

    while ((p = strstr(p, "\"packets\"")) != NULL) {
        const char *colon = strchr(p, ':');
        if (!colon)
            break;
        (*flows)++;
        *packets += strtoull(colon + 1, NULL, 10);
        p = colon + 1;
    }

    p = json;
    while ((p = strstr(p, "\"bytes\"")) != NULL) {
        const char *colon = strchr(p, ':');
        if (!colon)
            break;
        *bytes += strtoull(colon + 1, NULL, 10);
        p = colon + 1;
    }
}

static int write_signed_evidence(const char *evidence, const unsigned char *sig, size_t sig_len)
{
    const char *log_path = getenv("IMMUTABLE_AUDIT_LOG");
    if (!log_path)
        log_path = "./immutable_audit.log";

    FILE *fp = fopen(log_path, "a");
    if (!fp)
        return -1;

    char *sig_b64 = calloc(4 * ((sig_len + 2) / 3) + 1, 1);
    if (!sig_b64) {
        fclose(fp);
        return -1;
    }
    EVP_EncodeBlock((unsigned char *)sig_b64, sig, (int)sig_len);

    time_t now = time(NULL);
    fprintf(fp, "{\"ts\":%ld,\"evidence\":%s,\"signature_b64\":\"%s\"}\n", now, evidence, sig_b64);
    free(sig_b64);
    fclose(fp);
    return 0;
}

static int sign_and_log(const char *evidence)
{
    unsigned char digest[SHA256_DIGEST_LENGTH];
    if (!SHA256((const unsigned char *)evidence, strlen(evidence), digest))
        return -1;

    if (pkcs11_initialize_from_env() != 0)
        return -1;

    int rc = sign_log_hsm(digest, sizeof(digest));
    if (rc == 0) {
        size_t sig_len = 0;
        const unsigned char *sig = pkcs11_last_signature(&sig_len);
        if (!sig || sig_len == 0 || write_signed_evidence(evidence, sig, sig_len) != 0)
            rc = -1;
    }

    pkcs11_cleanup();
    return rc;
}

static void *metrics_server(void *arg)
{
    const char *addr = arg ? (const char *)arg : "9108";
    int port = atoi(addr);

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return NULL;

    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in sa = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port), .sin_addr.s_addr = htonl(INADDR_ANY)};
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0 || listen(fd, 8) != 0) {
        close(fd);
        return NULL;
    }

    while (g_running) {
        int cfd = accept(fd, NULL, NULL);
        if (cfd < 0)
            continue;

        char req[512];
        ssize_t rcv = read(cfd, req, sizeof(req));
        (void)rcv;

        struct metrics_snapshot local;
        pthread_mutex_lock(&g_metrics_mu);
        local = g_metrics;
        pthread_mutex_unlock(&g_metrics_mu);

        char body[1024];
        int body_len = snprintf(body, sizeof(body),
            "# HELP ato_flow_records Number of aggregated flow records\n"
            "# TYPE ato_flow_records gauge\n"
            "ato_flow_records %lu\n"
            "# HELP ato_packets_total Aggregated packet count\n"
            "# TYPE ato_packets_total counter\n"
            "ato_packets_total %llu\n"
            "# HELP ato_bytes_total Aggregated byte count\n"
            "# TYPE ato_bytes_total counter\n"
            "ato_bytes_total %llu\n"
            "# HELP ato_signed_batches Total successfully signed evidence batches\n"
            "# TYPE ato_signed_batches counter\n"
            "ato_signed_batches %llu\n"
            "# HELP ato_signing_failures Total signing failures\n"
            "# TYPE ato_signing_failures counter\n"
            "ato_signing_failures %llu\n",
            local.flow_records,
            local.packets_total,
            local.bytes_total,
            local.signed_batches,
            local.signing_failures);

        char header[256];
        int hdr_len = snprintf(header, sizeof(header),
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: %d\r\n\r\n",
            body_len);

        ssize_t w1 = write(cfd, header, (size_t)hdr_len);
        ssize_t w2 = write(cfd, body, (size_t)body_len);
        (void)w1;
        (void)w2;
        close(cfd);
    }

    close(fd);
    return NULL;
}

int main(void)
{
    signal(SIGINT, stop_handler);
    signal(SIGTERM, stop_handler);

    const char *metrics_port = getenv("DISPATCHER_METRICS_PORT");
    if (!metrics_port)
        metrics_port = "9108";

    pthread_t tid;
    if (pthread_create(&tid, NULL, metrics_server, (void *)metrics_port) != 0) {
        fprintf(stderr, "failed to start metrics server\n");
        return 1;
    }

    while (g_running) {
        char json[EVIDENCE_MAX];
        int rc = read_command_output("bpftool -j map dump pinned /sys/fs/bpf/flow_v4_map 2>/dev/null", json, sizeof(json));
        if (rc != 0) {
            snprintf(json, sizeof(json), "[]");
        }

        unsigned long flows = 0;
        unsigned long long packets = 0;
        unsigned long long bytes = 0;
        extract_totals_from_json(json, &flows, &packets, &bytes);

        unsigned char src[16] = {192, 168, 10, 45};
        unsigned char dst[16] = {172, 16, 1, 99};
        mask_ip_pair(src, dst);

        char evidence[EVIDENCE_MAX];
        snprintf(evidence, sizeof(evidence),
                 "{\"flows\":%lu,\"packets\":%llu,\"bytes\":%llu,\"masked_src\":\"%u.%u.%u.0\",\"masked_dst\":\"%u.%u.%u.0\"}",
                 flows, packets, bytes, src[0], src[1], src[2], dst[0], dst[1], dst[2]);

        pthread_mutex_lock(&g_metrics_mu);
        g_metrics.flow_records = flows;
        g_metrics.packets_total = packets;
        g_metrics.bytes_total = bytes;
        pthread_mutex_unlock(&g_metrics_mu);

        if (sign_and_log(evidence) == 0) {
            pthread_mutex_lock(&g_metrics_mu);
            g_metrics.signed_batches += 1;
            pthread_mutex_unlock(&g_metrics_mu);
        } else {
            pthread_mutex_lock(&g_metrics_mu);
            g_metrics.signing_failures += 1;
            pthread_mutex_unlock(&g_metrics_mu);
        }

        sleep(5);
    }

    pthread_join(tid, NULL);
    return 0;
}
