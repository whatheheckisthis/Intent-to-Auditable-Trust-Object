#include <fcntl.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "pkcs11_signer.h"

typedef struct Command {
    const char *name;
    int (*handler)(int argc, char **argv);
} Command;

extern void fast_clear_buffer(void *buf, size_t len);

static void clear_heap_string(char *s) {
    if (s == NULL) {
        return;
    }

    fast_clear_buffer(s, strlen(s));
    free(s);
}

static void scrub_cli_args(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] != NULL) {
            fast_clear_buffer(argv[i], strlen(argv[i]));
        }
    }
}

static int capture_parser_json(const char *policy_file, char **json_out) {
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        perror("pipe");
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }

    if (pid == 0) {
        char *args[] = {"python3", "iam_parser.py", "--file", (char *)policy_file, NULL};
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        execvp("python3", args);
        perror("execvp");
        _exit(127);
    }

    close(pipefd[1]);

    size_t cap = 4096;
    size_t used = 0;
    char *json = calloc(cap, 1);
    if (json == NULL) {
        close(pipefd[0]);
        fprintf(stderr, "Out of memory while capturing parser output.\n");
        return -1;
    }

    while (1) {
        if (used + 1024 >= cap) {
            size_t next_cap = cap * 2;
            char *next = realloc(json, next_cap);
            if (next == NULL) {
                fprintf(stderr, "Out of memory expanding parser output buffer.\n");
                clear_heap_string(json);
                close(pipefd[0]);
                return -1;
            }
            json = next;
            cap = next_cap;
        }

        ssize_t n = read(pipefd[0], json + used, cap - used - 1);
        if (n < 0) {
            perror("read");
            clear_heap_string(json);
            close(pipefd[0]);
            return -1;
        }

        if (n == 0) {
            break;
        }

        used += (size_t)n;
    }

    close(pipefd[0]);
    json[used] = '\0';

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        clear_heap_string(json);
        return -1;
    }

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "IAM parser failed with status %d\n", WEXITSTATUS(status));
        clear_heap_string(json);
        return -1;
    }

    *json_out = json;
    return 0;
}

static int base64_encode(const unsigned char *data, size_t data_len, char **out) {
    size_t encoded_len = 4 * ((data_len + 2) / 3);
    char *encoded = calloc(encoded_len + 1, 1);
    if (encoded == NULL) {
        return -1;
    }

    int written = EVP_EncodeBlock((unsigned char *)encoded, data, (int)data_len);
    if (written <= 0) {
        free(encoded);
        return -1;
    }

    encoded[written] = '\0';
    *out = encoded;
    return 0;
}

static int build_signed_payload(const char *json, const char *sig_b64, char **payload_out) {
    const char *template_str = "{\"result\":%s,\"signature\":\"%s\"}";
    size_t needed = snprintf(NULL, 0, template_str, json, sig_b64) + 1;
    char *payload = calloc(needed, 1);
    if (payload == NULL) {
        return -1;
    }

    snprintf(payload, needed, template_str, json, sig_b64);
    *payload_out = payload;
    return 0;
}

static int upload_payload(const char *payload) {
    const char *rest_url = getenv("REST_URL");
    const char *api_key = getenv("API_KEY");

    if (rest_url == NULL || api_key == NULL) {
        fprintf(stderr, "REST_URL and API_KEY are required for signed upload.\n");
        return -1;
    }

    char path[] = "/tmp/dispatcher_payload_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        perror("mkstemp");
        return -1;
    }

    size_t payload_len = strlen(payload);
    if (write(fd, payload, payload_len) != (ssize_t)payload_len) {
        perror("write");
        close(fd);
        unlink(path);
        return -1;
    }
    close(fd);

    size_t auth_len = strlen("Authorization: Bearer ") + strlen(api_key) + 1;
    char *auth_header = calloc(auth_len, 1);
    if (auth_header == NULL) {
        unlink(path);
        return -1;
    }
    snprintf(auth_header, auth_len, "Authorization: Bearer %s", api_key);

    char *curl_args[] = {
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "-X",
        "POST",
        (char *)rest_url,
        "-H",
        auth_header,
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        NULL,
        NULL,
    };

    size_t data_arg_len = strlen("@") + strlen(path) + 1;
    char *data_arg = calloc(data_arg_len, 1);
    if (data_arg == NULL) {
        clear_heap_string(auth_header);
        unlink(path);
        return -1;
    }
    snprintf(data_arg, data_arg_len, "@%s", path);
    curl_args[12] = data_arg;

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        clear_heap_string(auth_header);
        clear_heap_string(data_arg);
        unlink(path);
        return -1;
    }

    if (pid == 0) {
        execvp("curl", curl_args);
        perror("execvp");
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        clear_heap_string(auth_header);
        clear_heap_string(data_arg);
        unlink(path);
        return -1;
    }

    clear_heap_string(auth_header);
    clear_heap_string(data_arg);
    unlink(path);

    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        return 0;
    }

    fprintf(stderr, "curl upload failed with status %d\n", WEXITSTATUS(status));
    return -1;
}

static int handle_parse_iam(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: parse-iam <policy.json>\n");
        return 1;
    }

    char *json = NULL;
    char *sig_b64 = NULL;
    char *payload = NULL;
    int rc = 2;

    if (capture_parser_json(argv[1], &json) != 0) {
        goto cleanup;
    }

    unsigned char digest[SHA256_DIGEST_LENGTH];
    if (SHA256((unsigned char *)json, strlen(json), digest) == NULL) {
        fprintf(stderr, "Failed to hash parser JSON output.\n");
        goto cleanup;
    }

    if (pkcs11_initialize_from_env() != 0) {
        goto cleanup;
    }

    if (sign_log_hsm(digest, sizeof(digest)) != 0) {
        goto cleanup;
    }

    size_t sig_len = 0;
    const unsigned char *sig = pkcs11_last_signature(&sig_len);
    if (sig == NULL || sig_len == 0) {
        fprintf(stderr, "HSM returned an empty signature.\n");
        goto cleanup;
    }

    if (base64_encode(sig, sig_len, &sig_b64) != 0) {
        fprintf(stderr, "Failed to Base64 encode signature.\n");
        goto cleanup;
    }

    if (build_signed_payload(json, sig_b64, &payload) != 0) {
        fprintf(stderr, "Failed to build signed JSON payload.\n");
        goto cleanup;
    }

    printf("%s\n", payload);

    if (upload_payload(payload) != 0) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    pkcs11_cleanup();
    clear_heap_string(sig_b64);
    clear_heap_string(payload);
    clear_heap_string(json);
    return rc;
}

static void print_usage(const char *program_name) {
    fprintf(stderr, "Usage: %s <command> [args]\n", program_name);
    fprintf(stderr, "Commands:\n");
    fprintf(stderr, "  parse-iam <policy.json>\n");
}

static Command COMMANDS[] = {
    {"parse-iam", handle_parse_iam},
};

static size_t command_count(void) {
    return sizeof(COMMANDS) / sizeof(COMMANDS[0]);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *token = argv[1];

    for (size_t i = 0; i < command_count(); ++i) {
        if (strcmp(token, COMMANDS[i].name) == 0) {
            int rc = COMMANDS[i].handler(argc - 1, argv + 1);
            scrub_cli_args(argc, argv);
            return rc;
        }
    }

    fprintf(stderr, "Unknown command: %s\n", token);
    print_usage(argv[0]);
    scrub_cli_args(argc, argv);
    return 1;
}
