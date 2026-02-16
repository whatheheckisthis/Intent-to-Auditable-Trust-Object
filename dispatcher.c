#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

extern void fast_mask(void *buf, size_t len);

static int run_parser_and_capture(const char *policy_file, char *out, size_t out_size, int *parser_status) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "python3 iam_parser.py --file \"%s\"", policy_file);

    FILE *fp = POPEN(cmd, "r");
    if (!fp) {
        fprintf(stderr, "Failed to start parser process.\n");
        *parser_status = 2;
        return 2;
    }

    size_t used = 0;
    while (used + 1 < out_size) {
        size_t n = fread(out + used, 1, out_size - used - 1, fp);
        if (n == 0) {
            break;
        }
        used += n;
    }
    out[used] = '\0';

    int exit_code = PCLOSE(fp);
    *parser_status = exit_code;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <policy.json>\n", argv[0]);
        return 1;
    }

    const char *policy_file = argv[1];
    char parser_output[65536];

    int parser_status = 0;
    int rc = run_parser_and_capture(policy_file, parser_output, sizeof(parser_output), &parser_status);
    if (rc != 0) {
        return rc;
    }

    printf("%s", parser_output);

    fast_mask(parser_output, strlen(parser_output));

    if (parser_status != 0) {
        fprintf(stderr, "Parser exited with non-zero status: %d\n", parser_status);
        return 3;
    }

    return 0;
}
