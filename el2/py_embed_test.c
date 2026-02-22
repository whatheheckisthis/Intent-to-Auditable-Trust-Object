#include <stdint.h>
#include <stdio.h>

#include "py_embed.h"

int main(void) {
    uint8_t cred[149] = {0};
    uint64_t base = 0, limit = 0;
    uint8_t perms = 0;
    int rc;

    rc = iato_py_embed_init();
    if (rc != IATO_PY_OK) { return 1; }

    rc = iato_py_validate_credential(cred, sizeof(cred), 7U, &base, &limit, &perms);
    if (rc != IATO_PY_OK) { return 1; }
    if ((base != 0x1000ULL) || (limit != 0x3000ULL) || (perms != 3U)) { return 1; }

    iato_py_embed_shutdown();
    puts("py_embed_test: ok");
    return 0;
}
