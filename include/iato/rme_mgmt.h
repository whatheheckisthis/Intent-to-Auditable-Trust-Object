#ifndef IATO_RME_MGMT_H
#define IATO_RME_MGMT_H

#include <stdint.h>

typedef struct rme_realm {
    uint64_t realm_id;
    uint64_t vmid;
    uint64_t flags;
} rme_realm_t;

#endif
