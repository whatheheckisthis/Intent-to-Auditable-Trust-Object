#include "expiry_sweep.h"

extern uint64_t iato_wall_time_s(void);

iato_binding_entry_t iato_binding_table[IATO_SMMU_MAX_STREAMS];
static volatile uint32_t iato_binding_lock;

static void lock_acquire(volatile uint32_t *l) {
    while (__sync_lock_test_and_set(l, 1U) != 0U) {
    }
}

static void lock_release(volatile uint32_t *l) {
    __sync_lock_release(l);
}

int iato_expiry_sweep(uint64_t elapsed_ns) {
    uint32_t to_fault[IATO_SMMU_MAX_STREAMS];
    uint32_t n = 0U;
    uint32_t i;
    uint64_t now;
    (void)elapsed_ns;

    now = iato_wall_time_s();
    lock_acquire(&iato_binding_lock);
    for (i = 0U; i < IATO_SMMU_MAX_STREAMS; ++i) {
        if ((iato_binding_table[i].status == IATO_STATUS_ACTIVE) &&
            (iato_binding_table[i].expiry_unix_s <= now)) {
            to_fault[n++] = iato_binding_table[i].stream_id;
        }
    }
    lock_release(&iato_binding_lock);

    for (i = 0U; i < n; ++i) {
        #ifdef IATO_MOCK_SMMU
        extern int iato_mock_smmu_fault_ste(uint32_t stream_id);
        (void)iato_mock_smmu_fault_ste(to_fault[i]);
#else
        (void)iato_smmu_fault_ste(to_fault[i]);
#endif
    }

    lock_acquire(&iato_binding_lock);
    for (i = 0U; i < n; ++i) {
        iato_binding_table[to_fault[i]].status = IATO_STATUS_AWAITING_POLICY;
    }
    lock_release(&iato_binding_lock);
    return (int)n;
}
