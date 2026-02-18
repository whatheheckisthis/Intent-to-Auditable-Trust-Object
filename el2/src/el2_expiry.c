#include "el2_expiry.h"
#include "el2_binding_table.h"

static el2_time_t g_mock_time_ns;
static const el2_time_t g_interval_ns = 1000000000ULL;

void el2_set_mock_time_ns(el2_time_t t) { g_mock_time_ns = t; }

el2_time_t el2_current_time_ns(void) {
    return g_mock_time_ns;
}

void el2_expiry_init(void) {
    (void)g_interval_ns;
}

void el2_expiry_handler(void) {
    binding_entry_t *tbl = el2_binding_table_raw();
    el2_time_t now = el2_current_time_ns();
    for (size_t i = 0; i < MAX_BINDINGS; i++) {
        if (tbl[i].status == BINDING_ACTIVE && tbl[i].expiry_ns <= now) {
            el2_binding_fault(tbl[i].stream_id);
        }
    }
}
