#ifndef EL2_EXPIRY_H
#define EL2_EXPIRY_H

#include "el2_types.h"

void el2_expiry_init(void);
void el2_expiry_handler(void);
el2_time_t el2_current_time_ns(void);

/* test hook */
void el2_set_mock_time_ns(el2_time_t t);

#endif
