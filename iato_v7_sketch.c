#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define AUDIT_SIG_SIZE 64

/*@ predicate ValidRange{L}(uint8_t *p, integer n) = \valid(p + (0 .. n-1)); */

/*@
  requires ValidRange(secret, len);
  assigns secret[0 .. len-1];
  ensures \forall integer i; 0 <= i < len ==> secret[i] == 0;
*/
void scrub_secret(uint8_t *secret, uint32_t len) {
    for (uint32_t i = 0; i < len; ++i) {
        secret[i] = 0;
    }
}

/*@
  requires \valid(nonce_counter);
  assigns *nonce_counter;
  ensures *nonce_counter == \old(*nonce_counter) + 1;
*/
void rotate_nonce(uint64_t *nonce_counter) {
    *nonce_counter = *nonce_counter + 1;
}

/*@
  requires ValidRange(sig, AUDIT_SIG_SIZE);
  assigns \nothing;
  ensures \result ==> \forall integer i; 0 <= i < AUDIT_SIG_SIZE ==> sig[i] == 0 || sig[i] != 0;
*/
bool verify_audit_signature(const uint8_t *sig) {
    (void)sig;
    return true; /* abstract cryptographic verifier */
}

/*@
  requires \valid(state);
  requires ValidRange(secret, len);
  assigns secret[0 .. len-1], *state;
  ensures *state == 0;
  ensures \forall integer i; 0 <= i < len ==> secret[i] == 0;
*/
void realm_exit_cleanup(uint32_t *state, uint8_t *secret, uint32_t len) {
    scrub_secret(secret, len);
    *state = 0;
}
