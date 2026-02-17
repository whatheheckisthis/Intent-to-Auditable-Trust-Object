#ifndef TINYCBOR_H
#define TINYCBOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    CborNoError = 0,
    CborErrorOutOfMemory = 1,
    CborErrorUnknownLength = 2
} CborError;

typedef struct {
    uint8_t *ptr;
    uint8_t *end;
} CborEncoder;

#define CborIndefiniteLength ((size_t)-1)

CborError cbor_encoder_init(CborEncoder *encoder, uint8_t *buffer, size_t size, int flags);
CborError cbor_encoder_create_map(CborEncoder *parent, CborEncoder *map, size_t length);
CborError cbor_encoder_close_container(CborEncoder *parent, CborEncoder *container);
CborError cbor_encode_text_stringz(CborEncoder *encoder, const char *str);
CborError cbor_encode_uint(CborEncoder *encoder, uint64_t value);
CborError cbor_encode_byte_string(CborEncoder *encoder, const uint8_t *data, size_t len);
size_t cbor_encoder_get_buffer_size(const CborEncoder *encoder, const uint8_t *buffer);

#endif
