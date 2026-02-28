#include "tinycbor.h"

#include <string.h>

static CborError put_u8(CborEncoder *e, uint8_t v) {
    if (e->ptr >= e->end) {
        return CborErrorOutOfMemory;
    }
    *e->ptr++ = v;
    return CborNoError;
}

static CborError put_be(CborEncoder *e, uint64_t v, size_t n) {
    if ((size_t)(e->end - e->ptr) < n) {
        return CborErrorOutOfMemory;
    }
    for (size_t i = 0; i < n; ++i) {
        e->ptr[n - 1 - i] = (uint8_t)(v & 0xff);
        v >>= 8;
    }
    e->ptr += n;
    return CborNoError;
}

static CborError encode_type_val(CborEncoder *e, uint8_t major, uint64_t v) {
    if (v <= 23) {
        return put_u8(e, (uint8_t)((major << 5) | (uint8_t)v));
    }
    if (v <= 0xff) {
        CborError err = put_u8(e, (uint8_t)((major << 5) | 24));
        return err != CborNoError ? err : put_u8(e, (uint8_t)v);
    }
    if (v <= 0xffff) {
        CborError err = put_u8(e, (uint8_t)((major << 5) | 25));
        return err != CborNoError ? err : put_be(e, v, 2);
    }
    if (v <= 0xffffffffu) {
        CborError err = put_u8(e, (uint8_t)((major << 5) | 26));
        return err != CborNoError ? err : put_be(e, v, 4);
    }
    CborError err = put_u8(e, (uint8_t)((major << 5) | 27));
    return err != CborNoError ? err : put_be(e, v, 8);
}

CborError cbor_encoder_init(CborEncoder *encoder, uint8_t *buffer, size_t size, int flags) {
    (void)flags;
    encoder->ptr = buffer;
    encoder->end = buffer + size;
    return CborNoError;
}

CborError cbor_encoder_create_map(CborEncoder *parent, CborEncoder *map, size_t length) {
    if (length == CborIndefiniteLength) {
        return CborErrorUnknownLength;
    }
    CborError err = encode_type_val(parent, 5, (uint64_t)length);
    if (err != CborNoError) {
        return err;
    }
    map->ptr = parent->ptr;
    map->end = parent->end;
    return CborNoError;
}

CborError cbor_encoder_close_container(CborEncoder *parent, CborEncoder *container) {
    parent->ptr = container->ptr;
    return CborNoError;
}

CborError cbor_encode_text_stringz(CborEncoder *encoder, const char *str) {
    size_t len = strlen(str);
    CborError err = encode_type_val(encoder, 3, len);
    if (err != CborNoError) {
        return err;
    }
    if ((size_t)(encoder->end - encoder->ptr) < len) {
        return CborErrorOutOfMemory;
    }
    memcpy(encoder->ptr, str, len);
    encoder->ptr += len;
    return CborNoError;
}

CborError cbor_encode_uint(CborEncoder *encoder, uint64_t value) {
    return encode_type_val(encoder, 0, value);
}

CborError cbor_encode_byte_string(CborEncoder *encoder, const uint8_t *data, size_t len) {
    CborError err = encode_type_val(encoder, 2, len);
    if (err != CborNoError) {
        return err;
    }
    if ((size_t)(encoder->end - encoder->ptr) < len) {
        return CborErrorOutOfMemory;
    }
    memcpy(encoder->ptr, data, len);
    encoder->ptr += len;
    return CborNoError;
}

size_t cbor_encoder_get_buffer_size(const CborEncoder *encoder, const uint8_t *buffer) {
    return (size_t)(encoder->ptr - buffer);
}
