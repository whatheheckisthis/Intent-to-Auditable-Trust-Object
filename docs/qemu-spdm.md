# QEMU + libspdm setup

```bash
git clone https://github.com/DMTF/libspdm
cd libspdm && mkdir build && cd build
cmake .. -DARCH=x64 -DTOOLCHAIN=GCC -DTARGET=Debug \
         -DCRYPTO=mbedtls -DDISABLE_TESTS=1
make spdm_responder_emu

./bin/spdm_responder_emu \
  --trans socket \
  --socket-port 0 \
  --socket-path /tmp/iato-spdm.sock

-device spdm-socket,socket=/tmp/iato-spdm.sock
```
