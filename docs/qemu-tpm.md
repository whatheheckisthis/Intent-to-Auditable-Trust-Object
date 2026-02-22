# QEMU + swtpm setup

```bash
mkdir -p /tmp/swtpm
swtpm socket \
  --tpmstate dir=/tmp/swtpm \
  --ctrl type=unixio,path=/tmp/swtpm/swtpm-sock \
  --log level=20 \
  --tpm2 \
  --daemon

# QEMU flags
-chardev socket,id=chrtpm,path=/tmp/swtpm/swtpm-sock \
-tpmdev emulator,id=tpm0,chardev=chrtpm \
-device tpm-tis,tpmdev=tpm0
```
