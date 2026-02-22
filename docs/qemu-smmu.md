# QEMU SMMUv3 flags

```bash
-machine virt,iommu=smmuv3 \
-device virtio-net-pci,iommu_platform=on,ats=on \
-device ioh3420,id=pcie.1,bus=pcie.0

cat /sys/bus/platform/drivers/arm-smmu-v3/*/iommu/*/type
cat /proc/iomem | grep smmu
```
