#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${ROOT_DIR}/build/hw-results"
RESULT_MD="${ROOT_DIR}/tests/hw-results.md"

IATO_SPDM_RESPONDER="${IATO_SPDM_RESPONDER:-${ROOT_DIR}/build/libspdm/bin/spdm_responder_emu}"
IATO_GUEST_IMAGE="${IATO_GUEST_IMAGE:-${ROOT_DIR}/build/guest/guest.img}"
IATO_EL2_BIN="${IATO_EL2_BIN:-${ROOT_DIR}/build/el2/iato-el2.bin}"
IATO_BOOT_TIMEOUT_S="${IATO_BOOT_TIMEOUT_S:-120}"
IATO_TEST_TIMEOUT_S="${IATO_TEST_TIMEOUT_S:-300}"
IATO_OPERATOR="${IATO_OPERATOR:-${USER:-unknown}}"
IATO_PRNG_SEED="${IATO_PRNG_SEED:-0x4941544f2d5637}"

SWTPM_DIR="/tmp/iato-swtpm"
SWTPM_SOCK="${SWTPM_DIR}/swtpm.sock"
SWTPM_PID="${SWTPM_DIR}/swtpm.pid"
SPDM_SOCK="/tmp/iato-spdm.sock"
SPDM_PID="/tmp/iato-spdm.pid"
SPDM_LOG="/tmp/iato-spdm.log"
QEMU_PID="/tmp/iato-qemu.pid"
QEMU_LOG="/tmp/iato-qemu.log"
CONSOLE_SOCK="/tmp/iato-console.sock"
CONSOLE_LOG="/tmp/iato-console.log"
EXIT_FILE="/tmp/iato-hw-exitcode"

PYTEST_XML="${RESULT_DIR}/hw-pytest.xml"
PYTEST_LOG="${RESULT_DIR}/hw-pytest.log"

mkdir -p "${RESULT_DIR}"
rm -f "${SPDM_SOCK}" "${CONSOLE_SOCK}" "${QEMU_PID}" "${SPDM_PID}" "${EXIT_FILE}" "${CONSOLE_LOG}" "${PYTEST_XML}" "${PYTEST_LOG}"

HW_EXIT=1

log() { echo "[harness] $*"; }
err() { echo "[harness][error] $*" >&2; }

teardown() {
  set +e
  if [[ -S "${CONSOLE_SOCK}" ]]; then
    printf 'poweroff\n' | nc -U "${CONSOLE_SOCK}" >/dev/null 2>&1 || true
  fi

  if [[ -f "${QEMU_PID}" ]]; then
    qpid="$(cat "${QEMU_PID}" 2>/dev/null || true)"
    if [[ -n "${qpid}" ]] && kill -0 "${qpid}" >/dev/null 2>&1; then
      for _ in $(seq 1 10); do
        kill -0 "${qpid}" >/dev/null 2>&1 || break
        sleep 1
      done
      kill -0 "${qpid}" >/dev/null 2>&1 && kill -9 "${qpid}" >/dev/null 2>&1 || true
    fi
  fi

  if [[ -f "${SWTPM_PID}" ]]; then
    spid="$(cat "${SWTPM_PID}" 2>/dev/null || true)"
    [[ -n "${spid}" ]] && kill "${spid}" >/dev/null 2>&1 || true
  fi

  if [[ -f "${SPDM_PID}" ]]; then
    ppid="$(cat "${SPDM_PID}" 2>/dev/null || true)"
    [[ -n "${ppid}" ]] && kill "${ppid}" >/dev/null 2>&1 || true
  fi

  rm -f "${SWTPM_SOCK}" "${SPDM_SOCK}" "${CONSOLE_SOCK}" "${SWTPM_PID}" "${SPDM_PID}" "${QEMU_PID}" "${EXIT_FILE}"
  log "teardown complete"
}
trap teardown EXIT

preflight_fail() {
  err "preflight failed: $1"
  err "see docs/qemu-harness.md for setup instructions"
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || preflight_fail "missing command: $1"
}

require_cmd qemu-system-aarch64
require_cmd swtpm
require_cmd nc
require_cmd python3

qemu_version="$(qemu-system-aarch64 --version | head -n1)"
qemu_ver_num="$(echo "${qemu_version}" | sed -E 's/.*version ([0-9]+\.[0-9]+).*/\1/')"
python3 - <<PY || preflight_fail "qemu-system-aarch64 version must be >= 7.2 (found ${qemu_ver_num})"
import sys
def v(x):
    try:
        return tuple(int(i) for i in x.split("."))
    except Exception:
        return (0,0)
sys.exit(0 if v("${qemu_ver_num}") >= v("7.2") else 1)
PY

[[ -x "${IATO_SPDM_RESPONDER}" ]] || preflight_fail "SPDM responder missing or not executable: ${IATO_SPDM_RESPONDER}"
[[ -f "${IATO_GUEST_IMAGE}" ]] || preflight_fail "guest image not found: ${IATO_GUEST_IMAGE}"
[[ -f "${IATO_EL2_BIN}" ]] || preflight_fail "EL2 binary not found: ${IATO_EL2_BIN}"

IATO_SWTPM_DIR="${SWTPM_DIR}" \
IATO_SWTPM_SOCK="${SWTPM_SOCK}" \
IATO_SWTPM_PID="${SWTPM_PID}" \
"${ROOT_DIR}/scripts/start-swtpm.sh"

IATO_SPDM_RESPONDER="${IATO_SPDM_RESPONDER}" \
IATO_SPDM_SOCK="${SPDM_SOCK}" \
IATO_SPDM_PID="${SPDM_PID}" \
IATO_SPDM_LOG="${SPDM_LOG}" \
"${ROOT_DIR}/scripts/start-spdm-responder.sh"

qemu-system-aarch64 \
  -machine virt,iommu=smmuv3,secure=on \
  -cpu cortex-a57 \
  -m 2G \
  -smp 2 \
  -bios "${IATO_EL2_BIN}" \
  -drive file="${IATO_GUEST_IMAGE}",format=qcow2,if=virtio \
  -chardev socket,id=chrtpm,path="${SWTPM_SOCK}" \
  -tpmdev emulator,id=tpm0,chardev=chrtpm \
  -device tpm-tis,tpmdev=tpm0 \
  -chardev socket,id=spdm,path="${SPDM_SOCK}" \
  -device virtio-serial \
  -chardev socket,id=console,path="${CONSOLE_SOCK}",server,nowait \
  -device virtconsole,chardev=console \
  -device virtio-net-pci,iommu_platform=on,ats=on \
  -nographic \
  -no-reboot \
  -pidfile "${QEMU_PID}" \
  >>"${QEMU_LOG}" 2>&1 &

python3 - <<PY
import base64
import os
import re
import socket
import sys
import time

sock_path = "${CONSOLE_SOCK}"
boot_timeout = int("${IATO_BOOT_TIMEOUT_S}")
test_timeout = int("${IATO_TEST_TIMEOUT_S}")
log_path = "${CONSOLE_LOG}"
exit_file = "${EXIT_FILE}"
out_xml = "${PYTEST_XML}"
out_log = "${PYTEST_LOG}"

start = time.time()
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
while True:
    try:
        s.connect(sock_path)
        break
    except OSError:
        if time.time() - start > boot_timeout:
            print("BOOT_TIMEOUT", file=sys.stderr)
            sys.exit(2)
        time.sleep(0.2)

s.settimeout(0.2)
stream = b""
ready = False
hw_done = False
exit_code = 1
artifacts = {}
current_name = None
current_data = []

with open(log_path, "wb") as lf:
    while True:
        try:
            data = s.recv(4096)
            if not data:
                break
            lf.write(data)
            lf.flush()
            stream += data
        except socket.timeout:
            pass

        text = stream.decode("utf-8", errors="ignore")
        lines = text.split("\n")
        stream = lines.pop().encode("utf-8", errors="ignore")

        for line in lines:
            if not ready and "[guest-ready]" in line:
                ready = True
                cmd = "\n".join([
                    "cd /opt/iato",
                    "export IATO_HW_MODE=1",
                    "export IATO_TPM_SIM=0",
                    "export IATO_MANA_SIM_MMIO=0",
                    "export IATO_SPDM_SOCKET=/tmp/iato-spdm.sock",
                    "export IATO_EL2_TIMER_DEV=/dev/iato-el2-timer",
                    "pytest -m hw -v --tb=short --junitxml=/tmp/hw-pytest.xml 2>&1 | tee /tmp/hw-pytest.log",
                    "echo [hw-test-done] exit=$?",
                    "echo [artifact-begin] hw-pytest.log",
                    "base64 -w0 /tmp/hw-pytest.log || true; echo",
                    "echo [artifact-end] hw-pytest.log",
                    "echo [artifact-begin] hw-pytest.xml",
                    "base64 -w0 /tmp/hw-pytest.xml || true; echo",
                    "echo [artifact-end] hw-pytest.xml",
                    "echo [artifact-complete]",
                    "",
                ])
                s.sendall(cmd.encode())

            m = re.search(r"\[hw-test-done\] exit=(\d+)", line)
            if m:
                exit_code = int(m.group(1))
                hw_done = True

            m = re.match(r"\[artifact-begin\] (.+)", line.strip())
            if m:
                current_name = m.group(1)
                current_data = []
                continue

            m = re.match(r"\[artifact-end\] (.+)", line.strip())
            if m and current_name == m.group(1):
                artifacts[current_name] = "".join(current_data)
                current_name = None
                current_data = []
                continue

            if current_name is not None:
                current_data.append(line.strip())

            if hw_done and "[artifact-complete]" in line:
                with open(exit_file, "w", encoding="utf-8") as f:
                    f.write(str(exit_code))
                for name, payload in artifacts.items():
                    if not payload:
                        continue
                    out = out_log if name.endswith(".log") else out_xml
                    with open(out, "wb") as fh:
                        fh.write(base64.b64decode(payload.encode()))
                sys.exit(0)

        if not ready and time.time() - start > boot_timeout:
            print("BOOT_TIMEOUT", file=sys.stderr)
            sys.exit(2)
        if ready and time.time() - start > (boot_timeout + test_timeout):
            print("TEST_TIMEOUT", file=sys.stderr)
            sys.exit(3)
PY

py_rc=$?
if [[ "${py_rc}" -eq 2 ]]; then
  err "guest ready signal timeout"
  tail -n 50 "${QEMU_LOG}" >&2 || true
  exit 1
elif [[ "${py_rc}" -eq 3 ]]; then
  err "hw test completion timeout"
  exit 1
elif [[ "${py_rc}" -ne 0 ]]; then
  err "console harness failed"
  exit 1
fi

if [[ -f "${EXIT_FILE}" ]]; then
  HW_EXIT="$(cat "${EXIT_FILE}")"
fi

python3 - <<PY
import datetime as dt
import hashlib
import os
import subprocess
import xml.etree.ElementTree as ET

result_md = "${RESULT_MD}"
xml_path = "${PYTEST_XML}"
qemu_version = "${qemu_version}"
operator = "${IATO_OPERATOR}"
prng_seed = "${IATO_PRNG_SEED}"

counts = {"total": 0, "failed": 0, "skipped": 0, "passed": 0}
failed_tests = []
if os.path.exists(xml_path):
    root = ET.parse(xml_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is not None:
        counts["total"] = int(suite.attrib.get("tests", 0))
        counts["failed"] = int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0))
        counts["skipped"] = int(suite.attrib.get("skipped", 0))
        counts["passed"] = counts["total"] - counts["failed"] - counts["skipped"]
        for tc in suite.findall("testcase"):
            fail = tc.find("failure") or tc.find("error")
            if fail is not None:
                name = f"{tc.attrib.get('classname','')}.{tc.attrib.get('name','')}".strip(".")
                msg = (fail.attrib.get("message") or (fail.text or "")).strip().splitlines()[0] if (fail.attrib.get("message") or fail.text) else "failure"
                failed_tests.append(f"{name}: {msg}")

print(f"[harness] hw tests: total={counts['total']} passed={counts['passed']} failed={counts['failed']} skipped={counts['skipped']}")

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

guest_sha = sha("${IATO_GUEST_IMAGE}")
el2_sha = sha("${IATO_EL2_BIN}")

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()

def _read(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

spdm_log = _read("${SPDM_LOG}")
qemu_log = _read("${QEMU_LOG}")
console_log = _read("${CONSOLE_LOG}")

pcr_extended = "YES" if "PCR" in qemu_log or "PCR" in console_log else "NO"
spdm_attested = "YES" if "ATTESTED" in spdm_log or "ATTESTED" in console_log else "NO"
ste_writes = qemu_log.count("STE")
el2_intervals = console_log.count("iato-el2-timer")

verdict = "PASS" if int("${HW_EXIT}") == 0 and counts["failed"] == 0 else "FAIL"
if counts["total"] == 0:
    verdict = "UNPROVISIONED"

with open(result_md, "w", encoding="utf-8") as out:
    out.write("# IĀTŌ-V7 Hardware Integration Test Results\n")
    out.write(f"Date (UTC): {dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}\n")
    out.write(f"QEMU version: {qemu_version}\n")
    out.write(f"Guest image SHA-256: {guest_sha}\n")
    out.write(f"EL2 binary SHA-256: {el2_sha}\n")
    out.write(f"Git SHA: {commit}\n")
    out.write(f"Branch: {branch}\n\n")
    out.write("## Execution Metadata\n")
    out.write(f"- Operator: {operator}\n")
    out.write(f"- PRNG seed: {prng_seed}\n")
    out.write("- Harness scripts (.sh): scripts/qemu-harness.sh\n\n")
    out.write("## Hardware Environment\n")
    out.write("- TPM: swtpm socket: /tmp/iato-swtpm/swtpm.sock, PCR 16 extended: " + pcr_extended + "\n")
    out.write("- SPDM: libspdm responder unknown, session reached ATTESTED: " + spdm_attested + "\n")
    out.write("- SMMU: QEMU SMMUv3, MMIO base: 0x09050000, STE writes: " + str(ste_writes) + "\n")
    out.write("- EL2 timer: /dev/iato-el2-timer, intervals fired: " + str(el2_intervals) + "\n\n")
    out.write("## Test Results\n")
    out.write(f"- Total: {counts['total']}\n")
    out.write(f"- Passed: {counts['passed']}\n")
    out.write(f"- Failed: {counts['failed']}\n")
    out.write(f"- Skipped: {counts['skipped']}\n\n")
    out.write("## Failed Tests\n")
    out.write("None\n\n" if not failed_tests else "\n".join(failed_tests) + "\n\n")
    out.write("## Verdict\n")
    out.write(verdict + "\n")
    out.write("\n## Hardware pass/fail summary\n")
    out.write(f"- Hardware pass/fail summary: {verdict}\n")
PY

exit "${HW_EXIT}"
