#!/usr/bin/env bash
# Build-staging workflow: compile-only validation, no restore/runtime execution.
set -euo pipefail

SEGMENT_MS="${TME_SEGMENT_BUILD_OFFLINE_MS:-8000}"

# shellcheck disable=SC1091
source "$(dirname "$0")/timing_mitigation_engine.sh"

log() { printf '[build-offline] %s\n' "$*"; }
fail() { printf '[build-offline][error] %s\n' "$*" >&2; exit 46; }

tme_run_segment "build-offline.ensure-dotnet" "${TME_SEGMENT_BUILD_ENSURE_MS:-2200}" -- bash "$(dirname "$0")/ensure-dotnet.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/activate-dotnet-offline.sh"

log "EL2 isolation: compile-only mode enabled (no runtime entrypoints invoked)"

lock_file="src/NfcReader/packages.lock.json"
if [[ ! -f "${lock_file}" ]]; then
  printf '[build][error] packages.lock.json not found at src/NfcReader/packages.lock.json\n' >&2
  printf '[build][error] to fix: on a machine with internet access run:\n' >&2
  printf '[build][error]   dotnet restore src/NfcReader/NfcReader.sln --use-lock-file\n' >&2
  printf '[build][error]   then commit packages.lock.json to the repository\n' >&2
  printf '[build][error] then populate the local NuGet cache by running:\n' >&2
  printf '[build][error]   dotnet restore src/NfcReader/NfcReader.sln\n' >&2
  printf '[build][error]   and copy $HOME/.nuget/packages to /opt/nuget-cache or $HOME/.nuget/packages on the offline machine\n' >&2
  exit 1
fi

cache_candidates=("${NUGET_PACKAGES:-}" "/opt/nuget-cache" "${HOME}/.nuget/packages")
resolved_cache=""
for cache_dir in "${cache_candidates[@]}"; do
  [[ -n "${cache_dir}" && -d "${cache_dir}" ]] || continue
  if find "${cache_dir}" -mindepth 1 -maxdepth 1 -type d | read -r _; then
    resolved_cache="${cache_dir}"
    break
  fi
done

if [[ -z "${resolved_cache}" ]]; then
  printf '[build][error] NuGet package cache is empty\n' >&2
  printf '[build][error] to fix: copy a populated NuGet cache to one of:\n' >&2
  printf '[build][error]   %s\n' "${NUGET_PACKAGES:-}" >&2
  printf '[build][error]   %s\n' "/opt/nuget-cache" >&2
  printf '[build][error]   %s\n' "${HOME}/.nuget/packages" >&2
  exit 1
fi

export NUGET_PACKAGES="${resolved_cache}"
log "using NuGet cache: ${NUGET_PACKAGES}"
log "restoring src/NfcReader/NfcReader.sln with --locked-mode --no-dependencies"
dotnet restore src/NfcReader/NfcReader.sln --locked-mode --no-dependencies

log "building src/NfcReader/NfcReader.sln with --no-restore"

tme_run_segment "build-offline.compile" "${SEGMENT_MS}" -- dotnet build src/NfcReader/NfcReader.sln \
  --no-restore \
  -p:RestorePackages=false \
  -p:EnableEL2Runtime=false

results_dir="build/TestResults/dotnet"
trx_file="${results_dir}/dotnet-results.trx"
mkdir -p "${results_dir}"

set +e
dotnet test src/NfcReader/NfcReader.sln \
  --no-build \
  --configuration Release \
  --logger "trx;LogFileName=dotnet-results.trx" \
  --logger "console;verbosity=normal" \
  --results-directory build/TestResults/dotnet \
  --blame-hang \
  --blame-hang-timeout 60s
test_rc=$?
set -e

if [[ ! -f "${trx_file}" ]]; then
  printf '[build][error] TRX result file not written; dotnet test may have crashed before executing\n' >&2
  printf '[build][error] check build/TestResults/dotnet/ for partial output\n' >&2
  exit 1
fi

python - "${trx_file}" <<'PY'
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

trx_path = sys.argv[1]
ns = {'t': 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}

def parse_time(raw):
    if not raw:
        return None
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z'):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    if raw.endswith('Z'):
        for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
            try:
                return datetime.strptime(raw[:-1], fmt)
            except ValueError:
                pass
    return None

root = ET.parse(trx_path).getroot()

counters = root.find('.//t:ResultSummary/t:Counters', ns)
if counters is None:
    total = passed = failed = skipped = 0
else:
    total = int(counters.attrib.get('total', '0'))
    passed = int(counters.attrib.get('passed', '0'))
    failed = int(counters.attrib.get('failed', '0'))
    skipped = int(counters.attrib.get('notExecuted', '0'))

times = root.find('.//t:Times', ns)
duration_s = 0
if times is not None:
    start = parse_time(times.attrib.get('start'))
    finish = parse_time(times.attrib.get('finish'))
    if start and finish:
        duration_s = int((finish - start).total_seconds())

unit_tests = {}
for ut in root.findall('.//t:TestDefinitions/t:UnitTest', ns):
    test_id = ut.attrib.get('id')
    method = ut.find('t:TestMethod', ns)
    class_name = method.attrib.get('className', '') if method is not None else ''
    method_name = method.attrib.get('name', '') if method is not None else ut.attrib.get('name', '')
    if test_id:
        unit_tests[test_id] = (class_name, method_name)

print(f'[build][test-summary] total={total} passed={passed} failed={failed} skipped={skipped}', file=sys.stderr)
print(f'[build][test-summary] duration={duration_s}s', file=sys.stderr)

for result in root.findall('.//t:Results/t:UnitTestResult', ns):
    if result.attrib.get('outcome') != 'Failed':
        continue
    test_id = result.attrib.get('testId', '')
    class_name, method_name = unit_tests.get(test_id, ('', result.attrib.get('testName', 'unknown')))
    if not class_name and '.' in method_name:
        class_name, method_name = method_name.rsplit('.', 1)
    message_node = result.find('t:Output/t:ErrorInfo/t:Message', ns)
    message = ''
    if message_node is not None and message_node.text:
        message = ' '.join(message_node.text.strip().split())
    full_name = f'{class_name}.{method_name}' if class_name else method_name
    print(f'[build][test-failure] {full_name}: {message}', file=sys.stderr)
PY

log "build completed"
exit "${test_rc}"
