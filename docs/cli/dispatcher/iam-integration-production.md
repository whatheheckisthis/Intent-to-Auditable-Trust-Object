# IAM Integration Production Notes

This deployment includes additional Python integration modules:

- `iato_ops/api_keys.py` for environment-based API key loading.
- `iato_ops/endpoints.py` for authenticated endpoint calls.
- `iato_ops/syscalls.py` for runtime syscall metadata snapshots.

## API Key Requirements

Set the following environment variables in deployment runtime:

- `IATO_POLICY_SERVICE_API_KEY`
- `IATO_TELEMETRY_API_KEY`

No API keys are committed to source control.
