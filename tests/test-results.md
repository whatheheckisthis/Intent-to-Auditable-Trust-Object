# Test Run Results

Date (UTC): 2026-02-16T02:10:47Z

## Commands Executed

### 1) `dotnet test src/NfcReader/NfcReader.sln`
- **Exit code:** `127`
- **Result:** Failed due to missing .NET SDK/CLI in this environment.
- **Output:**

```text
bash: command not found: dotnet
```

### 2) `pytest`
- **Exit code:** `4`
- **Result:** Failed before running tests because `pyproject.toml` parsing errored.
- **Output:**

```text
ERROR: /workspace/Intent-to-Auditable-Trust-Object/pyproject.toml: Cannot overwrite a value (at line 39, column 34)
```

### 3) `python -m unittest discover`
- **Exit code:** `0`
- **Result:** Succeeded, but no tests were discovered.
- **Output:**

```text
----------------------------------------------------------------------
Ran 0 tests in 0.000s

OK
```

## Summary
- No application tests were executed successfully in this environment.
- Primary blockers:
  - `.NET CLI` is not installed (`dotnet` missing).
  - `pytest` is blocked by a `pyproject.toml` parse error.
- The standard library unittest discovery ran but found `0` tests.
