# Test Run Results

Date (UTC): 2026-02-16T02:16:30Z

## Commands Executed

### 1) `pytest -q`
- **Exit code:** `0`
- **Result:** Passed.
- **Output:**

```text
..........                                                               [100%]
10 passed in 0.08s
```

### 2) `dotnet test src/NfcReader/NfcReader.sln`
- **Exit code:** `127`
- **Result:** Could not execute in this environment (missing .NET CLI).
- **Output:**

```text
bash: command not found: dotnet
```

### 3) `python -m unittest discover`
- **Exit code:** `0`
- **Result:** Command succeeded, but no `unittest`-style tests were discovered.
- **Output:**

```text
----------------------------------------------------------------------
Ran 0 tests in 0.000s

OK
```

## Summary
- Added and ran Python unit tests for the recently introduced `iato_ops` components.
- `pytest` now executes successfully with `10` passing tests.
- .NET tests remain blocked by missing `dotnet` in the current runtime environment.
