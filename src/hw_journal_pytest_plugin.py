from __future__ import annotations

import json
from pathlib import Path

from src.hw_journal import get_hw_journal


def _failures_dir() -> Path:
    path = Path("build/hw-journal/failures")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_nodeid(nodeid: str) -> str:
    return nodeid.replace("/", "_").replace(":", "_").replace("[", "_").replace("]", "_")


def pytest_runtest_setup(item):
    get_hw_journal().attach_test(item.nodeid)


def pytest_runtest_teardown(item, nextitem):
    journal = get_hw_journal()
    item._hw_journal_snapshot = journal.snapshot()
    journal.detach_test()


def pytest_runtest_makereport(item, call):
    if call.when != "call" or call.excinfo is None:
        return
    snapshot = getattr(item, "_hw_journal_snapshot", [])
    out = _failures_dir() / f"{_sanitize_nodeid(item.nodeid)}.json"
    out.write_text(json.dumps({"nodeid": item.nodeid, "events": snapshot}, indent=2), encoding="utf-8")


def pytest_sessionfinish(session, exitstatus):
    journal = get_hw_journal()
    journal.close()
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = getattr(reporter, "stats", {}) if reporter is not None else {}
    failures = sorted({report.nodeid for reports in stats.values() for report in reports if getattr(report, "failed", False)})
    summary = {
        "total_events": journal.total_events,
        "dropped_events": journal.dropped,
        "tests_with_failures": failures,
        "concern_counts": journal.concern_counts,
    }
    out = Path("build/hw-journal/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
