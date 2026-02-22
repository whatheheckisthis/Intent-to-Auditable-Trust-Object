#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    events = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"invalid json at line {i}: {exc}", file=sys.stderr)
                raise
    return events


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("journal_file")
    p.add_argument("--concern")
    p.add_argument("--event")
    p.add_argument("--stream-id", type=int)
    p.add_argument("--test")
    p.add_argument("--failures-only", action="store_true")
    p.add_argument("--since-event")
    p.add_argument("--format", choices=["table", "json", "timeline"], default="table")
    return p.parse_args()


def apply_filters(events: list[dict], args: argparse.Namespace) -> list[dict]:
    out = events
    if args.concern:
        out = [e for e in out if e.get("concern") == args.concern]
    if args.event:
        out = [e for e in out if e.get("event") == args.event]
    if args.stream_id is not None:
        out = [e for e in out if e.get("stream_id") == args.stream_id]
    if args.test:
        out = [e for e in out if e.get("test") == args.test]
    if args.failures_only:
        out = [e for e in out if e.get("test") and "fail" in str(e.get("test")).lower()]
    if args.since_event:
        idx = next((i for i, e in enumerate(out) if e.get("event") == args.since_event), None)
        out = out[idx:] if idx is not None else []
    return out


def print_table(events: list[dict]) -> None:
    for e in events:
        print(f"{e.get('ts', 0):.3f}\t{e.get('concern')}\t{e.get('event')}\tstream={e.get('stream_id')}\ttest={e.get('test')}\t{e.get('data')}")


def print_timeline(events: list[dict]) -> None:
    if not events:
        return
    t0 = events[0].get("monotonic", 0.0)
    for e in events:
        dt = float(e.get("monotonic", t0)) - float(t0)
        print(f"T+{dt:0.3f}  [{str(e.get('concern')):9}] {str(e.get('event')):17} {e.get('data')}")


def main() -> int:
    args = parse_args()
    try:
        events = load_events(Path(args.journal_file))
    except json.JSONDecodeError:
        return 1
    events = apply_filters(events, args)
    if args.format == "json":
        print(json.dumps(events, indent=2))
    elif args.format == "timeline":
        print_timeline(events)
    else:
        print_table(events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
