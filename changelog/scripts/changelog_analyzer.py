#!/usr/bin/env python3
"""Analyze changelog releases and summarize entry counts by category."""

from __future__ import annotations

import re
import sys
from collections import defaultdict

CHANGELOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "CHANGELOG.md"
RE_VERSION = re.compile(r"^##\s*\[?([^\]]+)\]?\s*-\s*([0-9\-]+)?", re.IGNORECASE)
RE_CATEGORY = re.compile(r"^###?\s*(Added|Changed|Removed|Fixed|Security|Deprecated|Breaking)", re.IGNORECASE)
RE_ITEM = re.compile(r"^\s*[-*]\s+(.*)")


def parse_changelog(path: str) -> list[dict]:
    releases: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()

    current_release = None
    current_category = None

    for line in lines:
        version_match = RE_VERSION.match(line)
        category_match = RE_CATEGORY.match(line)
        item_match = RE_ITEM.match(line)

        if version_match:
            if current_release:
                releases.append(current_release)
            current_release = {
                "version": version_match.group(1).strip(),
                "date": version_match.group(2).strip() if version_match.group(2) else "Unreleased",
                "categories": defaultdict(list),
            }
            current_category = None
            continue

        if category_match and current_release:
            current_category = category_match.group(1).capitalize()
            continue

        if item_match and current_release and current_category:
            current_release["categories"][current_category].append(item_match.group(1).strip())

    if current_release:
        releases.append(current_release)

    return releases


def main() -> int:
    try:
        releases = parse_changelog(CHANGELOG_PATH)
    except FileNotFoundError:
        print(f"ERROR: changelog file not found: {CHANGELOG_PATH}")
        return 1

    print(f"Changelog Analysis: {CHANGELOG_PATH}")
    print(f"Total releases found: {len(releases)}")

    for release in releases:
        total = sum(len(items) for items in release["categories"].values())
        print(f"\nVersion {release['version']} ({release['date']})")
        print(f"Total entries: {total}")
        for category, items in release["categories"].items():
            print(f"  {category}: {len(items)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
