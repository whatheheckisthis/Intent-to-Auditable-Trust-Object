#!/usr/bin/env python3
"""Decode a Base64 payload from an XML tag into the release Python script target."""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PY_OUT = REPO_ROOT / "lean/iato_v7/release/iato-v7-nmap-audit-0.7.0.py"


class ExtractionError(RuntimeError):
    """Raised when the XML/Base64 decode workflow fails."""


def _decode_xml_entities(value: str) -> str:
    return (
        value.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )


def read_tag_text(xml_path: Path, tag: str) -> str:
    try:
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        raise ExtractionError(f"invalid XML in {xml_path}: {exc}") from exc

    element = root.find(f".//{tag}")
    if element is None:
        raise ExtractionError(f"tag <{tag}> not found in {xml_path}")

    text = (element.text or "").strip()
    if not text:
        raise ExtractionError(f"tag <{tag}> is empty in {xml_path}")

    compact = re.sub(r"\s+", "", _decode_xml_entities(text))
    if not compact:
        raise ExtractionError(f"tag <{tag}> contains only whitespace in {xml_path}")
    return compact


def decode_base64_to_file(b64_data: str, output_path: Path) -> None:
    try:
        payload = base64.b64decode(b64_data, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError("Base64 payload is invalid") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    output_path.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode Base64 from an XML tag to the release Python script"
    )
    parser.add_argument("--xml", type=Path, required=True, help="Input XML file")
    parser.add_argument("--py-tag", type=str, required=True, help="XML tag containing Base64 Python script payload")
    parser.add_argument("--py-out", type=Path, default=DEFAULT_PY_OUT, help="Target python script path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    py_payload = read_tag_text(args.xml, args.py_tag)
    decode_base64_to_file(py_payload, args.py_out)

    print(f"Decoded python script written to: {args.py_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
