"""Restore a LanceDB memory_entries folder from an extraction trace JSON.

Example:
  python scripts/restore_lance_from_extraction_trace.py \
    --trace deepseek-chat/locomo10_sample_0_extraction_trace.json \
    --output-db deepseek-chat/memory_entries.lance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from database.vector_store import VectorStore
from models.memory_entry import MemoryEntry


def load_entries_from_trace(trace_path: Path) -> list[MemoryEntry]:
    with trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    if not isinstance(trace, list):
        raise ValueError("Extraction trace JSON must be a list.")

    dedup: dict[str, MemoryEntry] = {}
    generated = 0

    for item in trace:
        if not isinstance(item, dict):
            continue
        raw_entries = item.get("extracted_entries", [])
        if not isinstance(raw_entries, list):
            continue

        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            generated += 1
            try:
                entry = MemoryEntry.model_validate(raw)
            except Exception:
                # Skip malformed rows so one bad trace item won't block recovery.
                continue
            dedup[entry.entry_id] = entry

    if not dedup:
        raise ValueError("No valid extracted_entries found in trace.")

    print(f"Parsed {generated} extracted entries; deduplicated to {len(dedup)} by entry_id.")
    return list(dedup.values())


def normalize_entry(entry: MemoryEntry) -> MemoryEntry:
    def _clean_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(v) for v in value if str(v).strip()]

    return MemoryEntry(
        entry_id=entry.entry_id,
        lossless_restatement=(entry.lossless_restatement or "").strip(),
        keywords=_clean_list(entry.keywords),
        timestamp=(entry.timestamp or None),
        location=(entry.location or None),
        persons=_clean_list(entry.persons),
        entities=_clean_list(entry.entities),
        topic=(entry.topic or None),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover a memory_entries LanceDB folder from extraction trace JSON.")
    parser.add_argument("--trace", required=True, help="Path to *_extraction_trace.json")
    parser.add_argument("--output-db", required=True, help="Output LanceDB folder path, e.g. deepseek-chat/memory_entries.lance")
    parser.add_argument("--table-name", default="memory_entries", help="LanceDB table name (default: memory_entries)")
    parser.add_argument("--clear", action="store_true", help="If table exists, clear it before writing")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    output_db = Path(args.output_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)

    entries = [normalize_entry(e) for e in load_entries_from_trace(trace_path)]

    store = VectorStore(db_path=str(output_db), table_name=args.table_name)
    if args.clear:
        store.clear()

    store.add_entries(entries)
    store.optimize()

    restored_count = len(store.get_all_entries())
    print(f"Restored {restored_count} entries into: {output_db} (table={args.table_name})")


if __name__ == "__main__":
    main()
