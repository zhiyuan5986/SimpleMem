"""Restore a LanceDB memory_entries folder from a *_longllmlingua_filtered.json.

Example:
  python scripts/restore_lance_from_extraction_trace.py \
    --filtered-json deepseek-chat/locomo10_sample_0_longllmlingua_filtered.json \
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


def load_entries_from_filtered_json(filtered_json_path: Path) -> list[MemoryEntry]:
    with filtered_json_path.open("r", encoding="utf-8") as f:
        filtered = json.load(f)

    if not isinstance(filtered, dict):
        raise ValueError("Filtered JSON must be an object with a `results` field.")

    results = filtered.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Filtered JSON field `results` must be a list.")

    dedup: dict[str, MemoryEntry] = {}
    generated = 0

    for item in results:
        if not isinstance(item, dict):
            continue
        raw_entries = item.get("entries", [])
        if not isinstance(raw_entries, list):
            continue

        trace_item_index = item.get("trace_item_index")
        for entry_index, raw in enumerate(raw_entries):
            if not isinstance(raw, dict):
                continue
            generated += 1
            entry_id = raw.get("id")
            if not entry_id:
                # Keep entry_id strictly aligned with the filtered JSON schema.
                continue

            entry_text = str(raw.get("entry text", "")).strip()
            if not entry_text:
                # A blank memory is not useful for retrieval.
                continue

            entry_metadata = raw.get("entry_metadata", {})
            if not isinstance(entry_metadata, dict):
                entry_metadata = {}

            memory_entry_payload = {
                "entry_id": entry_id,
                "lossless_restatement": entry_text,
                "keywords": entry_metadata.get("keywords", []),
                "timestamp": entry_metadata.get("timestamp"),
                "location": entry_metadata.get("location"),
                "persons": entry_metadata.get("persons", []),
                "entities": entry_metadata.get("entities", []),
                "topic": entry_metadata.get("topic"),
            }
            try:
                entry = MemoryEntry.model_validate(memory_entry_payload)
            except Exception:
                # Skip malformed rows so one bad row won't block recovery.
                continue
            dedup[entry.entry_id] = entry

    if not dedup:
        raise ValueError("No valid entries found in filtered JSON.")

    print(f"Parsed {generated} filtered entries; deduplicated to {len(dedup)} by entry_id.")
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
    parser = argparse.ArgumentParser(description="Recover a memory_entries LanceDB folder from *_longllmlingua_filtered.json.")
    parser.add_argument("--filtered-json", required=True, help="Path to *_longllmlingua_filtered.json")
    parser.add_argument("--output-db", required=True, help="Output LanceDB folder path, e.g. deepseek-chat/memory_entries.lance")
    parser.add_argument("--table-name", default="memory_entries", help="LanceDB table name (default: memory_entries)")
    parser.add_argument("--clear", action="store_true", help="If table exists, clear it before writing")
    args = parser.parse_args()

    filtered_json_path = Path(args.filtered_json)
    if not filtered_json_path.exists():
        raise FileNotFoundError(f"Filtered JSON file not found: {filtered_json_path}")

    output_db = Path(args.output_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)

    entries = [normalize_entry(e) for e in load_entries_from_filtered_json(filtered_json_path)]

    store = VectorStore(db_path=str(output_db), table_name=args.table_name)
    if args.clear:
        store.clear()

    store.add_entries(entries)
    store.optimize()

    restored_count = len(store.get_all_entries())
    print(f"Restored {restored_count} entries into: {output_db} (table={args.table_name})")


if __name__ == "__main__":
    main()
