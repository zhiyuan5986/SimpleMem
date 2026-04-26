#!/usr/bin/env python3
"""Transfer legacy LoCoMo logs into the richer metadata format and aligned span DB.

Inputs (legacy):
- *_extraction_trace.json (dialogue_context + extracted_entries as plain strings)
- *_longllmlingua_filtered.json (entry_index + llm_spans)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer legacy LoCoMo logs into metadata-rich trace + aligned span DB")
    parser.add_argument("--legacy-trace-json", type=Path, required=True)
    parser.add_argument("--legacy-filtered-json", type=Path, required=True)
    parser.add_argument("--output-trace-json", type=Path, required=True)
    parser.add_argument("--output-filtered-json", type=Path, required=True)
    parser.add_argument("--spans-db-dir", type=Path, required=True)
    return parser.parse_args()


def init_spans_db(spans_db_dir: Path) -> sqlite3.Connection:
    spans_db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(spans_db_dir / "llm_spans.sqlite")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_spans (
            entry_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def upsert_span_row(conn: sqlite3.Connection, entry_id: str, text: str, metadata: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO llm_spans (entry_id, text, metadata_json)
        VALUES (?, ?, ?)
        ON CONFLICT(entry_id) DO UPDATE SET
            text = excluded.text,
            metadata_json = excluded.metadata_json
        """,
        (entry_id, text, json.dumps(metadata, ensure_ascii=False)),
    )


def main() -> None:
    args = parse_args()

    legacy_trace = json.loads(args.legacy_trace_json.read_text(encoding="utf-8"))
    legacy_filtered = json.loads(args.legacy_filtered_json.read_text(encoding="utf-8"))

    transformed_trace: list[dict[str, Any]] = []
    entry_id_map: dict[tuple[int, int], str] = {}

    for trace_idx, item in enumerate(legacy_trace):
        context = item.get("dialogue_context", [])
        entries = item.get("extracted_entries", [])

        dialogue_text = [
            {
                "dialogue_id": turn_idx + 1,
                "speaker": "Speaker",
                "content": str(turn_text),
                "timestamp": None,
                "metadata": {
                    "legacy": True,
                    "trace_item_index": trace_idx,
                    "turn_index": turn_idx,
                },
            }
            for turn_idx, turn_text in enumerate(context if isinstance(context, list) else [])
        ]

        extracted_entries = []
        for entry_idx, entry_text in enumerate(entries if isinstance(entries, list) else []):
            entry_id = f"legacy_trace_{trace_idx}_entry_{entry_idx}"
            entry_id_map[(trace_idx, entry_idx)] = entry_id
            extracted_entries.append(
                {
                    "entry_id": entry_id,
                    "lossless_restatement": str(entry_text),
                    "keywords": [],
                    "timestamp": None,
                    "location": None,
                    "persons": [],
                    "entities": [],
                    "topic": None,
                    "metadata": {
                        "legacy": True,
                        "trace_item_index": trace_idx,
                        "entry_index": entry_idx,
                    },
                }
            )

        transformed_trace.append(
            {
                "dialogue_text": dialogue_text,
                "dialogue_context": [str(c) for c in (context if isinstance(context, list) else [])],
                "context": item.get("context", []),
                "context_entries": item.get("context_entries", []),
                "extracted_entries": extracted_entries,
                "extracted_entry_texts": [e["lossless_restatement"] for e in extracted_entries],
            }
        )

    transformed_filtered = dict(legacy_filtered)
    transformed_results = []
    conn = init_spans_db(args.spans_db_dir)

    for result in legacy_filtered.get("results", []):
        trace_item_index = int(result.get("trace_item_index", -1))
        transformed_entries = []

        for entry in result.get("entries", []):
            entry_index = int(entry.get("entry_index", -1))
            if entry_index < 0 or trace_item_index < 0:
                continue
            entry_id = entry_id_map.get((trace_item_index, entry_index), f"legacy_trace_{trace_item_index}_entry_{entry_index}")
            entry_text = str(entry.get("entry_text", ""))
            spans = entry.get("llm_spans", [])
            span_text_joined = " ".join(
                [str(span.get("span_text", "")).strip() for span in spans if isinstance(span, dict) and span.get("span_text")]
            ).strip()
            metadata = {
                "legacy": True,
                "trace_item_index": trace_item_index,
                "entry_index": entry_index,
                "entry_id": entry_id,
                "entry_text": entry_text,
                "llm_spans": spans,
                "llm_raw_spans": entry.get("llm_raw_spans", []),
                "llm_response": entry.get("llm_response", {}),
            }
            upsert_span_row(conn, entry_id, span_text_joined, metadata)

            transformed_entry = dict(entry)
            transformed_entry["entry_id"] = entry_id
            transformed_entry["entry_metadata"] = {
                "legacy": True,
                "trace_item_index": trace_item_index,
                "entry_index": entry_index,
            }
            transformed_entries.append(transformed_entry)

        transformed_result = dict(result)
        transformed_result["entries"] = transformed_entries
        transformed_results.append(transformed_result)

    transformed_filtered["results"] = transformed_results
    transformed_filtered.setdefault("config", {})
    transformed_filtered["config"]["spans_db_dir"] = str(args.spans_db_dir)

    args.output_trace_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_filtered_json.parent.mkdir(parents=True, exist_ok=True)

    args.output_trace_json.write_text(json.dumps(transformed_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_filtered_json.write_text(json.dumps(transformed_filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    conn.commit()
    conn.close()

    print(f"Saved transformed trace: {args.output_trace_json}")
    print(f"Saved transformed filtered result: {args.output_filtered_json}")
    print(f"Saved spans DB: {args.spans_db_dir / 'llm_spans.sqlite'}")


if __name__ == "__main__":
    main()
