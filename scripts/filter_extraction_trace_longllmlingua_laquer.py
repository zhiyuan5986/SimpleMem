#!/usr/bin/env python3
"""Coarse-to-fine extraction trace filtering with LAQuer-style LLM span mining.

This script keeps the coarse LongLLMLingua stage (turn-window PPL ranking) and
replaces token-level fine filtering with an LLM-based evidence span extraction
step inspired by LAQuer (ACL 2025).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.filter_extraction_trace_longllmlingua import TopKPPLPromptCompressor, load_trace
from src.consts import LFQA_TASK
from src.laquer_methods.llm_method import LLMBasedAlignment  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coarse LongLLMLingua + LAQuer-style LLM span extraction for *_extraction_trace.json"
    )
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)

    parser.add_argument("--compressor-model-name", type=str, required=True)
    parser.add_argument("--compressor-device-map", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--turn-window-k", type=int, default=2)
    parser.add_argument("--turn-separator", type=str, default="\n")
    parser.add_argument(
        "--first-stage-filter",
        choices=["coarse_topk_by_ppl", "fine_topk_by_contrastive_ppl"],
        default="coarse_topk_by_ppl",
        help="Document/window ranking strategy used in stage-1 selection.",
    )
    parser.add_argument("--condition-in-question", choices=["none", "before", "after"], default="after")
    parser.add_argument("--condition-text", type=str, default="")
    parser.add_argument("--condition-placement", choices=["none", "prepend", "append"], default="prepend")

    parser.add_argument("--model", type=str, required=True, help="Model identifier passed to LAQuer inference wrapper.")

    parser.add_argument("--max-trace-items", type=int, default=-1)
    parser.add_argument("--max-entries-per-item", type=int, default=-1)
    parser.add_argument("--min-span-confidence", type=float, default=0.0)
    parser.add_argument(
        "--spans-db-dir",
        type=Path,
        default=None,
        help="Directory for storing entry-aligned LLM span sqlite DB (default: sibling folder of output-json).",
    )
    return parser.parse_args()


def normalize_spans(rows: list[dict[str, Any]], context_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_map = {f"turn_{turn['turn_index']}": turn for turn in context_turns}
    normalized: list[dict[str, Any]] = []

    def _parse_first_offset(raw_offsets: Any) -> tuple[int, int]:
        if not isinstance(raw_offsets, list) or len(raw_offsets) == 0:
            return -1, -1
        first = raw_offsets[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            try:
                return int(first[0]), int(first[1])
            except (TypeError, ValueError):
                return -1, -1
        return -1, -1

    for row in rows:
        # print(row)
        source_id = str(row.get("documentFile", ""))
        turn = source_map.get(source_id)
        if not turn:
            continue

        offset_start, offset_end = _parse_first_offset(row.get("docSpanOffsets", []))

        normalized.append(
            {
                "turn_index": turn["turn_index"],
                "speaker": turn["speaker"],
                "span_text": str(row.get("docSpanText", "")).strip(),
                "char_start": offset_start,
                "char_end": offset_end,
                "is_verbatim_match": offset_start >= 0,
            }
        )

    normalized.sort(key=lambda x: (x["turn_index"], x["char_start"]))
    return normalized


def align_entry_with_laquer(aligner: LLMBasedAlignment, entry_text: str, context_turns: list[dict[str, Any]]) -> dict[str, Any]:
    source_spans = {f"turn_{turn['turn_index']}": turn["text"] for turn in context_turns}
    source_metadata = {
        f"turn_{turn['turn_index']}": [
            {
                "documentFile": f"turn_{turn['turn_index']}",
                "docSpanText": turn["text"],
                "docSpanOffsets": [[0, len(turn["text"])]],
                "docSentCharIdx": 0,
                "docSentText": turn["text"],
            }
        ]
        for turn in context_turns
    }
    datapoint = {
        "topic": "locomo_trace",
        "unique_id": "locomo_trace",
        "source_spans": source_spans,
        "source_metadata": source_metadata,
        "sentence": entry_text,
        "scuSpanOffsets": [0, len(entry_text)],
        "complete_scuSentence": entry_text,
        "is_sampled": False,
        "source_granularity": "sentence",
        "fact_idx": 0,
        "question": entry_text,
    }
    return aligner.extract_attribution(datapoint)


def turn_window_to_context_turns(coarse_window: dict[str, Any], full_context: list[str]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for idx in coarse_window.get("turn_indices", []):
        turn_text = str(full_context[idx])
        speaker = "Speaker"
        # if ":" in turn_text:
        #     left, right = turn_text.split(":", 1)
        #     if left.strip() and len(left.strip()) <= 32:
        #         speaker = left.strip()
        #         turn_text = right.strip()
        turns.append({"turn_index": idx, "speaker": speaker, "text": turn_text})
    return turns


def normalize_context_turns(item: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    dialogue_text = item.get("dialogue_text", [])
    if isinstance(dialogue_text, list) and dialogue_text and isinstance(dialogue_text[0], dict):
        full_context = [str(turn.get("content", "")) for turn in dialogue_text]
        turn_metadata = []
        for idx, turn in enumerate(dialogue_text):
            turn_metadata.append(
                {
                    "turn_index": idx,
                    "speaker": turn.get("speaker", "Speaker"),
                    "text": str(turn.get("content", "")),
                    "dialogue_id": turn.get("dialogue_id"),
                    "timestamp": turn.get("timestamp"),
                    "metadata": turn.get("metadata", {}),
                }
            )
        return full_context, turn_metadata

    context = item.get("dialogue_context", [])
    if isinstance(context, list):
        return [str(c) for c in context], [{"turn_index": i, "speaker": "Speaker", "text": str(c)} for i, c in enumerate(context)]
    return [], []


def parse_entry(entry_obj: Any, trace_item_index: int, entry_index: int) -> tuple[str, str, dict[str, Any]]:
    if isinstance(entry_obj, dict):
        entry_id = str(entry_obj.get("entry_id") or f"trace_{trace_item_index}_entry_{entry_index}")
        entry_text = str(entry_obj.get("lossless_restatement", ""))
        return entry_id, entry_text, entry_obj

    entry_text = str(entry_obj)
    entry_id = f"trace_{trace_item_index}_entry_{entry_index}"
    return entry_id, entry_text, {"entry_id": entry_id, "lossless_restatement": entry_text}


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

    trace_data = load_trace(args.trace_json)
    compressor = TopKPPLPromptCompressor(
        model_name=args.compressor_model_name,
        device_map=args.compressor_device_map,
    )
    aligner = LLMBasedAlignment(task=LFQA_TASK, args=args)

    max_items = len(trace_data) if args.max_trace_items < 0 else min(args.max_trace_items, len(trace_data))
    results: list[dict[str, Any]] = []

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    spans_db_dir = args.spans_db_dir or (args.output_json.parent / f"{args.output_json.stem}_spans_db")
    spans_conn = init_spans_db(spans_db_dir)

    for item_idx in range(max_items):
        item = trace_data[item_idx]
        context, context_turn_meta = normalize_context_turns(item)
        entries = item.get("extracted_entries", [])
        if not isinstance(context, list) or not isinstance(entries, list):
            continue

        max_entries = len(entries) if args.max_entries_per_item < 0 else min(args.max_entries_per_item, len(entries))
        entry_results: list[dict[str, Any]] = []

        for entry_idx in range(max_entries):
            entry_id, entry_text, entry_metadata = parse_entry(entries[entry_idx], item_idx, entry_idx)
            coarse = compressor.compress_with_coarse_topk(
                context=[str(c) for c in context],
                entry_text=entry_text,
                top_k=args.top_k,
                condition_in_question=args.condition_in_question,
                condition_text=args.condition_text,
                condition_placement=args.condition_placement,
                turn_window_k=args.turn_window_k,
                turn_separator=args.turn_separator,
                entry_budget_multiplier=1.0,
                first_stage_filter=args.first_stage_filter,
                ppl_plot_dir=Path(os.path.splitext(args.output_json)[0]) / "ppl_plot",
                ppl_plot_file_stem=f"trace_{item_idx:04d}_entry_{entry_idx:04d}",
            )

            merged_window = coarse.get("coarse_merged_window", {})
            support_turns = turn_window_to_context_turns(merged_window, [str(c) for c in context]) if merged_window else []
            for turn in support_turns:
                idx = turn["turn_index"]
                if idx < len(context_turn_meta):
                    turn.update(context_turn_meta[idx])

            align_result = align_entry_with_laquer(aligner=aligner, entry_text=entry_text, context_turns=support_turns) if support_turns else {}
            raw_rows = (
                align_result["results"].to_dict("records")
                if align_result and "results" in align_result and hasattr(align_result["results"], "to_dict")
                else []
            )
            spans = normalize_spans(rows=raw_rows, context_turns=support_turns)
            span_text_joined = " ".join([span.get("span_text", "") for span in spans if span.get("span_text")]).strip()
            span_db_metadata = {
                "trace_item_index": item_idx,
                "entry_index": entry_idx,
                "entry_id": entry_id,
                "entry_text": entry_text,
                "entry_metadata": entry_metadata,
                "llm_spans": spans,
                "llm_raw_spans": raw_rows,
                "llm_response": {k: v for k, v in align_result.items() if k != "results"} if align_result else {},
            }
            upsert_span_row(spans_conn, entry_id=entry_id, text=span_text_joined, metadata=span_db_metadata)

            entry_results.append(
                {
                    "entry_index": entry_idx,
                    "entry_id": entry_id,
                    "entry_text": entry_text,
                    "entry_metadata": entry_metadata,
                    "coarse_topk_indices": coarse["chosen_doc_indices"],
                    "coarse_topk_windows": coarse["chosen_windows"],
                    "coarse_topk_ppl": coarse["chosen_doc_ppl"],
                    "coarse_top1_window": coarse.get("coarse_top1_window", {}),
                    "coarse_merged_window": merged_window,
                    "llm_raw_spans": raw_rows,
                    "llm_response": {k: v for k, v in align_result.items() if k != "results"} if align_result else {},
                    "llm_spans": spans,
                    "num_llm_spans": len(spans),
                }
            )
            print(json.dumps(entry_results[-1], ensure_ascii=False, indent=2))

        results.append(
            {
                "trace_item_index": item_idx,
                "dialogue_context_size": len(context),
                "num_entries_processed": len(entry_results),
                "entries": entry_results,
            }
        )

    output = {
        "config": {
            "trace_json": str(args.trace_json),
            "compressor_model_name": args.compressor_model_name,
            "compressor_device_map": args.compressor_device_map,
            "top_k": args.top_k,
            "turn_window_k": args.turn_window_k,
            "turn_separator": args.turn_separator,
            "first_stage_filter": args.first_stage_filter,
            "condition_in_question": args.condition_in_question,
            "condition_text": args.condition_text,
            "condition_placement": args.condition_placement,
            "model": args.model,
            "max_trace_items": args.max_trace_items,
            "max_entries_per_item": args.max_entries_per_item,
            "spans_db_dir": str(spans_db_dir),
            "min_span_confidence": args.min_span_confidence,
        },
        "results": results,
    }

    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    spans_conn.commit()
    spans_conn.close()
    print(f"Saved filtered+aligned results to: {args.output_json}")
    print(f"Saved entry-aligned LLM spans DB to: {spans_db_dir / 'llm_spans.sqlite'}")


if __name__ == "__main__":
    main()
