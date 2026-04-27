#!/usr/bin/env python3
"""Batch consolidate LoCoMo extraction traces with LongLLMLingua+LAQuer span mining.

For each ``locomo10_sample_{idx}_extraction_trace.json`` found under a directory, this
script will:
1) run the same coarse-to-fine filtering flow as
   ``scripts/filter_extraction_trace_longllmlingua_laquer.py``;
2) auto-locate the corresponding sample DB directory (``locomo10_sample_{idx}``);
3) upsert extracted spans into a ``llm_spans`` table in that DB (or fallback DB dir).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from database.vector_store import RawContextVectorStore
from models.raw_context import RawContextEntry
from scripts.filter_extraction_trace_longllmlingua import TopKPPLPromptCompressor, load_trace
from scripts.filter_extraction_trace_longllmlingua_laquer import (
    align_entry_with_laquer,
    normalize_context_turns,
    normalize_spans,
    parse_entry,
    turn_window_to_context_turns,
)
from src.consts import LFQA_TASK
from src.laquer_methods.llm_method import LLMBasedAlignment  # type: ignore


SAMPLE_TRACE_RE = re.compile(r"locomo10_sample_(\d+)_extraction_trace\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto consolidate LoCoMo traces and span DBs for all samples.")
    parser.add_argument("--logs-dir", type=Path, required=True, help="Directory containing locomo10 sample trace logs.")
    parser.add_argument(
        "--db-search-roots",
        type=Path,
        nargs="+",
        default=[Path(".")],
        help="One or more roots used to auto-find per-sample DB dirs named locomo10_sample_{idx}.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="longllmlingua_filtered",
        help="Suffix for per-sample output JSON: locomo10_sample_{idx}_{suffix}.json",
    )

    parser.add_argument("--compressor-model-name", type=str, required=True)
    parser.add_argument("--compressor-device-map", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--turn-window-k", type=int, default=2)
    parser.add_argument("--turn-separator", type=str, default="\n")
    parser.add_argument(
        "--first-stage-filter",
        choices=["coarse_topk_by_ppl", "fine_topk_by_contrastive_ppl"],
        default="coarse_topk_by_ppl",
    )
    parser.add_argument("--condition-in-question", choices=["none", "before", "after"], default="after")
    parser.add_argument("--condition-text", type=str, default="")
    parser.add_argument("--condition-placement", choices=["none", "prepend", "append"], default="prepend")
    parser.add_argument("--model", type=str, required=True, help="Model identifier passed to LAQuer inference wrapper.")
    parser.add_argument("--max-trace-items", type=int, default=-1)
    parser.add_argument("--max-entries-per-item", type=int, default=-1)
    parser.add_argument(
        "--fallback-spans-root",
        type=Path,
        default=Path("outputs/locomo10_spans_fallback"),
        help="Fallback root if sample DB dir is not found; uses <root>/locomo10_sample_{idx}.",
    )
    return parser.parse_args()


def find_trace_files(logs_dir: Path) -> list[tuple[int, Path]]:
    traces: list[tuple[int, Path]] = []
    for path in logs_dir.glob("locomo10_sample_*_extraction_trace.json"):
        m = SAMPLE_TRACE_RE.search(path.name)
        if not m:
            continue
        traces.append((int(m.group(1)), path))
    return sorted(traces, key=lambda x: x[0])


def find_sample_db_dir(sample_idx: int, search_roots: list[Path]) -> Path | None:
    target_name = f"locomo10_sample_{sample_idx}"
    for root in search_roots:
        root = root.resolve()
        direct = root / target_name
        if direct.exists() and direct.is_dir():
            return direct
        for candidate in root.glob(f"**/{target_name}"):
            if candidate.is_dir():
                return candidate
    return None


def init_spans_db(spans_db_dir: Path) -> RawContextVectorStore:
    spans_db_dir.mkdir(parents=True, exist_ok=True)
    store = RawContextVectorStore(db_path=str(spans_db_dir), table_name="llm_spans")
    # Ensure FTS is available for keyword/BM25 retrieval even when opening an existing table.
    # This avoids "Cannot perform full text search unless an INVERTED index has been created".
    store._init_fts_index()
    return store


def upsert_span_row(store: RawContextVectorStore, entry_id: str, text: str, metadata: dict[str, Any]) -> None:
    store.upsert_entry(RawContextEntry(entry_id=entry_id, text=text, metadata=metadata))


def attach_turn_dia_ids_to_spans(
    spans: list[dict[str, Any]],
    support_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Backfill dia/dialogue ids from support turns onto each normalized span."""
    turn_id_map: dict[int, dict[str, Any]] = {}
    for turn in support_turns:
        try:
            turn_idx = int(turn.get("turn_index", -1))
        except (TypeError, ValueError):
            continue
        turn_id_map[turn_idx] = {
            "dia_id": turn.get("dialogue_id"),
            "dialogue_id": turn.get("dialogue_id"),
        }

    enriched: list[dict[str, Any]] = []
    for span in spans:
        out = dict(span)
        try:
            turn_idx = int(out.get("turn_index", -1))
        except (TypeError, ValueError):
            turn_idx = -1
        if turn_idx in turn_id_map:
            out.update(turn_id_map[turn_idx])
        enriched.append(out)
    return enriched


def process_single_sample(
    *,
    args: argparse.Namespace,
    sample_idx: int,
    trace_json: Path,
    compressor: TopKPPLPromptCompressor,
    aligner: LLMBasedAlignment,
) -> dict[str, Any]:
    trace_data = load_trace(trace_json)
    output_json = trace_json.with_name(f"locomo10_sample_{sample_idx}_{args.output_suffix}.json")

    db_dir = find_sample_db_dir(sample_idx, args.db_search_roots)
    used_fallback_db = False
    if db_dir is None:
        db_dir = args.fallback_spans_root / f"locomo10_sample_{sample_idx}"
        used_fallback_db = True
    spans_store = init_spans_db(db_dir)

    max_items = len(trace_data) if args.max_trace_items < 0 else min(args.max_trace_items, len(trace_data))
    results: list[dict[str, Any]] = []

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
                ppl_plot_dir=Path(os.path.splitext(output_json)[0]) / "ppl_plot",
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
            spans = attach_turn_dia_ids_to_spans(spans=spans, support_turns=support_turns)
            support_turn_dia_ids = sorted(
                {
                    str(turn.get("dialogue_id")).strip()
                    for turn in support_turns
                    if turn.get("dialogue_id") is not None and str(turn.get("dialogue_id")).strip()
                }
            )

            span_text_joined = " ".join([span.get("span_text", "") for span in spans if span.get("span_text")]).strip()
            span_db_metadata = {
                "sample_idx": sample_idx,
                "trace_item_index": item_idx,
                "entry_index": entry_idx,
                "entry_id": entry_id,
                "entry_text": entry_text,
                "entry_metadata": entry_metadata,
                "support_turns": support_turns,
                "support_turn_dia_ids": support_turn_dia_ids,
                "llm_spans": spans,
                "llm_raw_spans": raw_rows,
                "llm_response": {k: v for k, v in align_result.items() if k != "results"} if align_result else {},
            }
            upsert_span_row(spans_store, entry_id=entry_id, text=span_text_joined, metadata=span_db_metadata)

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

        results.append(
            {
                "trace_item_index": item_idx,
                "dialogue_context_size": len(context),
                "num_entries_processed": len(entry_results),
                "entries": entry_results,
            }
        )

    output_json.write_text(
        json.dumps(
            {
                "config": {
                    "trace_json": str(trace_json),
                    "output_json": str(output_json),
                    "sample_idx": sample_idx,
                    "sample_db_dir": str(db_dir),
                    "used_fallback_db": used_fallback_db,
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
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    spans_store.optimize()
    return {
        "sample_idx": sample_idx,
        "trace_json": str(trace_json),
        "output_json": str(output_json),
        "sample_db_dir": str(db_dir),
        "used_fallback_db": used_fallback_db,
        "trace_items_processed": len(results),
    }


def main() -> None:
    args = parse_args()
    args.logs_dir = args.logs_dir.resolve()
    args.db_search_roots = [p.resolve() for p in args.db_search_roots]
    args.fallback_spans_root = args.fallback_spans_root.resolve()

    traces = find_trace_files(args.logs_dir)
    if not traces:
        raise FileNotFoundError(f"No locomo traces found under: {args.logs_dir}")

    compressor = TopKPPLPromptCompressor(model_name=args.compressor_model_name, device_map=args.compressor_device_map)
    aligner = LLMBasedAlignment(task=LFQA_TASK, args=args)

    summary: list[dict[str, Any]] = []
    for sample_idx, trace_json in traces:
        print(f"[LoCoMo] processing sample={sample_idx} trace={trace_json}")
        summary.append(
            process_single_sample(
                args=args,
                sample_idx=sample_idx,
                trace_json=trace_json,
                compressor=compressor,
                aligner=aligner,
            )
        )

    print("\n[Done] consolidated samples:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
