#!/usr/bin/env python3
"""Coarse-to-fine extraction trace filtering with LAQuer-style LLM span mining.

This script keeps the coarse LongLLMLingua stage (turn-window PPL ranking) and
replaces token-level fine filtering with an LLM-based evidence span extraction
step inspired by LAQuer (ACL 2025).
"""

from __future__ import annotations

import argparse
import json
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
        if ":" in turn_text:
            left, right = turn_text.split(":", 1)
            if left.strip() and len(left.strip()) <= 32:
                speaker = left.strip()
                turn_text = right.strip()
        turns.append({"turn_index": idx, "speaker": speaker, "text": turn_text})
    return turns


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

    for item_idx in range(max_items):
        item = trace_data[item_idx]
        context = item.get("dialogue_context", [])
        entries = item.get("extracted_entries", [])
        if not isinstance(context, list) or not isinstance(entries, list):
            continue

        max_entries = len(entries) if args.max_entries_per_item < 0 else min(args.max_entries_per_item, len(entries))
        entry_results: list[dict[str, Any]] = []

        for entry_idx in range(max_entries):
            entry_text = str(entries[entry_idx])
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
            )

            merged_window = coarse.get("coarse_merged_window", {})
            support_turns = turn_window_to_context_turns(merged_window, [str(c) for c in context]) if merged_window else []

            align_result = align_entry_with_laquer(aligner=aligner, entry_text=entry_text, context_turns=support_turns) if support_turns else {}
            raw_rows = (
                align_result["results"].to_dict("records")
                if align_result and "results" in align_result and hasattr(align_result["results"], "to_dict")
                else []
            )
            spans = normalize_spans(rows=raw_rows, context_turns=support_turns)

            entry_results.append(
                {
                    "entry_index": entry_idx,
                    "entry_text": entry_text,
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
            "min_span_confidence": args.min_span_confidence,
        },
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved filtered+aligned results to: {args.output_json}")


if __name__ == "__main__":
    main()
