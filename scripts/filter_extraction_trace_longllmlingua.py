#!/usr/bin/env python3
"""Coarse-to-fine LongLLMLingua filtering over SimpleMem extraction traces.

Given a ``*_extraction_trace.json`` produced by ``test_locomo10.py``, this script:
1. Treats each ``dialogue_context`` item as one document.
2. For every extracted entry, uses PPL-based coarse ranking to keep top-k documents.
3. Runs LongLLMLingua token-level compression on the selected top-k documents.

Compared with stock LongLLMLingua usage, this script provides:
- explicit coarse top-k by PPL before token-level filtering;
- flexible question conditioning with user-provided condition text placement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llmlingua import PromptCompressor


class TopKPPLPromptCompressor(PromptCompressor):
    """PromptCompressor extension with explicit top-k PPL coarse filtering."""

    @staticmethod
    def build_effective_question(
        question: str,
        condition_text: str,
        condition_placement: str,
    ) -> str:
        question = (question or "").strip()
        condition_text = (condition_text or "").strip()
        if not condition_text or condition_placement == "none":
            return question
        if condition_placement == "prepend":
            return f"{condition_text}\n{question}" if question else condition_text
        if condition_placement == "append":
            return f"{question}\n{condition_text}" if question else condition_text
        raise ValueError(f"Unknown condition placement: {condition_placement}")

    def get_condition_ppl_flexible(
        self,
        text: str,
        question: str,
        condition_in_question: str,
        condition_text: str,
        condition_placement: str,
    ):
        effective_question = self.build_effective_question(
            question=question,
            condition_text=condition_text,
            condition_placement=condition_placement,
        )

        if condition_in_question == "none":
            return self.get_ppl(text, granularity="sentence")
        if condition_in_question == "before":
            return self.get_ppl(
                effective_question + text,
                granularity="sentence",
                condition_mode="after",
                condition_pos_id=self.get_token_length(effective_question) - 1,
            )
        if condition_in_question == "after":
            return self.get_ppl(
                text + effective_question,
                granularity="sentence",
                condition_mode="after",
                condition_pos_id=self.get_token_length(text) - 1,
            )
        raise ValueError(
            f"condition_in_question must be one of [none, before, after], got {condition_in_question}"
        )

    def coarse_topk_by_ppl(
        self,
        context: list[str],
        question: str,
        top_k: int,
        condition_in_question: str,
        condition_text: str,
        condition_placement: str,
    ) -> tuple[list[int], list[float]]:
        if not context:
            return [], []

        ppl_values: list[float] = []
        for doc in context:
            ppl = self.get_condition_ppl_flexible(
                text=doc,
                question=question,
                condition_in_question=condition_in_question,
                condition_text=condition_text,
                condition_placement=condition_placement,
            )
            ppl_values.append(float(ppl.detach().cpu().numpy().item()))

        sort_direct = -1 if condition_in_question == "none" else 1
        ranked_indices = sorted(
            range(len(context)),
            key=lambda idx: sort_direct * ppl_values[idx],
        )
        keep_n = min(max(top_k, 1), len(context))
        return ranked_indices[:keep_n], ppl_values

    def compress_with_coarse_topk(
        self,
        context: list[str],
        entry_text: str,
        top_k: int,
        condition_in_question: str,
        condition_text: str,
        condition_placement: str,
        instruction: str,
        rate: float,
        target_token: float,
        iterative_size: int,
        use_sentence_level_filter: bool,
        use_token_level_filter: bool,
        keep_split: bool,
        token_budget_ratio: float,
        force_context_ids: list[int] | None,
    ) -> dict[str, Any]:
        chosen_indices, ppl_values = self.coarse_topk_by_ppl(
            context=context,
            question=entry_text,
            top_k=top_k,
            condition_in_question=condition_in_question,
            condition_text=condition_text,
            condition_placement=condition_placement,
        )
        chosen_context = [context[i] for i in chosen_indices]

        effective_question = self.build_effective_question(
            question=entry_text,
            condition_text=condition_text,
            condition_placement=condition_placement,
        )

        compression = super().compress_prompt(
            context=chosen_context,
            instruction=instruction,
            question=effective_question,
            rate=rate,
            target_token=target_token,
            iterative_size=iterative_size,
            force_context_ids=force_context_ids,
            use_sentence_level_filter=use_sentence_level_filter,
            use_context_level_filter=False,
            use_token_level_filter=use_token_level_filter,
            keep_split=keep_split,
            token_budget_ratio=token_budget_ratio,
            condition_in_question="none",
            rank_method="llmlingua",
        )

        compression["chosen_doc_indices"] = chosen_indices
        compression["chosen_doc_ppl"] = [ppl_values[i] for i in chosen_indices]
        compression["effective_question"] = effective_question
        compression["original_question"] = entry_text
        return compression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coarse top-k + fine token-level LongLLMLingua filtering for extraction traces."
    )
    parser.add_argument("--trace-json", type=Path, required=True, help="Path to *_extraction_trace.json")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to save filtered output JSON")

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model path/name for llmlingua PromptCompressor, e.g. /path/to/llama-2-7b-hf",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="cuda",
        help="Device map for PromptCompressor (e.g., cuda, cpu, auto)",
    )

    parser.add_argument("--top-k", type=int, default=5, help="Number of dialogue_context docs kept by coarse PPL ranking")
    parser.add_argument("--instruction", type=str, default="", help="Optional instruction passed to compressor")
    parser.add_argument("--rate", type=float, default=0.5, help="Compression rate target")
    parser.add_argument("--target-token", type=float, default=-1, help="Optional target token budget, -1 to disable")
    parser.add_argument("--iterative-size", type=int, default=200, help="Iterative compression chunk size")
    parser.add_argument("--token-budget-ratio", type=float, default=1.4, help="Sentence-level budget ratio")

    parser.add_argument(
        "--condition-in-question",
        choices=["none", "before", "after"],
        default="after",
        help="Conditioning mode used in PPL computation",
    )
    parser.add_argument(
        "--condition-text",
        type=str,
        default="",
        help="User-defined condition text that can be injected around question",
    )
    parser.add_argument(
        "--condition-placement",
        choices=["none", "prepend", "append"],
        default="prepend",
        help="Where to place condition_text relative to question",
    )

    parser.add_argument("--use-sentence-level-filter", action="store_true", help="Enable sentence-level filtering in fine stage")
    parser.add_argument("--disable-token-level-filter", action="store_true", help="Disable token-level filtering in fine stage")
    parser.add_argument("--keep-split", action="store_true", help="Keep original separators")

    parser.add_argument("--max-trace-items", type=int, default=-1, help="Only process first N trace items, -1 for all")
    parser.add_argument("--max-entries-per-item", type=int, default=-1, help="Only process first N extracted entries per trace item")
    return parser.parse_args()


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Trace JSON must be a list")
    return data


def main() -> None:
    args = parse_args()
    trace_data = load_trace(args.trace_json)

    compressor = TopKPPLPromptCompressor(
        model_name=args.model_name,
        device_map=args.device_map,
    )

    max_items = len(trace_data) if args.max_trace_items < 0 else min(args.max_trace_items, len(trace_data))
    results: list[dict[str, Any]] = []

    for item_idx in range(max_items):
        item = trace_data[item_idx]
        context = item.get("dialogue_context", [])
        extracted_entries = item.get("extracted_entries", [])

        if not isinstance(context, list) or not isinstance(extracted_entries, list):
            continue

        max_entries = len(extracted_entries) if args.max_entries_per_item < 0 else min(args.max_entries_per_item, len(extracted_entries))

        entry_results: list[dict[str, Any]] = []
        for entry_idx in range(max_entries):
            entry_text = str(extracted_entries[entry_idx])
            res = compressor.compress_with_coarse_topk(
                context=[str(c) for c in context],
                entry_text=entry_text,
                top_k=args.top_k,
                condition_in_question=args.condition_in_question,
                condition_text=args.condition_text,
                condition_placement=args.condition_placement,
                instruction=args.instruction,
                rate=args.rate,
                target_token=args.target_token,
                iterative_size=args.iterative_size,
                use_sentence_level_filter=args.use_sentence_level_filter,
                use_token_level_filter=not args.disable_token_level_filter,
                keep_split=args.keep_split,
                token_budget_ratio=args.token_budget_ratio,
                force_context_ids=None,
            )
            entry_results.append(
                {
                    "entry_index": entry_idx,
                    "entry_text": entry_text,
                    "coarse_topk_indices": res["chosen_doc_indices"],
                    "coarse_topk_docs": [context[i] for i in res["chosen_doc_indices"]],
                    "coarse_topk_ppl": res["chosen_doc_ppl"],
                    "effective_question": res["effective_question"],
                    "compressed_prompt": res["compressed_prompt"],
                    "origin_tokens": res["origin_tokens"],
                    "compressed_tokens": res["compressed_tokens"],
                    "ratio": res["ratio"],
                    "rate": res["rate"],
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

    output = {
        "config": {
            "trace_json": str(args.trace_json),
            "model_name": args.model_name,
            "device_map": args.device_map,
            "top_k": args.top_k,
            "instruction": args.instruction,
            "rate": args.rate,
            "target_token": args.target_token,
            "iterative_size": args.iterative_size,
            "condition_in_question": args.condition_in_question,
            "condition_text": args.condition_text,
            "condition_placement": args.condition_placement,
            "use_sentence_level_filter": args.use_sentence_level_filter,
            "use_token_level_filter": not args.disable_token_level_filter,
            "keep_split": args.keep_split,
            "max_trace_items": args.max_trace_items,
            "max_entries_per_item": args.max_entries_per_item,
        },
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved filtered results to: {args.output_json}")


if __name__ == "__main__":
    main()
