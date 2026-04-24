#!/usr/bin/env python3
"""Coarse-to-fine LongLLMLingua filtering over SimpleMem extraction traces.

Given a ``*_extraction_trace.json`` produced by ``test_locomo10.py``, this script:
1. Groups adjacent dialogue turns into coarse windows.
2. For every extracted entry, uses PPL-based coarse ranking over turn windows.
3. Takes coarse top-1 window and runs token-level contrastive filtering with a budget.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from llmlingua import PromptCompressor


class TopKPPLPromptCompressor(PromptCompressor):
    """PromptCompressor extension with explicit top-k PPL coarse filtering."""

    @staticmethod
    def save_ppl_values_plot(
        ppl_values: list[float],
        plot_dir: Path,
        plot_file_stem: str,
        title: str = "PPL values by window",
    ) -> Path | None:
        if not ppl_values:
            return None

        plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = plot_dir / f"{plot_file_stem}.pdf"

        x = list(range(len(ppl_values)))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, ppl_values, marker="o", linewidth=1.5)
        ax.set_xlabel("Window index")
        ax.set_ylabel("PPL value")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_path, format="pdf")
        plt.close(fig)
        return output_path

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

    @staticmethod
    def build_turn_windows(context: list[str], window_k: int, turn_sep: str) -> list[dict[str, Any]]:
        if window_k <= 0:
            raise ValueError("window_k must be positive")
        windows: list[dict[str, Any]] = []
        # Build stride-1 sliding windows, e.g. k=3 -> [0,1,2], [1,2,3], ...
        # If context is shorter than window_k, keep a single short window.
        if len(context) <= window_k:
            starts = [0] if context else []
        else:
            starts = range(0, len(context) - window_k + 1)

        for start in starts:
            end = min(start + window_k, len(context))
            turns = [str(t) for t in context[start:end]]
            windows.append(
                {
                    "window_index": len(windows),
                    "turn_start": start,
                    "turn_end_exclusive": end,
                    "turn_indices": list(range(start, end)),
                    "text": turn_sep.join(turns),
                }
            )
        return windows

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

    def fine_topk_by_contrastive_ppl(
        self,
        context: list[str],
        question: str,
        top_k: int,
        condition_text: str,
        condition_placement: str,
    ) -> tuple[list[int], list[float]]:
        """Rank docs by mean token-level contrastive PPL (lower is better)."""
        if not context:
            return [], []

        effective_question = self.build_effective_question(
            question=question,
            condition_text=condition_text,
            condition_placement=condition_placement,
        )

        score_values: list[float] = []
        for doc in context:
            plain_loss = self.get_ppl(doc, granularity="token")

            if effective_question:
                q_ids = self.tokenizer(effective_question, return_tensors="pt")["input_ids"].to(self.device)
                doc_ids = self.tokenizer(doc, return_tensors="pt")["input_ids"].to(self.device)
                combined_ids = torch.cat([q_ids, doc_ids], dim=1)
                combined_attention = torch.ones_like(combined_ids)
                cond_loss = self.get_ppl(
                    text="",
                    granularity="token",
                    input_ids=combined_ids,
                    attention_mask=combined_attention,
                    condition_mode="after",
                    condition_pos_id=q_ids.shape[1] - 1,
                )
                cond_doc_loss = cond_loss[: plain_loss.shape[0]]
            else:
                cond_doc_loss = plain_loss

            contrastive = (cond_doc_loss - plain_loss).detach().cpu()
            score_values.append(float(torch.mean(contrastive).item()))

        ranked_indices = sorted(range(len(context)), key=lambda idx: score_values[idx])
        keep_n = min(max(top_k, 1), len(context))
        return ranked_indices[:keep_n], score_values

    def token_level_contrastive_compress(
        self,
        context_text: str,
        entry_text: str,
        condition_text: str,
        condition_placement: str,
        budget_multiplier: float,
    ) -> dict[str, Any]:
        effective_question = self.build_effective_question(
            question=entry_text,
            condition_text=condition_text,
            condition_placement=condition_placement,
        )

        tokenized_text = self.tokenizer(context_text, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(self.device)
        text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        if input_ids.shape[1] <= 1:
            return {
                "compressed_prompt": context_text,
                "origin_tokens": int(input_ids.shape[1]),
                "compressed_tokens": int(input_ids.shape[1]),
                "ratio": 1.0,
                "contrastive_token_scores": [],
                "selected_token_positions": list(range(int(input_ids.shape[1]))),
            }

        plain_loss = self.get_ppl(context_text, granularity="token")

        if effective_question:
            q_ids = self.tokenizer(effective_question, return_tensors="pt")["input_ids"].to(self.device)
            combined_ids = torch.cat([q_ids, input_ids], dim=1)
            combined_attention = torch.ones_like(combined_ids)
            cond_loss = self.get_ppl(
                text="",
                granularity="token",
                input_ids=combined_ids,
                attention_mask=combined_attention,
                condition_mode="after",
                condition_pos_id=q_ids.shape[1] - 1,
            )
            cond_text_loss = cond_loss[: plain_loss.shape[0]]
        else:
            cond_text_loss = plain_loss

        contrastive = (cond_text_loss - plain_loss).detach().cpu()

        entry_tokens = max(self.get_token_length(entry_text), 1)
        budget = max(1, int(math.ceil(entry_tokens * budget_multiplier)))
        available = int(plain_loss.shape[0])
        keep_n = min(budget, available)

        top_positions = torch.argsort(contrastive, descending=False)[:keep_n].tolist()
        selected_positions = sorted(top_positions)

        selected_tokens = [text_tokens[pos + 1] for pos in selected_positions]
        compressed_prompt = self.tokenizer.convert_tokens_to_string(selected_tokens)

        return {
            "compressed_prompt": compressed_prompt,
            "origin_tokens": int(input_ids.shape[1]),
            "compressed_tokens": len(selected_tokens),
            "ratio": (len(selected_tokens) / max(int(input_ids.shape[1]), 1)),
            "contrastive_token_scores": [float(contrastive[pos].item()) for pos in selected_positions],
            "selected_token_positions": selected_positions,
            "effective_question": effective_question,
            "entry_token_budget": budget,
        }

    def compress_with_coarse_topk(
        self,
        context: list[str],
        entry_text: str,
        top_k: int,
        condition_in_question: str,
        condition_text: str,
        condition_placement: str,
        turn_window_k: int,
        turn_separator: str,
        entry_budget_multiplier: float,
        first_stage_filter: str = "coarse_topk_by_ppl",
        ppl_plot_dir: Path | None = None,
        ppl_plot_file_stem: str | None = None,
    ) -> dict[str, Any]:
        turn_windows = self.build_turn_windows(context=context, window_k=turn_window_k, turn_sep=turn_separator)
        window_texts = [w["text"] for w in turn_windows]

        if first_stage_filter == "coarse_topk_by_ppl":
            chosen_indices, ppl_values = self.coarse_topk_by_ppl(
                context=window_texts,
                question=entry_text,
                top_k=top_k,
                condition_in_question=condition_in_question,
                condition_text=condition_text,
                condition_placement=condition_placement,
            )
        elif first_stage_filter == "fine_topk_by_contrastive_ppl":
            chosen_indices, ppl_values = self.fine_topk_by_contrastive_ppl(
                context=window_texts,
                question=entry_text,
                top_k=top_k,
                condition_text=condition_text,
                condition_placement=condition_placement,
            )
        else:
            raise ValueError(
                f"first_stage_filter must be one of [coarse_topk_by_ppl, fine_topk_by_contrastive_ppl], got {first_stage_filter}"
            )

        ppl_plot_path: str | None = None
        if ppl_plot_dir is not None and ppl_plot_file_stem:
            saved_plot = self.save_ppl_values_plot(
                ppl_values=ppl_values,
                plot_dir=ppl_plot_dir,
                plot_file_stem=ppl_plot_file_stem,
                title=f"PPL values by window ({first_stage_filter})",
            )
            if saved_plot is not None:
                ppl_plot_path = str(saved_plot)

        if not chosen_indices:
            return {
                "chosen_doc_indices": [],
                "chosen_doc_ppl": [],
                "chosen_windows": [],
                "coarse_merged_window": {},
                "effective_question": self.build_effective_question(entry_text, condition_text, condition_placement),
                "original_question": entry_text,
                "compressed_prompt": "",
                "origin_tokens": 0,
                "compressed_tokens": 0,
                "ratio": 0.0,
                "entry_token_budget": 0,
                "selected_token_positions": [],
                "contrastive_token_scores": [],
                "ppl_plot_path": ppl_plot_path,
            }

        chosen_windows = [turn_windows[i] for i in chosen_indices]
        merged_turn_indices = sorted(
            {
                turn_idx
                for window in chosen_windows
                for turn_idx in window.get("turn_indices", [])
            }
        )
        merged_context_text = turn_separator.join([str(context[i]) for i in merged_turn_indices]) if merged_turn_indices else ""

        fine = self.token_level_contrastive_compress(
            context_text=merged_context_text,
            entry_text=entry_text,
            condition_text=condition_text,
            condition_placement=condition_placement,
            budget_multiplier=entry_budget_multiplier,
        )

        return {
            **fine,
            "chosen_doc_indices": chosen_indices,
            "chosen_doc_ppl": [ppl_values[i] for i in chosen_indices],
            "chosen_windows": chosen_windows,
            "coarse_top1_window": chosen_windows[0],
            "coarse_merged_window": {
                "window_index": -1,
                "turn_indices": merged_turn_indices,
                "text": merged_context_text,
            },
            "original_question": entry_text,
            "effective_question": fine["effective_question"],
            "ppl_plot_path": ppl_plot_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coarse turn-window top-k + fine token-level contrastive LongLLMLingua filtering for extraction traces."
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

    parser.add_argument("--top-k", type=int, default=1, help="Number of coarse turn windows kept by PPL ranking")
    parser.add_argument("--turn-window-k", type=int, default=2, help="Number of adjacent turns merged into one coarse window")
    parser.add_argument("--turn-separator", type=str, default="\n", help="Separator used to concatenate turns in each coarse window")
    parser.add_argument(
        "--entry-token-budget-multiplier",
        type=float,
        default=1.0,
        help="Fine-stage token budget multiplier relative to entry token length",
    )

    parser.add_argument(
        "--condition-in-question",
        choices=["none", "before", "after"],
        default="after",
        help="Conditioning mode used in coarse PPL computation",
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

    parser.add_argument("--max-trace-items", type=int, default=-1, help="Only process first N trace items, -1 for all")
    parser.add_argument("--max-entries-per-item", type=int, default=-1, help="Only process first N extracted entries per trace item")
    parser.add_argument(
        "--ppl-plot-subdir",
        type=str,
        default="ppl_plots",
        help="Subdirectory (under output_json parent) used to save per-entry coarse-window PPL PDF plots; empty disables plotting",
    )
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
    ppl_plot_dir = (
        args.output_json.parent / args.ppl_plot_subdir
        if str(args.ppl_plot_subdir).strip()
        else None
    )

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
                turn_window_k=args.turn_window_k,
                turn_separator=args.turn_separator,
                entry_budget_multiplier=args.entry_token_budget_multiplier,
                ppl_plot_dir=ppl_plot_dir,
                ppl_plot_file_stem=f"trace_{item_idx:04d}_entry_{entry_idx:04d}",
            )
            entry_results.append(
                {
                    "entry_index": entry_idx,
                    "entry_text": entry_text,
                    "coarse_topk_indices": res["chosen_doc_indices"],
                    "coarse_topk_windows": res["chosen_windows"],
                    "coarse_topk_ppl": res["chosen_doc_ppl"],
                    "coarse_top1_window": res.get("coarse_top1_window", {}),
                    "coarse_merged_window": res.get("coarse_merged_window", {}),
                    "effective_question": res["effective_question"],
                    "compressed_prompt": res["compressed_prompt"],
                    "origin_tokens": res["origin_tokens"],
                    "compressed_tokens": res["compressed_tokens"],
                    "ratio": res["ratio"],
                    "entry_token_budget": res["entry_token_budget"],
                    "selected_token_positions": res["selected_token_positions"],
                    "contrastive_token_scores": res["contrastive_token_scores"],
                    "ppl_plot_path": res.get("ppl_plot_path"),
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
            "turn_window_k": args.turn_window_k,
            "turn_separator": args.turn_separator,
            "entry_token_budget_multiplier": args.entry_token_budget_multiplier,
            "condition_in_question": args.condition_in_question,
            "condition_text": args.condition_text,
            "condition_placement": args.condition_placement,
            "max_trace_items": args.max_trace_items,
            "max_entries_per_item": args.max_entries_per_item,
            "ppl_plot_subdir": args.ppl_plot_subdir,
        },
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved filtered results to: {args.output_json}")


if __name__ == "__main__":
    main()
