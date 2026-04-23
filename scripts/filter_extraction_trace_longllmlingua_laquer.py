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
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.filter_extraction_trace_longllmlingua import TopKPPLPromptCompressor, load_trace

try:
    from laquer_methods.llm_method import LLMBasedAlignment  # type: ignore
except Exception:  # pragma: no cover - fallback when LAQuer source is not installed

    class LLMBasedAlignment:  # type: ignore[override]
        """Small compatibility base class mirroring LAQuer's LLM alignment role."""

        def __init__(self, model_name: str, client: OpenAI, temperature: float = 0.0, max_tokens: int = 800):
            self.model_name = model_name
            self.client = client
            self.temperature = temperature
            self.max_tokens = max_tokens

        def build_prompt(self, query: str, context_turns: list[dict[str, Any]]) -> str:
            raise NotImplementedError

        def call_llm(self, prompt: str) -> str:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful evidence span extractor. Output valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()

        def parse_response(self, response_text: str) -> list[dict[str, Any]]:
            try:
                obj = json.loads(response_text)
            except json.JSONDecodeError:
                m = re.search(r"\{[\s\S]*\}", response_text)
                if not m:
                    return []
                obj = json.loads(m.group(0))
            spans = obj.get("spans", []) if isinstance(obj, dict) else []
            return spans if isinstance(spans, list) else []

        def align(self, query: str, context_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
            prompt = self.build_prompt(query=query, context_turns=context_turns)
            return self.parse_response(self.call_llm(prompt))


@dataclass
class LLMConfig:
    model_name: str
    api_key: str
    base_url: str | None
    temperature: float
    max_tokens: int


class LoCoMoLAQuerAlignment(LLMBasedAlignment):
    """LAQuer-style span miner customized for LoCoMo-like dialogue turns."""

    def build_prompt(self, query: str, context_turns: list[dict[str, Any]]) -> str:
        examples = self._few_shot_examples()
        context_block = "\n".join(
            f"[Turn {t['turn_index']}] {t['speaker']}: {t['text']}" for t in context_turns
        )
        return (
            "Task: Given one extracted memory entry and a dialogue snippet, find minimal supporting spans.\n"
            "Return strict JSON only: {\"spans\": [{\"turn_index\": int, \"span_text\": str, "
            "\"reason\": str, \"confidence\": float}]}.\n"
            "Rules:\n"
            "1) span_text must be verbatim from one turn.\n"
            "2) Prefer shortest span that still supports the entry.\n"
            "3) If unsupported, return {\"spans\": []}.\n"
            "4) confidence in [0,1].\n\n"
            f"{examples}\n\n"
            f"Entry:\n{query}\n\n"
            f"Dialogue:\n{context_block}\n\n"
            "Now output JSON."
        )

    def _few_shot_examples(self) -> str:
        return (
            "Example 1\n"
            "Entry: Alex's flight to Seattle is on Friday morning.\n"
            "Dialogue:\n"
            "[Turn 0] User: I finally booked my Seattle trip; the flight is this Friday at 8 a.m.\n"
            "[Turn 1] Friend: Nice, hope the weather is good.\n"
            "Output:\n"
            "{\"spans\":[{\"turn_index\":0,\"span_text\":\"Seattle trip; the flight is this Friday at 8 a.m.\","
            "\"reason\":\"States destination and time\",\"confidence\":0.96}]}\n\n"
            "Example 2\n"
            "Entry: Mia moved to Chicago last year.\n"
            "Dialogue:\n"
            "[Turn 0] Mia: I'm still in Austin for now.\n"
            "[Turn 1] Mia: I might move someday, but no plans.\n"
            "Output:\n"
            "{\"spans\":[]}"
        )


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
    parser.add_argument("--condition-in-question", choices=["none", "before", "after"], default="after")
    parser.add_argument("--condition-text", type=str, default="")
    parser.add_argument("--condition-placement", choices=["none", "prepend", "append"], default="prepend")

    parser.add_argument("--llm-model-name", type=str, required=True)
    parser.add_argument("--llm-api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--llm-base-url", type=str, default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=800)

    parser.add_argument("--max-trace-items", type=int, default=-1)
    parser.add_argument("--max-entries-per-item", type=int, default=-1)
    parser.add_argument("--min-span-confidence", type=float, default=0.0)
    return parser.parse_args()


def build_llm_config(args: argparse.Namespace) -> LLMConfig:
    if not args.llm_api_key:
        raise ValueError("llm-api-key missing. Provide --llm-api-key or OPENAI_API_KEY")
    return LLMConfig(
        model_name=args.llm_model_name,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
    )


def create_aligner(cfg: LLMConfig) -> LoCoMoLAQuerAlignment:
    client_kwargs: dict[str, Any] = {"api_key": cfg.api_key}
    if cfg.base_url:
        client_kwargs["base_url"] = cfg.base_url
    client = OpenAI(**client_kwargs)
    return LoCoMoLAQuerAlignment(
        model_name=cfg.model_name,
        client=client,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


def normalize_spans(raw_spans: list[dict[str, Any]], context_turns: list[dict[str, Any]], min_conf: float) -> list[dict[str, Any]]:
    idx2turn = {t["turn_index"]: t for t in context_turns}
    normalized: list[dict[str, Any]] = []

    for span in raw_spans:
        if not isinstance(span, dict):
            continue
        try:
            turn_index = int(span.get("turn_index"))
        except Exception:
            continue
        turn = idx2turn.get(turn_index)
        if not turn:
            continue

        span_text = str(span.get("span_text", "")).strip()
        if not span_text:
            continue

        confidence = float(span.get("confidence", 0.0) or 0.0)
        if confidence < min_conf:
            continue

        source = turn["text"]
        start = source.find(span_text)
        if start < 0:
            # fallback: keep but mark unmatched; helps diagnose LLM hallucinated quote
            start, end = -1, -1
        else:
            end = start + len(span_text)

        normalized.append(
            {
                "turn_index": turn_index,
                "speaker": turn["speaker"],
                "span_text": span_text,
                "reason": str(span.get("reason", "")),
                "confidence": confidence,
                "char_start": start,
                "char_end": end,
                "is_verbatim_match": start >= 0,
            }
        )

    normalized.sort(key=lambda x: (x["turn_index"], -(x["confidence"])))
    return normalized


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
    llm_cfg = build_llm_config(args)

    trace_data = load_trace(args.trace_json)
    compressor = TopKPPLPromptCompressor(
        model_name=args.compressor_model_name,
        device_map=args.compressor_device_map,
    )
    aligner = create_aligner(llm_cfg)

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
            )

            top1_window = coarse.get("coarse_top1_window", {})
            support_turns = turn_window_to_context_turns(top1_window, [str(c) for c in context]) if top1_window else []

            raw_spans = aligner.align(query=entry_text, context_turns=support_turns) if support_turns else []
            spans = normalize_spans(raw_spans=raw_spans, context_turns=support_turns, min_conf=args.min_span_confidence)

            entry_results.append(
                {
                    "entry_index": entry_idx,
                    "entry_text": entry_text,
                    "coarse_topk_indices": coarse["chosen_doc_indices"],
                    "coarse_topk_windows": coarse["chosen_windows"],
                    "coarse_topk_ppl": coarse["chosen_doc_ppl"],
                    "coarse_top1_window": top1_window,
                    "llm_raw_spans": raw_spans,
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

    output = {
        "config": {
            "trace_json": str(args.trace_json),
            "compressor_model_name": args.compressor_model_name,
            "compressor_device_map": args.compressor_device_map,
            "top_k": args.top_k,
            "turn_window_k": args.turn_window_k,
            "turn_separator": args.turn_separator,
            "condition_in_question": args.condition_in_question,
            "condition_text": args.condition_text,
            "condition_placement": args.condition_placement,
            "llm_model_name": args.llm_model_name,
            "llm_base_url": args.llm_base_url,
            "llm_temperature": args.llm_temperature,
            "llm_max_tokens": args.llm_max_tokens,
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
