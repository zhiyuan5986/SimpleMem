#!/usr/bin/env python3
"""Recompute LoCoMo evidence recall from test_locomo10_dualview_qa.py JSON outputs.

This script supports grid search over:
- mem-sem-weight / mem-lex-weight
- raw-sem-weight / raw-lex-weight
- final-mem-weight / final-raw-weight / final-agree-weight
- top-n (offline cutoff)
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

RECALL_CATEGORIES = {1, 2, 3, 4}


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _safe_scores(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scores = result.get("dualview_scores_all") or result.get("dualview_scores") or {}
    return scores if isinstance(scores, dict) else {}


def _safe_entry_to_dia(result: dict[str, Any]) -> dict[str, list[str]]:
    mapping = result.get("entry_id_to_dia_ids") or {}
    if not isinstance(mapping, dict):
        return {}
    fixed: dict[str, list[str]] = {}
    for k, v in mapping.items():
        if isinstance(v, list):
            fixed[str(k)] = [str(x) for x in v]
    return fixed


def recompute_recall_for_result(
    result: dict[str, Any],
    mem_sem_weight: float,
    mem_lex_weight: float,
    raw_sem_weight: float,
    raw_lex_weight: float,
    final_mem_weight: float,
    final_raw_weight: float,
    final_agree_weight: float,
    top_n: int,
) -> float | None:
    category = result.get("category")
    if category not in RECALL_CATEGORIES:
        return None

    gold_ids = set(str(x) for x in (result.get("gold_dia_ids") or []) if str(x).strip())
    if not gold_ids:
        return None

    scores = _safe_scores(result)
    entry_to_dia = _safe_entry_to_dia(result)
    if not scores or not entry_to_dia:
        return None

    ranked: list[tuple[str, float]] = []
    for entry_id, detail in scores.items():
        mem_sem = float(detail.get("mem_sem", 0.0))
        mem_lex = float(detail.get("mem_lex", 0.0))
        raw_sem = float(detail.get("raw_sem", 0.0))
        raw_lex = float(detail.get("raw_lex", 0.0))

        mem_score = mem_sem_weight * mem_sem + mem_lex_weight * mem_lex
        raw_score = raw_sem_weight * raw_sem + raw_lex_weight * raw_lex
        agree_score = math.sqrt(max(mem_score * raw_score, 0.0))
        final_score = (
            final_mem_weight * mem_score
            + final_raw_weight * raw_score
            + final_agree_weight * agree_score
        )
        ranked.append((entry_id, final_score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    selected_ids = [entry_id for entry_id, _ in ranked[:top_n]]

    predicted_ids: set[str] = set()
    for entry_id in selected_ids:
        predicted_ids.update(entry_to_dia.get(entry_id, []))

    if not predicted_ids:
        return 0.0
    return len(predicted_ids.intersection(gold_ids)) / len(gold_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search dual-view recall from QA result JSON.")
    parser.add_argument("--input", type=str, required=True, help="Path to locomo10_dualview_results.json")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--mem-sem-weights", type=str, required=True, help="e.g. 0.65,0.7")
    parser.add_argument("--mem-lex-weights", type=str, required=True, help="e.g. 0.35,0.3")
    parser.add_argument("--raw-sem-weights", type=str, required=True)
    parser.add_argument("--raw-lex-weights", type=str, required=True)
    parser.add_argument("--final-mem-weights", type=str, required=True)
    parser.add_argument("--final-raw-weights", type=str, required=True)
    parser.add_argument("--final-agree-weights", type=str, required=True)
    parser.add_argument("--top-n-list", type=str, required=True, help="e.g. 3,5,10")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = data.get("results") or []

    mem_sem_weights = parse_float_list(args.mem_sem_weights)
    mem_lex_weights = parse_float_list(args.mem_lex_weights)
    raw_sem_weights = parse_float_list(args.raw_sem_weights)
    raw_lex_weights = parse_float_list(args.raw_lex_weights)
    final_mem_weights = parse_float_list(args.final_mem_weights)
    final_raw_weights = parse_float_list(args.final_raw_weights)
    final_agree_weights = parse_float_list(args.final_agree_weights)
    top_n_list = parse_int_list(args.top_n_list)

    rows: list[dict[str, Any]] = []
    for combo in itertools.product(
        mem_sem_weights,
        mem_lex_weights,
        raw_sem_weights,
        raw_lex_weights,
        final_mem_weights,
        final_raw_weights,
        final_agree_weights,
        top_n_list,
    ):
        mem_sem_w, mem_lex_w, raw_sem_w, raw_lex_w, final_mem_w, final_raw_w, final_agree_w, top_n = combo

        recalls_by_cat: dict[int, list[float]] = {c: [] for c in RECALL_CATEGORIES}
        for result in results:
            recall = recompute_recall_for_result(
                result,
                mem_sem_w,
                mem_lex_w,
                raw_sem_w,
                raw_lex_w,
                final_mem_w,
                final_raw_w,
                final_agree_w,
                top_n,
            )
            cat = result.get("category")
            if recall is not None and cat in RECALL_CATEGORIES:
                recalls_by_cat[cat].append(recall)

        all_vals = [x for vals in recalls_by_cat.values() for x in vals]
        row = {
            "mem_sem_weight": mem_sem_w,
            "mem_lex_weight": mem_lex_w,
            "raw_sem_weight": raw_sem_w,
            "raw_lex_weight": raw_lex_w,
            "final_mem_weight": final_mem_w,
            "final_raw_weight": final_raw_w,
            "final_agree_weight": final_agree_w,
            "top_n": top_n,
            "overall_mean_recall": (sum(all_vals) / len(all_vals)) if all_vals else 0.0,
            "overall_count": len(all_vals),
            "category_recall": {
                str(c): {
                    "mean_recall": (sum(recalls_by_cat[c]) / len(recalls_by_cat[c])) if recalls_by_cat[c] else 0.0,
                    "count": len(recalls_by_cat[c]),
                }
                for c in sorted(RECALL_CATEGORIES)
            },
        }
        rows.append(row)

    rows.sort(key=lambda x: x["overall_mean_recall"], reverse=True)
    output = {
        "input": args.input,
        "num_questions": len(results),
        "num_combinations": len(rows),
        "best": rows[0] if rows else None,
        "combinations": rows,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
