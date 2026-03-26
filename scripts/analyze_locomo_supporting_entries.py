#!/usr/bin/env python3
"""Analyze LoCoMo support using the real SimpleMem retrieval pipeline.

This script aligns with `test_locomo10.py` retrieval logic:
- Initialize `SimpleMemSystem`
- Rebuild per-sample memory DB by loading `locomo10_sample_{idx}_memory_entries.json`
- Retrieve with `self.system.hybrid_retriever.retrieve(question)`
- Run incremental k analysis and LLM judging
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openai import OpenAI

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import SimpleMemSystem
    from models.memory_entry import MemoryEntry

DEFAULT_K_VALUES = [5, 10, 15, 20, 25, 30, 40, 50]
DEFAULT_CATEGORIES = [1, 2, 4]


@dataclass
class QAItem:
    question: str
    answer: str
    category: int
    qa_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LoCoMo support with SimpleMem hybrid_retriever.retrieve().")
    parser.add_argument("--dataset-json", type=Path, default=Path("test_ref/locomo10.json"), help="LoCoMo dataset JSON path.")
    parser.add_argument("--entries-dir", type=Path, default=Path("."), help="Directory with locomo10_sample_*_memory_entries.json files.")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/locomo_support_analysis.json"), help="Output JSON report path.")
    parser.add_argument("--categories", type=int, nargs="+", default=DEFAULT_CATEGORIES, help="QA categories to analyze.")
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES, help="Top-k values to evaluate.")
    parser.add_argument("--sample-indices", type=int, nargs="*", default=None, help="Optional subset of sample indices.")

    parser.add_argument("--model", type=str, required=True, help="LLM model name.")
    parser.add_argument("--base-url", type=str, required=True, help="LLM API base URL.")
    parser.add_argument("--api-key", type=str, default=None, help="LLM API key. If omitted, use OPENAI_API_KEY.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature for judge call.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per LLM judge call.")
    parser.add_argument("--entry-text-max-chars", type=int, default=400, help="Truncate each entry text in prompt.")
    parser.add_argument("--storage-check-top-n", type=int, default=120, help="Large-k retrieval budget for storage-level check.")

    parser.add_argument("--disable-reflection", action="store_true", help="Disable retrieval reflection for deterministic behavior.")
    parser.add_argument("--disable-planning", action="store_true", help="Disable retrieval planning and use semantic fallback only.")
    return parser.parse_args()


def parse_json_from_response(text: str) -> dict[str, Any]:
    text = text.strip()
    for pattern in [None, r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```", r"(\{.*\})"]:
        try:
            if pattern is None:
                obj = json.loads(text)
            else:
                m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if not m:
                    continue
                obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(f"Failed to parse judge JSON: {text[:300]}")


def entry_text(entry: dict[str, Any]) -> str:
    fields = ["lossless_restatement", "topic", "location", "timestamp"]
    pieces = [str(entry.get(field, "")) for field in fields if entry.get(field)]
    for field in ("keywords", "persons", "entities"):
        val = entry.get(field, [])
        if isinstance(val, list):
            pieces.extend(str(x) for x in val if x)
    return " ".join(pieces)


def to_prompt_entries(entries: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, e in enumerate(entries):
        txt = entry_text(e)
        if len(txt) > max_chars:
            txt = txt[:max_chars] + " ...[truncated]"
        out.append(
            {
                "idx": idx,
                "entry_id": e.get("entry_id"),
                "text": txt,
                "keywords": e.get("keywords", []),
                "persons": e.get("persons", []),
                "entities": e.get("entities", []),
                "timestamp": e.get("timestamp"),
                "location": e.get("location"),
            }
        )
    return out


def llm_judge_support(
    *,
    client: OpenAI,
    model: str,
    question: str,
    answer: str,
    entries: list[dict[str, Any]],
    temperature: float,
    max_retries: int,
    entry_text_max_chars: int,
) -> dict[str, Any]:
    payload = {
        "task": "Judge whether retrieved entries are sufficient to answer the question using the gold answer.",
        "rules": [
            "Return strict JSON only.",
            "If insufficient, set can_answer=false and minimal_supporting_entry_indices=[].",
            "If sufficient, return minimal supporting indices (fewest idx values).",
        ],
        "schema": {
            "can_answer": "boolean",
            "reason": "string",
            "confidence": "number in [0,1]",
            "minimal_supporting_entry_indices": [0],
            "predicted_answer": "string",
        },
        "question": question,
        "gold_answer": answer,
        "entries": to_prompt_entries(entries, entry_text_max_chars),
    }
    messages = [
        {"role": "system", "content": "You are an expert evaluator for retrieval support analysis. Return strict JSON."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            judge = parse_json_from_response(rsp.choices[0].message.content or "")
            raw_idxs = judge.get("minimal_supporting_entry_indices", [])
            if not isinstance(raw_idxs, list):
                raw_idxs = []
            idxs = sorted({int(i) for i in raw_idxs if isinstance(i, int) and 0 <= i < len(entries)})
            return {
                "can_answer": bool(judge.get("can_answer", False)),
                "reason": str(judge.get("reason", "")),
                "confidence": float(judge.get("confidence", 0.0)),
                "predicted_answer": str(judge.get("predicted_answer", "")),
                "minimal_supporting_entry_indices": idxs,
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(1.5**attempt)

    raise RuntimeError(f"LLM judge failed after retries: {last_err}")


def load_entries_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a list of entry dicts")
    return [x for x in data if isinstance(x, dict)]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of samples")
    return data


def extract_qas(sample: dict[str, Any], categories: set[int]) -> list[QAItem]:
    out: list[QAItem] = []
    qas = sample.get("qa", [])
    if not isinstance(qas, list):
        return out
    for qa_idx, qa in enumerate(qas):
        if not isinstance(qa, dict):
            continue
        cat = qa.get("category")
        if cat not in categories:
            continue
        q = str(qa.get("question", "")).strip()
        a = str(qa.get("answer", "")).strip()
        if q and a:
            out.append(QAItem(question=q, answer=a, category=int(cat), qa_idx=qa_idx))
    return out


def to_support_record(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "entry_id": e.get("entry_id"),
            "lossless_restatement": e.get("lossless_restatement", ""),
            "keywords": e.get("keywords", []),
        }
        for e in entries
    ]


def entry_dict_to_model(entry: dict[str, Any]) -> "MemoryEntry":
    from models.memory_entry import MemoryEntry

    return MemoryEntry(
        entry_id=str(entry.get("entry_id") or uuid.uuid4()),
        lossless_restatement=str(entry.get("lossless_restatement", "")),
        keywords=[str(x) for x in entry.get("keywords", []) if x is not None],
        timestamp=str(entry.get("timestamp")) if entry.get("timestamp") else None,
        location=str(entry.get("location")) if entry.get("location") else None,
        persons=[str(x) for x in entry.get("persons", []) if x is not None],
        entities=[str(x) for x in entry.get("entities", []) if x is not None],
        topic=str(entry.get("topic")) if entry.get("topic") else None,
    )


def load_entries_into_system(system: "SimpleMemSystem", entries_raw: list[dict[str, Any]]) -> None:
    system.vector_store.clear()
    models = [entry_dict_to_model(e) for e in entries_raw]
    if models:
        system.vector_store.add_entries(models)


def retrieve_with_hybrid(system: "SimpleMemSystem", question: str, k: int) -> list[dict[str, Any]]:
    retriever = system.hybrid_retriever
    retriever.semantic_top_k = k
    retriever.keyword_top_k = k
    retriever.structured_top_k = k

    contexts = retriever.retrieve(question)
    dedup = []
    seen_ids: set[str] = set()
    for item in contexts:
        if item.entry_id in seen_ids:
            continue
        seen_ids.add(item.entry_id)
        dedup.append(item)
        if len(dedup) >= k:
            break

    out: list[dict[str, Any]] = []
    for m in dedup:
        out.append(
            {
                "entry_id": m.entry_id,
                "lossless_restatement": m.lossless_restatement,
                "keywords": list(m.keywords),
                "timestamp": m.timestamp,
                "location": m.location,
                "persons": list(m.persons),
                "entities": list(m.entities),
                "topic": m.topic,
            }
        )
    return out


def analyze_sample(
    *,
    system: "SimpleMemSystem",
    judge_client: OpenAI,
    judge_model: str,
    sample_idx: int,
    sample: dict[str, Any],
    categories: set[int],
    k_values: list[int],
    temperature: float,
    max_retries: int,
    entry_text_max_chars: int,
    storage_check_top_n: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for qa in extract_qas(sample, categories):
        item: dict[str, Any] = {
            "sample_idx": sample_idx,
            "qa_idx": qa.qa_idx,
            "category": qa.category,
            "query": qa.question,
            "answer": qa.answer,
            "stored_support_exists": False,
            "stored_support_set": [],
            "k_evaluations": [],
            "first_success_k": None,
            "retrieval_issue": False,
        }

        storage_pool = retrieve_with_hybrid(system, qa.question, storage_check_top_n)
        storage_judge = llm_judge_support(
            client=judge_client,
            model=judge_model,
            question=qa.question,
            answer=qa.answer,
            entries=storage_pool,
            temperature=temperature,
            max_retries=max_retries,
            entry_text_max_chars=entry_text_max_chars,
        )
        if storage_judge["can_answer"]:
            supporting = [storage_pool[i] for i in storage_judge["minimal_supporting_entry_indices"]]
            item["stored_support_exists"] = True
            item["stored_support_set"] = to_support_record(supporting)

        for k in k_values:
            retrieved = retrieve_with_hybrid(system, qa.question, k)
            judge = llm_judge_support(
                client=judge_client,
                model=judge_model,
                question=qa.question,
                answer=qa.answer,
                entries=retrieved,
                temperature=temperature,
                max_retries=max_retries,
                entry_text_max_chars=entry_text_max_chars,
            )
            supporting = [retrieved[i] for i in judge["minimal_supporting_entry_indices"]]
            item["k_evaluations"].append(
                {
                    "k": k,
                    "retrieved_count": len(retrieved),
                    "can_answer": judge["can_answer"],
                    "judge_detail": {
                        "reason": judge["reason"],
                        "confidence": judge["confidence"],
                        "predicted_answer": judge["predicted_answer"],
                    },
                    "minimal_supporting_set": to_support_record(supporting),
                }
            )
            if judge["can_answer"]:
                item["first_success_k"] = k
                break

        if item["first_success_k"] is None:
            item["retrieval_issue"] = True
            item["failure_type"] = (
                "retrieval_failed_within_k_budget" if item["stored_support_exists"] else "missing_or_insufficient_memory"
            )

        results.append(item)

    return results


def summarize(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(all_results)
    if total == 0:
        return {"total_queries": 0}

    by_cat: dict[str, dict[str, int]] = {}
    success_count = 0
    retrieval_issue_count = 0
    memory_missing_count = 0

    for r in all_results:
        cat = str(r["category"])
        by_cat.setdefault(cat, {"count": 0, "success": 0, "retrieval_issue": 0})
        by_cat[cat]["count"] += 1

        if r.get("first_success_k") is not None:
            success_count += 1
            by_cat[cat]["success"] += 1
        if r.get("retrieval_issue"):
            retrieval_issue_count += 1
            by_cat[cat]["retrieval_issue"] += 1
        if r.get("failure_type") == "missing_or_insufficient_memory":
            memory_missing_count += 1

    return {
        "total_queries": total,
        "success_count": success_count,
        "success_rate": success_count / total,
        "retrieval_issue_count": retrieval_issue_count,
        "memory_missing_or_insufficient_count": memory_missing_count,
        "by_category": by_cat,
    }


def main() -> None:
    args = parse_args()
    args.k_values = sorted(set(args.k_values))

    if not args.dataset_json.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_json}")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Pass --api-key or set OPENAI_API_KEY.")

    judge_client = OpenAI(base_url=args.base_url, api_key=api_key)

    from main import SimpleMemSystem

    system = SimpleMemSystem(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        clear_db=True,
        enable_planning=not args.disable_planning,
        enable_reflection=not args.disable_reflection,
    )

    dataset = load_dataset(args.dataset_json)
    categories = set(args.categories)
    selected_indices = set(args.sample_indices) if args.sample_indices else None

    all_results: list[dict[str, Any]] = []
    missing_entry_files: list[int] = []

    for sample_idx, sample in enumerate(dataset):
        if selected_indices is not None and sample_idx not in selected_indices:
            continue

        file_path = args.entries_dir / f"locomo10_sample_{sample_idx}_memory_entries.json"
        if not file_path.exists():
            missing_entry_files.append(sample_idx)
            continue

        entries = load_entries_file(file_path)
        load_entries_into_system(system, entries)

        sample_results = analyze_sample(
            system=system,
            judge_client=judge_client,
            judge_model=args.model,
            sample_idx=sample_idx,
            sample=sample,
            categories=categories,
            k_values=args.k_values,
            temperature=args.temperature,
            max_retries=args.max_retries,
            entry_text_max_chars=args.entry_text_max_chars,
            storage_check_top_n=args.storage_check_top_n,
        )
        all_results.extend(sample_results)

    report = {
        "config": {
            "dataset_json": str(args.dataset_json),
            "entries_dir": str(args.entries_dir),
            "categories": sorted(categories),
            "k_values": args.k_values,
            "sample_indices": sorted(selected_indices) if selected_indices is not None else None,
            "retrieval_mode": "simplemem_system_hybrid_retriever",
            "retrieval_api": "system.hybrid_retriever.retrieve",
            "llm": {
                "model": args.model,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "max_retries": args.max_retries,
                "entry_text_max_chars": args.entry_text_max_chars,
                "storage_check_top_n": args.storage_check_top_n,
            },
        },
        "missing_entry_files": missing_entry_files,
        "summary": summarize(all_results),
        "results": all_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[Done] analyzed queries: {len(all_results)}")
    print(f"[Done] report saved: {args.output_json}")
    if missing_entry_files:
        print(f"[Warn] missing entry files for sample indices: {missing_entry_files}")


if __name__ == "__main__":
    main()
