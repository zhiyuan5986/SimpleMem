#!/usr/bin/env python3
"""Compare answering quality using different support entry sets from analysis output.

This script reads the JSON produced by `analyze_locomo_supporting_entries.py` and evaluates two
answering settings per query:
1) only `stored_support_set`
2) entries retrieved at `first_success_k`

For each setting, it generates an answer with an LLM and computes metrics (including token F1),
and LLM-judge correctness using the same judging criteria used in `test_locomo10.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from core.answer_generator import AnswerGenerator
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LoCoMo answering quality across support entry settings.")
    parser.add_argument("--analysis-json", type=Path, required=True, help="Output JSON path from analyze_locomo_supporting_entries.py")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/locomo_support_answer_compare.json"), help="Path to save evaluation results")
    parser.add_argument(
        "--per-sample-output-dir",
        type=Path,
        default=None,
        help="Directory to save per-sample comparison JSON files. Defaults to <output-json-stem>_samples.",
    )
    parser.add_argument(
        "--resume-per-sample",
        action="store_true",
        help="Resume from existing per-sample reports (locomo_support_answer_compare_sample_{idx}.json).",
    )

    parser.add_argument("--answer-model", type=str, required=True, help="LLM model for answer generation")
    parser.add_argument("--answer-base-url", type=str, required=True, help="Base URL for answer generation model")
    parser.add_argument("--answer-api-key", type=str, default=None, help="API key for answer model; fallback OPENAI_API_KEY")
    parser.add_argument("--answer-temperature", type=float, default=0.0, help="Sampling temperature for answer generation")

    parser.add_argument("--judge-model", type=str, required=True, help="LLM model for correctness judging")
    parser.add_argument("--judge-base-url", type=str, required=True, help="Base URL for judge model")
    parser.add_argument("--judge-api-key", type=str, default=None, help="API key for judge model; fallback OPENAI_API_KEY")
    parser.add_argument("--judge-temperature", type=float, default=0.3, help="Sampling temperature for judge")

    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for model calls")
    parser.add_argument("--entry-text-max-chars", type=int, default=500, help="Max chars per entry passed to LLM")
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
    raise ValueError(f"Failed to parse JSON response: {text[:300]}")


def simple_tokenize(text: str) -> list[str]:
    return str(text).lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calculate_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common_tokens = pred_tokens & ref_tokens
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _build_memory_entry(entry: dict[str, Any], max_chars: int) -> MemoryEntry:
    text = str(entry.get("lossless_restatement", "")).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " ...[truncated]"

    return MemoryEntry(
        entry_id=str(entry.get("entry_id", "")),
        lossless_restatement=text,
        keywords=[str(x) for x in (entry.get("keywords") or [])],
        timestamp=entry.get("timestamp"),
        location=entry.get("location"),
        persons=[str(x) for x in (entry.get("persons") or [])],
        entities=[str(x) for x in (entry.get("entities") or [])],
        topic=entry.get("topic"),
    )


def generate_answer(
    *,
    generator: AnswerGenerator,
    question: str,
    entries: list[dict[str, Any]],
    entry_text_max_chars: int,
) -> str:
    contexts = [_build_memory_entry(entry, entry_text_max_chars) for entry in entries]
    return generator.generate_answer(question, contexts)


def llm_judge_answers(
    *,
    client: OpenAI,
    model: str,
    prediction: str,
    reference: str,
    question: str,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    if not prediction or not reference:
        return {"llm_judge_score": 0.0, "llm_reasoning": "Empty prediction or reference"}

    prompt = f"""You are an expert Relevance & Accuracy Evaluator. Your task is to determine if the Predicted Answer successfully retrieves the necessary information to answer the Question, based on the Reference Answer.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Evaluation Criteria:

1. **Responsiveness to Query**:
   The predicted answer must directly address the specific question asked. It must contain highly relevant information that is topically aligned with the user's intent.

2. **Core Fact Preservation**:
   The prediction must capture the "Key Signal" or "Core Entity" from the reference. The primary subject (Who), event (What), or outcome must be factually grounded in the reference text.

3. **Informational Utility**:
   The answer must provide actionable or meaningful value. Even if brief, it must convey the essential message required by the question context.

4. **Acceptable Representational Variances (Robustness Protocol)**:
   To ensure fair evaluation of semantic meaning over syntactic rigidity, you must accept the following variations as **Valid Matches**:
   - **Temporal & Numerical Margins**: Accept timestamps within a reasonable proximity (e.g., +/- 1-2 days due to timezone/reporting differences) and rounded numerical approximations.
   - **Granularity Independence**: Accept answers at different levels of abstraction (e.g., "Afternoon" vs. "14:05", "Late October" vs. "Oct 25th") provided they encompass the truth.
   - **Information Subsetting**: A valid subset of the reference (e.g., mentioning 1 out of 3 reasons) is acceptable if it answers the core of the question.
   - **Synonymy**: Recognize domain-specific synonyms and different formats as equivalent.

Grading Logic:
- Score 1.0 (Pass): The prediction contains relevant core information, answers the question with sufficient utility, OR falls within the acceptable representational variances defined in criterion #4.
- Score 0.0 (Fail): The prediction contains NO relevant information, fails to identify the core subject/event, or provides no key info that matches the question's intent.

Output your evaluation in JSON format:
{{
  "score": 1.0,
  "reasoning": "Brief assessment focusing on information relevance and core match."
}}

Return ONLY the JSON, no other text.
"""

    messages = [
        {"role": "system", "content": "You are an expert evaluator. Always output valid JSON format."},
        {"role": "user", "content": prompt},
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
            result = parse_json_from_response(rsp.choices[0].message.content or "")
            return {
                "llm_judge_score": float(result.get("score", 0.0)),
                "llm_reasoning": str(result.get("reasoning", "No reasoning provided")),
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(1.5**attempt)

    return {"llm_judge_score": 0.0, "llm_reasoning": f"Evaluation failed: {last_err}"}


def evaluate_setting(
    *,
    answer_generator: AnswerGenerator,
    judge_client: OpenAI,
    judge_model: str,
    query: str,
    gold_answer: str,
    entries: list[dict[str, Any]],
    judge_temperature: float,
    max_retries: int,
    entry_text_max_chars: int,
) -> dict[str, Any]:
    prediction = generate_answer(
        generator=answer_generator,
        question=query,
        entries=entries,
        entry_text_max_chars=entry_text_max_chars,
    )
    judge = llm_judge_answers(
        client=judge_client,
        model=judge_model,
        prediction=prediction,
        reference=gold_answer,
        question=query,
        temperature=judge_temperature,
        max_retries=max_retries,
    )
    f1 = calculate_f1(prediction, gold_answer)
    exact_match = int(prediction.strip().lower() == gold_answer.strip().lower())
    return {
        "prediction": prediction,
        "exact_match": exact_match,
        "f1": f1,
        "llm_judge_score": judge["llm_judge_score"],
        "llm_judge_correct": int(judge["llm_judge_score"] >= 0.5),
        "llm_reasoning": judge["llm_reasoning"],
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"count": 0}
    n = len(items)
    return {
        "count": n,
        "accuracy_llm_judge": sum(x["llm_judge_correct"] for x in items) / n,
        "avg_llm_judge_score": sum(x["llm_judge_score"] for x in items) / n,
        "avg_f1": sum(x["f1"] for x in items) / n,
        "exact_match_rate": sum(x["exact_match"] for x in items) / n,
    }


def build_report(args: argparse.Namespace, compared: list[dict[str, Any]]) -> dict[str, Any]:
    stored_metrics = [x["stored_support_eval"] for x in compared if isinstance(x.get("stored_support_eval"), dict)]
    firstk_metrics = [x["first_success_k_eval"] for x in compared if isinstance(x.get("first_success_k_eval"), dict)]
    return {
        "config": {
            "analysis_json": str(args.analysis_json),
            "answer_model": args.answer_model,
            "answer_base_url": args.answer_base_url,
            "judge_model": args.judge_model,
            "judge_base_url": args.judge_base_url,
            "answer_temperature": args.answer_temperature,
            "judge_temperature": args.judge_temperature,
            "max_retries": args.max_retries,
            "entry_text_max_chars": args.entry_text_max_chars,
        },
        "summary": {
            "stored_support_only": summarize(stored_metrics),
            "first_success_k_entries": summarize(firstk_metrics),
        },
        "per_query": compared,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_existing_sample_results(path: Path, sample_idx: int) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Existing sample report is not a JSON object: {path}")
    per_query = payload.get("per_query")
    if not isinstance(per_query, list):
        raise ValueError(f"Existing sample report has invalid 'per_query': {path}")
    normalized: list[dict[str, Any]] = []
    for item in per_query:
        if not isinstance(item, dict):
            continue
        if int(item.get("sample_idx", sample_idx)) != sample_idx:
            continue
        normalized.append(item)
    return normalized


def main() -> None:
    args = parse_args()
    answer_api_key = args.answer_api_key or os.getenv("OPENAI_API_KEY")
    judge_api_key = args.judge_api_key or os.getenv("OPENAI_API_KEY")
    if not answer_api_key or not judge_api_key:
        raise ValueError("Missing API key. Pass --answer-api-key/--judge-api-key or set OPENAI_API_KEY.")

    with args.analysis_json.open("r", encoding="utf-8") as f:
        analysis = json.load(f)

    results = analysis.get("results", [])
    if not isinstance(results, list):
        raise ValueError("analysis-json format invalid: field 'results' must be a list")

    answer_generator = AnswerGenerator(
        llm_client=LLMClient(
            api_key=answer_api_key,
            model=args.answer_model,
            base_url=args.answer_base_url,
            enable_thinking=False,
            use_streaming=False,
        )
    )
    judge_client = OpenAI(base_url=args.judge_base_url, api_key=judge_api_key)

    results_by_sample: dict[int, list[dict[str, Any]]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        sample_idx_raw = item.get("sample_idx")
        if not isinstance(sample_idx_raw, int):
            continue
        results_by_sample.setdefault(sample_idx_raw, []).append(item)

    compared: list[dict[str, Any]] = []
    per_sample_output_dir = args.per_sample_output_dir
    if per_sample_output_dir is None:
        per_sample_output_dir = args.output_json.parent / f"{args.output_json.stem}_samples"

    for sample_idx in sorted(results_by_sample):
        sample_output_path = per_sample_output_dir / f"locomo_support_answer_compare_sample_{sample_idx}.json"
        if args.resume_per_sample:
            existing_results = load_existing_sample_results(sample_output_path, sample_idx)
            if existing_results is not None:
                compared.extend(existing_results)
                print(f"[Resume] skip sample {sample_idx}, loaded {len(existing_results)} records from: {sample_output_path}")
                continue

        sample_compared: list[dict[str, Any]] = []
        for item in results_by_sample[sample_idx]:
            query = str(item.get("query", "")).strip()
            gold = str(item.get("answer", "")).strip()
            if not query or not gold:
                continue

            stored_entries = item.get("stored_support_set") or []
            first_success_k = item.get("first_success_k")
            firstk_entries: list[dict[str, Any]] = []
            if isinstance(first_success_k, int):
                for ev in item.get("k_evaluations", []):
                    if isinstance(ev, dict) and ev.get("k") == first_success_k:
                        firstk_entries = ev.get("retrieved_entries") or []
                        break

            row: dict[str, Any] = {
                "sample_idx": item.get("sample_idx"),
                "qa_idx": item.get("qa_idx"),
                "category": item.get("category"),
                "query": query,
                "answer": gold,
                "first_success_k": first_success_k,
                "stored_support_entry_count": len(stored_entries),
                "first_success_k_entry_count": len(firstk_entries),
            }

            if stored_entries:
                row["stored_support_eval"] = evaluate_setting(
                    answer_generator=answer_generator,
                    judge_client=judge_client,
                    judge_model=args.judge_model,
                    query=query,
                    gold_answer=gold,
                    entries=stored_entries,
                    judge_temperature=args.judge_temperature,
                    max_retries=args.max_retries,
                    entry_text_max_chars=args.entry_text_max_chars,
                )
            else:
                row["stored_support_eval"] = None

            if firstk_entries:
                row["first_success_k_eval"] = evaluate_setting(
                    answer_generator=answer_generator,
                    judge_client=judge_client,
                    judge_model=args.judge_model,
                    query=query,
                    gold_answer=gold,
                    entries=firstk_entries,
                    judge_temperature=args.judge_temperature,
                    max_retries=args.max_retries,
                    entry_text_max_chars=args.entry_text_max_chars,
                )
            else:
                row["first_success_k_eval"] = None

            sample_compared.append(row)

        compared.extend(sample_compared)
        save_json(sample_output_path, build_report(args, sample_compared))
        print(f"[Done] sample report saved: {sample_output_path}")

    save_json(args.output_json, build_report(args, compared))

    print(f"[Done] evaluated queries: {len(compared)}")
    print(f"[Done] output saved: {args.output_json}")


if __name__ == "__main__":
    main()
