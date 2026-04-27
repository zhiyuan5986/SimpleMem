"""
LoCoMo10 QA script using dual-view retrieval:
- MemoryEntryVectorStore (entries)
- RawContextVectorStore (entry-aligned raw evidence)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from core.answer_generator import AnswerGenerator
from core.dual_view_hybrid_retriever import DualViewHybridRetriever
from database.vector_store import MemoryEntryVectorStore, RawContextVectorStore
from test_locomo10 import (
    aggregate_metrics,
    calculate_metrics,
    create_judge_llm_client,
    load_locomo_dataset,
)
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient


RECALL_CATEGORIES = {1, 2, 4}


def _collect_dia_ids_from_obj(obj: Any) -> set[str]:
    found: set[str] = set()

    def _walk(v: Any):
        if isinstance(v, dict):
            for k, val in v.items():
                key = str(k).lower()
                if key in {"dia_id", "dialogue_id", "evidence", "evidences"}:
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, (str, int)):
                                value = str(item).strip()
                                if value:
                                    found.add(value)
                    elif isinstance(val, (str, int)):
                        value = str(val).strip()
                        if value:
                            found.add(value)
                _walk(val)
            return
        if isinstance(v, list):
            for item in v:
                _walk(item)
            return
        if isinstance(v, str):
            for m in re.findall(r"(session_\d+:[^,\s]+)", v):
                value = str(m).strip()
                if value:
                    found.add(value)

    _walk(obj)
    return found


def compute_recall_from_contexts(
    contexts: list,
    qa_evidence: list[str],
    raw_store: RawContextVectorStore,
) -> tuple[float | None, list[str], list[str]]:
    """
    Compute evidence recall against LoCoMo evidence ids.
    Returns: (recall, predicted_ids_sorted, gold_ids_sorted)
    """
    gold_ids: set[str] = set()
    for e in qa_evidence or []:
        value = str(e).strip()
        if value:
            gold_ids.add(value)
    if not gold_ids:
        return None, [], []

    predicted_ids: set[str] = set()
    for ctx in contexts:
        entry_id = getattr(ctx, "entry_id", None)
        if not entry_id:
            continue
        raw_entry = raw_store.get_entry_by_id(entry_id)
        if raw_entry is None:
            continue
        predicted_ids.update(_collect_dia_ids_from_obj(raw_entry.metadata))

    matched = predicted_ids.intersection(gold_ids)
    recall = len(matched) / len(gold_ids) if gold_ids else None
    return recall, sorted(predicted_ids), sorted(gold_ids)


def generate_category5_answer(llm_client: LLMClient, answer_generator: AnswerGenerator, question: str, contexts: list, adversarial_answer: str) -> str:
    options = ["Not mentioned in the conversation", adversarial_answer]
    context_str = answer_generator._format_contexts(contexts)
    prompt = f"""
Based on the context below, answer the following question.

Context:
{context_str}

Question: {question}

Choose one option only:
Option A: {options[0]}
Option B: {options[1]}

Return JSON:
{{"answer": "Option text"}}
"""
    messages = [
        {"role": "system", "content": "Return valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    response = llm_client.chat_completion(messages, temperature=0.2)
    try:
        result = llm_client.extract_json(response)
        return result.get("answer", options[0])
    except Exception:
        return options[0]


def main():
    parser = argparse.ArgumentParser(description="Run LoCoMo10 QA with DualViewHybridRetriever.")
    parser.add_argument("--dataset", type=str, default="test_ref/locomo10.json")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--db-path", type=str, default="./lancedb_data")
    parser.add_argument("--memory-table", type=str, default="memory_entries")
    parser.add_argument("--raw-table", type=str, default="llm_spans")
    parser.add_argument("--result-file", type=str, default="locomo10_dualview_results.json")
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM-as-judge metric")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key")
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model name")
    parser.add_argument("--llm-base-url", type=str, default=None, help="LLM API base URL")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model name/path")
    parser.add_argument(
        "--embedding-use-optimization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable embedding-model optimization (use --no-embedding-use-optimization to disable)",
    )
    parser.add_argument(
        "--answer-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable answer generation and QA metrics (use --no-answer-generation to run recall-only)",
    )
    parser.add_argument("--semantic-top-k", type=int, default=5)
    parser.add_argument("--keyword-top-k", type=int, default=5)
    parser.add_argument("--structured-top-k", type=int, default=3)
    parser.add_argument(
        "--enable-planning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable planning in DualViewHybridRetriever",
    )
    parser.add_argument(
        "--enable-reflection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reflection in DualViewHybridRetriever",
    )
    parser.add_argument("--max-reflection-rounds", type=int, default=2)
    parser.add_argument(
        "--enable-parallel-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable parallel retrieval in DualViewHybridRetriever",
    )
    parser.add_argument("--max-retrieval-workers", type=int, default=3)
    parser.add_argument("--raw-semantic-top-k", type=int, default=None)
    parser.add_argument("--raw-keyword-top-k", type=int, default=None)
    parser.add_argument("--mem-sem-weight", type=float, default=0.65)
    parser.add_argument("--mem-lex-weight", type=float, default=0.35)
    parser.add_argument("--raw-sem-weight", type=float, default=0.45)
    parser.add_argument("--raw-lex-weight", type=float, default=0.55)
    parser.add_argument("--final-mem-weight", type=float, default=0.45)
    parser.add_argument("--final-raw-weight", type=float, default=0.45)
    parser.add_argument("--final-agree-weight", type=float, default=0.10)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    samples = load_locomo_dataset(args.dataset)
    if args.num_samples is not None:
        samples = samples[: args.num_samples]

    llm_client = LLMClient(
        api_key=args.llm_api_key,
        model=args.llm_model,
        base_url=args.llm_base_url,
    )
    embedding_model = EmbeddingModel(
        model_name=args.embedding_model,
        use_optimization=args.embedding_use_optimization,
    )
    answer_generator = AnswerGenerator(llm_client=llm_client)
    judge_client = create_judge_llm_client() if args.llm_judge else None

    all_results = []
    metrics_list = []
    categories = []
    recall_by_category: dict[int, list[float]] = {1: [], 2: [], 4: []}
    total_retrieval = 0.0
    total_answer = 0.0
    total_questions = 0

    base_db_path = Path(args.db_path)
    for sample_idx, sample in enumerate(samples):
        sample_db = base_db_path / f"locomo10_sample_{sample_idx}"
        memory_store = MemoryEntryVectorStore(
            db_path=str(sample_db),
            embedding_model=embedding_model,
            table_name=args.memory_table,
        )
        raw_store = RawContextVectorStore(
            db_path=str(sample_db),
            embedding_model=embedding_model,
            table_name=args.raw_table,
        )
        retriever = DualViewHybridRetriever(
            llm_client=llm_client,
            vector_store=memory_store,
            raw_vector_store=raw_store,
            semantic_top_k=args.semantic_top_k,
            keyword_top_k=args.keyword_top_k,
            structured_top_k=args.structured_top_k,
            enable_planning=args.enable_planning,
            enable_reflection=args.enable_reflection,
            max_reflection_rounds=args.max_reflection_rounds,
            enable_parallel_retrieval=args.enable_parallel_retrieval,
            max_retrieval_workers=args.max_retrieval_workers,
            raw_semantic_top_k=args.raw_semantic_top_k,
            raw_keyword_top_k=args.raw_keyword_top_k,
            mem_sem_weight=args.mem_sem_weight,
            mem_lex_weight=args.mem_lex_weight,
            raw_sem_weight=args.raw_sem_weight,
            raw_lex_weight=args.raw_lex_weight,
            final_mem_weight=args.final_mem_weight,
            final_raw_weight=args.final_raw_weight,
            final_agree_weight=args.final_agree_weight,
            rrf_k=args.rrf_k,
        )

        print(f"\n[DualView QA] sample={sample_idx} db={sample_db}")
        for qa_idx, qa in enumerate(sample.qa):
            question = qa.question
            category = qa.category if qa.category is not None else 0
            reference_answer = "Not mentioned in the conversation" if category == 5 else qa.final_answer

            t0 = time.time()
            contexts = retriever.retrieve(question, enable_reflection=(False if category == 5 else None))
            retrieval_time = time.time() - t0

            answer = None
            answer_time = 0.0
            if args.answer_generation:
                t1 = time.time()
                if category == 5:
                    answer = generate_category5_answer(
                        llm_client=llm_client,
                        answer_generator=answer_generator,
                        question=question,
                        contexts=contexts,
                        adversarial_answer=qa.adversarial_answer or "Unknown answer",
                    )
                else:
                    answer = answer_generator.generate_answer(question, contexts)
                answer_time = time.time() - t1

            metrics = {}
            if args.answer_generation and reference_answer and answer is not None:
                metrics = calculate_metrics(
                    answer,
                    reference_answer,
                    question=question,
                    judge_client=judge_client,
                    use_llm_judge=args.llm_judge,
                )
                if metrics:
                    metrics_list.append(metrics)
                    categories.append(category)

            recall = None
            predicted_dia_ids = []
            gold_dia_ids = []
            if category in RECALL_CATEGORIES:
                recall, predicted_dia_ids, gold_dia_ids = compute_recall_from_contexts(
                    contexts=contexts,
                    qa_evidence=qa.evidence or [],
                    raw_store=raw_store,
                )
                if recall is not None:
                    recall_by_category[category].append(recall)

            all_results.append(
                {
                    "sample_idx": sample_idx,
                    "qa_idx": qa_idx,
                    "question": question,
                    "category": category,
                    "reference": reference_answer,
                    "answer": answer,
                    "num_retrieved": len(contexts),
                    "retrieval_time": retrieval_time,
                    "answer_time": answer_time,
                    "metrics": metrics,
                    "evidence_recall": recall,
                    "predicted_dia_ids": predicted_dia_ids,
                    "gold_dia_ids": gold_dia_ids,
                    "dualview_scores": retriever.last_score_details,
                    "raw_evidence_by_entry_id": retriever.last_raw_evidence_by_entry_id,
                }
            )
            total_retrieval += retrieval_time
            total_answer += answer_time
            total_questions += 1
            print(f"  [Q{qa_idx+1}] retrieved={len(contexts)} rt={retrieval_time:.3f}s at={answer_time:.3f}s")
            if metrics:
                metric_line = (
                    f" f1={metrics.get('f1', 0):.3f}"
                    f" rougeL={metrics.get('rougeL_f', 0):.3f}"
                    f" bert_f1={metrics.get('bert_f1', 0):.3f}"
                    f" sbert={metrics.get('sbert_similarity', 0):.3f}"
                )
                if args.llm_judge:
                    metric_line += f" llm_judge={metrics.get('llm_judge_score', 0):.3f}"
                print(f"         metrics:{metric_line}")
            if recall is not None:
                print(f"         recall@evidence={recall:.3f}")

    aggregated_metrics = aggregate_metrics(metrics_list, categories) if metrics_list else {}
    recall_summary = {}
    all_recall_values: list[float] = []
    for c in sorted(RECALL_CATEGORIES):
        values = recall_by_category.get(c, [])
        recall_summary[f"category_{c}"] = {
            "mean_recall": (sum(values) / len(values)) if values else 0.0,
            "count": len(values),
        }
        all_recall_values.extend(values)
    recall_summary["overall"] = {
        "mean_recall": (sum(all_recall_values) / len(all_recall_values)) if all_recall_values else 0.0,
        "count": len(all_recall_values),
    }

    summary = {
        "num_samples": len(samples),
        "num_questions": total_questions,
        "avg_retrieval_time": (total_retrieval / total_questions) if total_questions else 0.0,
        "avg_answer_time": (total_answer / total_questions) if total_questions else 0.0,
        "llm_judge_enabled": args.llm_judge,
        "answer_generation_enabled": args.answer_generation,
    }
    output = {
        "summary": summary,
        "aggregated_metrics": aggregated_metrics,
        "recall_summary": recall_summary,
        "results": all_results,
    }
    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.result_file}")


if __name__ == "__main__":
    main()
