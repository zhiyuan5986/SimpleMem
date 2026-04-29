"""
LoCoMo10 recall evaluation using SimpleMem default single-view retrieval path.

This script follows the core flow in test_locomo10.py:
1) build per-sample memory with SimpleMem
2) run default HybridRetriever on each QA question
3) map retrieved memory entries back to dialogue ids via llm_spans table
4) compute evidence recall for LoCoMo categories {1,2,4}
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Any

from database.vector_store import RawContextVectorStore
from main import SimpleMemSystem
from models.memory_entry import Dialogue
from test_locomo10 import load_locomo_dataset

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


def compute_recall_from_contexts(contexts: list, qa_evidence: list[str], raw_store: RawContextVectorStore) -> tuple[float | None, list[str], list[str]]:
    gold_ids = {str(e).strip() for e in (qa_evidence or []) if str(e).strip()}
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


def convert_to_dialogues(sample) -> list[Dialogue]:
    dialogues: list[Dialogue] = []
    dialogue_id = 1
    for session_id in sorted(sample.conversation.sessions.keys()):
        session = sample.conversation.sessions[session_id]
        for turn in session.turns:
            dialogues.append(
                Dialogue(
                    dialogue_id=dialogue_id,
                    speaker=turn.speaker,
                    content=turn.text,
                    timestamp=session.date_time,
                    metadata={
                        "session_id": session_id,
                        "session_date_time": session.date_time,
                        "turn_metadata": turn.metadata,
                    },
                )
            )
            dialogue_id += 1
    return dialogues


def process_single_qa(
    system: SimpleMemSystem,
    raw_store: RawContextVectorStore,
    sample_idx: int,
    qa_idx: int,
    qa,
) -> tuple[dict[str, Any], float | None]:
    category = qa.category if qa.category is not None else 0
    if category not in RECALL_CATEGORIES:
        return {}, None

    retrieve_start = time.time()
    contexts = system.hybrid_retriever.retrieve(qa.question)
    retrieve_time = time.time() - retrieve_start

    recall, predicted_ids, gold_ids = compute_recall_from_contexts(contexts, qa.evidence, raw_store)
    result = {
        "sample_idx": sample_idx,
        "qa_idx": qa_idx,
        "category": category,
        "question": qa.question,
        "recall": recall,
        "num_retrieved_entries": len(contexts),
        "retrieval_time": retrieve_time,
        "predicted_dia_ids": predicted_ids,
        "gold_dia_ids": gold_ids,
    }
    return result, recall


def main():
    parser = argparse.ArgumentParser(description="LoCoMo10 single-view recall eval with llm_spans back-mapping")
    parser.add_argument("--dataset", type=str, default="test_ref/locomo10.json")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--db-path", type=str, default="./lancedb_data")
    parser.add_argument("--memory-table", type=str, default="memory_entries")
    parser.add_argument("--raw-table", type=str, default="llm_spans")
    parser.add_argument("--result-file", type=str, default="locomo10_singleview_results.json")
    parser.add_argument("--semantic-top-k", type=int, default=5)
    parser.add_argument("--keyword-top-k", type=int, default=3)
    parser.add_argument("--structured-top-k", type=int, default=3)
    parser.add_argument("--parallel-questions", action="store_true", help="Enable parallel QA retrieval within each sample")
    parser.add_argument("--test-workers", type=int, default=None, help="Number of parallel workers for QA retrieval")
    args = parser.parse_args()

    samples = load_locomo_dataset(args.dataset)
    if args.num_samples is not None:
        samples = samples[: args.num_samples]

    results: list[dict[str, Any]] = []
    recall_values: list[float] = []

    base_db_path = Path(args.db_path)

    for sample_idx, sample in enumerate(samples):
        sample_db_path = base_db_path / f"locomo10_sample_{sample_idx}"
        system = SimpleMemSystem(
            db_path=str(sample_db_path),
            table_name=args.memory_table,
            clear_db=False,
        )
        # Force default retrieval path params from script args.
        system.hybrid_retriever.semantic_top_k = args.semantic_top_k
        system.hybrid_retriever.keyword_top_k = args.keyword_top_k
        system.hybrid_retriever.structured_top_k = args.structured_top_k

        raw_store = RawContextVectorStore(
            db_path=str(sample_db_path),
            embedding_model=system.embedding_model,
            table_name=args.raw_table,
        )

        dialogues = convert_to_dialogues(sample)
        build_start = time.time()
        system.add_dialogues(dialogues)
        system.finalize()
        build_time = time.time() - build_start

        qa_results: list[dict[str, Any]] = []

        if args.parallel_questions and len(sample.qa) > 1:
            max_workers = args.test_workers if args.test_workers is not None else 16
            max_workers = min(max_workers, len(sample.qa), 20)
            max_workers = max(max_workers, 1)
            print(f"[Parallel QA] sample={sample_idx}, workers={max_workers}, questions={len(sample.qa)}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_single_qa, system, raw_store, sample_idx, qa_idx, qa): qa_idx
                    for qa_idx, qa in enumerate(sample.qa)
                }
                ordered: dict[int, tuple[dict[str, Any], float | None]] = {}
                for future in concurrent.futures.as_completed(futures):
                    qa_idx = futures[future]
                    ordered[qa_idx] = future.result()
                for qa_idx in sorted(ordered.keys()):
                    result, recall = ordered[qa_idx]
                    if not result:
                        continue
                    result["build_time"] = build_time
                    qa_results.append(result)
                    if recall is not None:
                        recall_values.append(recall)
        else:
            for qa_idx, qa in enumerate(sample.qa):
                result, recall = process_single_qa(system, raw_store, sample_idx, qa_idx, qa)
                if not result:
                    continue
                result["build_time"] = build_time
                qa_results.append(result)
                if recall is not None:
                    recall_values.append(recall)

        results.extend(qa_results)

    avg_recall = (sum(recall_values) / len(recall_values)) if recall_values else None
    summary = {
        "num_samples": len(samples),
        "num_questions_scored": len(recall_values),
        "average_recall": avg_recall,
    }

    output = {"summary": summary, "results": results}
    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("LoCoMo10 Single-View Recall Summary")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to: {args.result_file}")


if __name__ == "__main__":
    main()
