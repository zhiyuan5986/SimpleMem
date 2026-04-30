from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.answer_generator import AnswerGenerator
from database.vector_store import MemoryEntryVectorStore, RawContextVectorStore
from test_locomo10_dualview_qa import RECALL_CATEGORIES, _collect_dia_ids_from_obj
from utils.embedding import EmbeddingModel

try:
    import tiktoken

    _encoding = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(text: str) -> int:
        return len(_encoding.encode(text))
except Exception:

    def count_tokens(text: str) -> int:
        return len(text) // 4


def _parse_categories(category_arg: str | None) -> set[int] | None:
    if not category_arg:
        return None
    if category_arg.lower() == "all":
        return None
    cats: set[int] = set()
    for part in category_arg.split(","):
        part = part.strip()
        if not part:
            continue
        cats.add(int(part))
    return cats if cats else None


def compute_recall(entry_ids: list[str], gold_ids: list[str], raw_store: RawContextVectorStore) -> tuple[float | None, list[str]]:
    gold = {str(x).strip() for x in (gold_ids or []) if str(x).strip()}
    if not gold:
        return None, []
    pred = set()
    for eid in entry_ids:
        raw_entry = raw_store.get_entry_by_id(eid)
        if raw_entry is None:
            continue
        pred.update(_collect_dia_ids_from_obj(raw_entry.metadata))
    recall = len(pred.intersection(gold)) / len(gold) if gold else None
    return recall, sorted(pred)



def select_topk_by_semantic_similarity(
    query: str,
    candidate_ids: list[str],
    mem_store: MemoryEntryVectorStore,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> list[str]:
    if len(candidate_ids) <= top_k:
        return candidate_ids
    entries_map = mem_store.get_entries_by_ids(candidate_ids)
    filtered_ids = [eid for eid in candidate_ids if eid in entries_map]
    if len(filtered_ids) <= top_k:
        return filtered_ids

    query_vec = embedding_model.encode_single(query, is_query=True)
    texts = [entries_map[eid].lossless_restatement or "" for eid in filtered_ids]
    doc_vecs = embedding_model.encode_documents(texts)

    scored = []
    for i, eid in enumerate(filtered_ids):
        sim = float((query_vec * doc_vecs[i]).sum())
        scored.append((eid, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [eid for eid, _ in scored[:top_k]]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", default="locomo10_token_cost_summary.json")
    p.add_argument("--db-path", default="./lancedb_data")
    p.add_argument("--memory-table", default="memory_entries")
    p.add_argument("--raw-table", default="llm_spans")
    p.add_argument("--embedding-model", type=str, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--categories", type=str, default="1,2,4", help="Comma-separated categories or 'all'")
    args = p.parse_args()

    selected_categories = _parse_categories(args.categories)

    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    results = data.get("results", [])
    embedding_model = EmbeddingModel(model_name=args.embedding_model)
    answer_generator = AnswerGenerator(llm_client=None)

    by_sample: dict[int, tuple[MemoryEntryVectorStore, RawContextVectorStore]] = {}

    token_costs = []
    recalls = []
    output_rows = []

    for r in results:
        category = int(r.get("category", 0))
        if selected_categories is not None and category not in selected_categories:
            continue

        sample_idx = int(r.get("sample_idx", 0))
        if sample_idx not in by_sample:
            sample_db = Path(args.db_path) / f"locomo10_sample_{sample_idx}"
            mem = MemoryEntryVectorStore(str(sample_db), embedding_model=embedding_model, table_name=args.memory_table)
            raw = RawContextVectorStore(str(sample_db), embedding_model=embedding_model, table_name=args.raw_table)
            by_sample[sample_idx] = (mem, raw)

        mem_store, raw_store = by_sample[sample_idx]
        question = r.get("question", "")
        candidate_ids = r.get("retrieved_entry_ids") or list((r.get("dualview_scores") or {}).keys())
        candidate_ids = [str(eid).strip() for eid in candidate_ids if str(eid).strip()]

        reranked_ids = candidate_ids
        if args.top_k is not None and len(reranked_ids) > args.top_k:
            reranked_ids = select_topk_by_semantic_similarity(
                question,
                reranked_ids,
                mem_store,
                embedding_model,
                args.top_k,
            )

        entries_map = mem_store.get_entries_by_ids(reranked_ids)
        selected_contexts = [entries_map[eid] for eid in reranked_ids if eid in entries_map]
        context_text = answer_generator._format_contexts(selected_contexts)
        token_cost = count_tokens(context_text)

        recall = None
        predicted_ids = r.get("predicted_dia_ids") or []
        if category in RECALL_CATEGORIES:
            gold = r.get("gold_dia_ids") or []
            recall, predicted_ids = compute_recall(reranked_ids, gold, raw_store)
            if recall is not None:
                recalls.append(recall)

        token_costs.append(token_cost)
        output_rows.append({
            "sample_idx": sample_idx,
            "qa_idx": r.get("qa_idx"),
            "category": category,
            "question": question,
            "selected_entry_ids": reranked_ids,
            "token_cost": token_cost,
            "evidence_recall": recall,
            "predicted_dia_ids": predicted_ids,
            "gold_dia_ids": r.get("gold_dia_ids") or [],
        })

    summary = {
        "num_questions": len(output_rows),
        "num_recall_questions": len(recalls),
        "avg_token_cost_all": (sum(token_costs) / len(token_costs)) if token_costs else 0.0,
        "avg_token_cost_recall_questions": (
            sum(row["token_cost"] for row in output_rows if row["evidence_recall"] is not None)
            / len([row for row in output_rows if row["evidence_recall"] is not None])
        )
        if any(row["evidence_recall"] is not None for row in output_rows)
        else 0.0,
        "avg_recall": (sum(recalls) / len(recalls)) if recalls else 0.0,
        "top_k": args.top_k,
        "categories": args.categories,
    }

    Path(args.output_json).write_text(json.dumps({"summary": summary, "results": output_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
