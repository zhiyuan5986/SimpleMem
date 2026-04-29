"""
Dual-view hybrid retriever over memory entries + raw evidence.
"""
from __future__ import annotations

import concurrent.futures
from math import sqrt
import re
from typing import Any, Dict, List, Optional

from core.hybrid_retriever import HybridRetriever
from database.vector_store import MemoryEntryVectorStore, RawContextVectorStore
from models.memory_entry import MemoryEntry
from models.raw_context import RawContextEntry
from utils.llm_client import LLMClient


class DualViewHybridRetriever(HybridRetriever):
    """
    Hybrid retriever variant that fuses two aligned stores by entry_id:
    - memory-entry view
    - raw-evidence view
    """

    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: MemoryEntryVectorStore,
        raw_vector_store: RawContextVectorStore,
        semantic_top_k: int = None,
        keyword_top_k: int = None,
        structured_top_k: int = None,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        max_reflection_rounds: int = 2,
        enable_parallel_retrieval: bool = True,
        max_retrieval_workers: int = 3,
        raw_semantic_top_k: int = None,
        raw_keyword_top_k: int = None,
        mem_sem_weight: float = 0.65,
        mem_lex_weight: float = 0.35,
        raw_sem_weight: float = 0.45,
        raw_lex_weight: float = 0.55,
        final_mem_weight: float = 0.45,
        final_raw_weight: float = 0.45,
        final_agree_weight: float = 0.10,
        rrf_k: int = 60,
        keyword_extraction_mode: str = "llm",
    ):
        super().__init__(
            llm_client=llm_client,
            vector_store=vector_store,
            semantic_top_k=semantic_top_k,
            keyword_top_k=keyword_top_k,
            structured_top_k=structured_top_k,
            enable_planning=enable_planning,
            enable_reflection=enable_reflection,
            max_reflection_rounds=max_reflection_rounds,
            enable_parallel_retrieval=enable_parallel_retrieval,
            max_retrieval_workers=max_retrieval_workers,
        )
        self.raw_vector_store = raw_vector_store
        self.raw_semantic_top_k = raw_semantic_top_k or self.semantic_top_k
        self.raw_keyword_top_k = raw_keyword_top_k or self.keyword_top_k

        self.mem_sem_weight = mem_sem_weight
        self.mem_lex_weight = mem_lex_weight
        self.raw_sem_weight = raw_sem_weight
        self.raw_lex_weight = raw_lex_weight
        self.final_mem_weight = final_mem_weight
        self.final_raw_weight = final_raw_weight
        self.final_agree_weight = final_agree_weight
        self.rrf_k = rrf_k
        if keyword_extraction_mode not in {"llm", "lightweight"}:
            raise ValueError("keyword_extraction_mode must be one of: llm, lightweight")
        self.keyword_extraction_mode = keyword_extraction_mode

        self.last_score_details: Dict[str, Dict[str, Any]] = {}
        self.last_raw_evidence_by_entry_id: Dict[str, str] = {}

    def retrieve(self, query: str, enable_reflection: Optional[bool] = None) -> List[MemoryEntry]:
        if self.enable_planning:
            return self._retrieve_with_planning(query, enable_reflection)
        return self._dual_view_search(query, top_n=self.semantic_top_k)

    def _retrieve_with_planning(self, query: str, enable_reflection: Optional[bool] = None) -> List[MemoryEntry]:
        print(f"\n[DualView Planning] Analyzing information requirements for: {query}")
        information_plan = self._analyze_information_requirements(query)
        search_queries = self._generate_targeted_queries(query, information_plan)
        print(f"[DualView Planning] Generated {len(search_queries)} targeted queries")

        aggregated: List[MemoryEntry] = []
        for i, search_query in enumerate(search_queries, 1):
            print(f"[DualView Search {i}] {search_query}")
            aggregated.extend(self._dual_view_search(search_query, top_n=self.semantic_top_k))

        merged_results = self._merge_and_deduplicate_entries(aggregated)
        print(f"[DualView Planning] Found {len(merged_results)} unique results")

        should_use_reflection = enable_reflection if enable_reflection is not None else self.enable_reflection
        if should_use_reflection:
            merged_results = self._retrieve_with_intelligent_reflection(query, merged_results, information_plan)
        return merged_results

    def _semantic_search(self, query: str) -> List[MemoryEntry]:
        """
        Reflection and planning internals call this method for additional retrieval.
        Here we route it to dual-view retrieval.
        """
        return self._dual_view_search(query, top_n=self.semantic_top_k)

    def _structured_search(self, query_analysis: Dict[str, Any]) -> List[MemoryEntry]:
        """
        Structured/symbolic retrieval is disabled in this dual-view retriever.
        """
        return []

    def _dual_view_search(self, query: str, top_n: int) -> List[MemoryEntry]:
        keywords = self._extract_query_keywords(query)
        if not keywords:
            keywords = [query]

        def do_mem_sem() -> List[MemoryEntry]:
            return self.vector_store.semantic_search(query, top_k=self.semantic_top_k) or []

        def do_mem_lex() -> List[MemoryEntry]:
            return self.vector_store.keyword_search(keywords, top_k=self.keyword_top_k) or []

        def do_raw_sem() -> List[RawContextEntry]:
            return self.raw_vector_store.semantic_search(query, top_k=self.raw_semantic_top_k) or []

        def do_raw_lex() -> List[RawContextEntry]:
            return self.raw_vector_store.keyword_search(keywords, top_k=self.raw_keyword_top_k) or []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_mem_sem = executor.submit(do_mem_sem)
            future_mem_lex = executor.submit(do_mem_lex)
            future_raw_sem = executor.submit(do_raw_sem)
            future_raw_lex = executor.submit(do_raw_lex)

            mem_sem = future_mem_sem.result()
            mem_lex = future_mem_lex.result()
            raw_sem = future_raw_sem.result()
            raw_lex = future_raw_lex.result()

        mem_sem_rrf = self._rrf_by_entry_id(mem_sem)
        mem_lex_rrf = self._rrf_by_entry_id(mem_lex)
        raw_sem_rrf = self._rrf_by_entry_id(raw_sem)
        raw_lex_rrf = self._rrf_by_entry_id(raw_lex)
        raw_text_map = self._best_raw_text_by_entry(raw_sem, raw_lex)

        all_entry_ids = set(mem_sem_rrf) | set(mem_lex_rrf) | set(raw_sem_rrf) | set(raw_lex_rrf)
        scored: Dict[str, Dict[str, Any]] = {}
        for entry_id in all_entry_ids:
            mem_sem_score = mem_sem_rrf.get(entry_id, 0.0)
            mem_lex_score = mem_lex_rrf.get(entry_id, 0.0)
            raw_sem_score = raw_sem_rrf.get(entry_id, 0.0)
            raw_lex_score = raw_lex_rrf.get(entry_id, 0.0)

            mem_score = self.mem_sem_weight * mem_sem_score + self.mem_lex_weight * mem_lex_score
            raw_score = self.raw_sem_weight * raw_sem_score + self.raw_lex_weight * raw_lex_score
            agree_score = sqrt(max(mem_score * raw_score, 0.0))
            final_score = (
                self.final_mem_weight * mem_score
                + self.final_raw_weight * raw_score
                + self.final_agree_weight * agree_score
            )

            sources = []
            if entry_id in mem_sem_rrf:
                sources.append("mem_sem")
            if entry_id in mem_lex_rrf:
                sources.append("mem_lex")
            if entry_id in raw_sem_rrf:
                sources.append("raw_sem")
            if entry_id in raw_lex_rrf:
                sources.append("raw_lex")

            scored[entry_id] = {
                "mem_sem": mem_sem_score,
                "mem_lex": mem_lex_score,
                "raw_sem": raw_sem_score,
                "raw_lex": raw_lex_score,
                "mem_score": mem_score,
                "raw_score": raw_score,
                "agree_score": agree_score,
                "final_score": final_score,
                "sources": sources,
            }

        ranked_entry_ids = sorted(scored.keys(), key=lambda eid: scored[eid]["final_score"], reverse=True)[:top_n]
        memory_entries = self._materialize_memory_entries(ranked_entry_ids, raw_text_map)

        debug_details: Dict[str, Dict[str, Any]] = {}
        raw_map_top: Dict[str, str] = {}
        for entry in memory_entries:
            detail = dict(scored.get(entry.entry_id, {}))
            detail["entry_text"] = entry.lossless_restatement
            if entry.entry_id in raw_text_map:
                detail["raw_text"] = raw_text_map[entry.entry_id]
                raw_map_top[entry.entry_id] = raw_text_map[entry.entry_id]
            debug_details[entry.entry_id] = detail

        self.last_score_details = debug_details
        self.last_raw_evidence_by_entry_id = raw_map_top

        return memory_entries

    def _extract_query_keywords(self, query: str) -> List[str]:
        if self.keyword_extraction_mode == "llm":
            query_analysis = self._analyze_query(query)
            return query_analysis.get("keywords", []) or []

        query_lower = query.lower()
        skip_words = {
            "what", "when", "where", "who", "why", "how", "does", "did", "have", "has",
            "the", "about", "from", "with", "like", "think", "are", "is", "was", "were",
        }
        keywords = [
            w.strip("?.,!'\"")
            for w in query_lower.split()
            if len(w) > 2 and w.lower() not in skip_words
        ]

        possessive_match = re.search(r"([A-Za-z]+)'s\s+(\w+)", query)
        if possessive_match:
            attr_word = possessive_match.group(2).lower()
            keywords.extend([attr_word, f"{attr_word}s", f"{attr_word}ed"])

        deduped: List[str] = []
        seen = set()
        for kw in keywords:
            if kw and kw not in seen:
                seen.add(kw)
                deduped.append(kw)
        return deduped[:5]

    def _rrf_by_entry_id(self, results: List[Any]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for rank, item in enumerate(results, start=1):
            entry_id = getattr(item, "entry_id", None)
            if not entry_id:
                continue
            scores[entry_id] = max(scores.get(entry_id, 0.0), 1.0 / (self.rrf_k + rank))
        return scores

    def _best_raw_text_by_entry(
        self,
        raw_sem: List[RawContextEntry],
        raw_lex: List[RawContextEntry],
    ) -> Dict[str, str]:
        raw_map: Dict[str, str] = {}
        for item in raw_sem + raw_lex:
            if item.entry_id not in raw_map and item.text:
                raw_map[item.entry_id] = item.text
        return raw_map

    def _materialize_memory_entries(self, entry_ids: List[str], raw_text_map: Dict[str, str]) -> List[MemoryEntry]:
        by_id = self.vector_store.get_entries_by_ids(entry_ids)
        materialized: List[MemoryEntry] = []
        for entry_id in entry_ids:
            entry = by_id.get(entry_id)
            if entry is not None:
                materialized.append(entry)
                continue

            fallback = MemoryEntry(
                entry_id=entry_id,
                lossless_restatement=raw_text_map.get(entry_id, ""),
                keywords=[],
                timestamp=None,
                location=None,
                persons=[],
                entities=[],
                topic=None,
            )
            materialized.append(fallback)
        return materialized
