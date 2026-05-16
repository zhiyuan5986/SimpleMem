#!/usr/bin/env python3
"""Analyze shortest connectivity paths between multi-evidence turns on LoCoMo bipartite graphs.

For each sample index where both files exist:
- locomo10_sample_{idx}_memory_entries.json
- locomo10_sample_{idx}_longllmlingua_filtered.json

This script builds an undirected bipartite graph (entry <-> keyword), then for QA items
with >=2 evidence turns, maps each evidence turn to entry nodes through
`results[*].inserted_raw_context_entries[*].metadata.turn_dia_id` and `links`.
It computes pairwise shortest paths between evidence turns' entry node sets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

SAMPLE_MEM_RE = re.compile(r"locomo10_sample_(\d+)_memory_entries\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze shortest paths among LoCoMo multi-evidence turns.")
    p.add_argument("--input-dir", type=Path, default=Path("."))
    p.add_argument("--dataset-json", type=Path, default=Path("test_ref/locomo10.json"))
    p.add_argument("--output-json", type=Path, default=Path("outputs/locomo_multievidence_paths.json"))
    p.add_argument("--summary-json", type=Path, default=Path("outputs/locomo_multievidence_path_length_stats.json"))
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_keyword_token(token: str) -> str:
    return re.sub(r"[^\w]+", "", token).casefold()


def build_graph(
    entries: list[dict[str, Any]],
    excluded_keywords: set[str] | None = None,
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    adj: dict[str, set[str]] = defaultdict(set)
    excluded_keywords = excluded_keywords or set()
    node_info: dict[str, dict[str, Any]] = {}
    for idx, entry in enumerate(entries):
        entry_id = str(entry.get("entry_id") or f"entry_idx_{idx}")
        e_node = f"entry::{entry_id}"
        node_info[e_node] = {
            "type": "entry",
            "entry_id": entry_id,
            "entry_index": idx,
            "lossless_restatement": str(entry.get("lossless_restatement", "")),
            "keywords": entry.get("keywords", []),
            "entities": entry.get("entities", []),
        }
        for kw in entry.get("keywords", []) if isinstance(entry.get("keywords", []), list) else []:
            kw = str(kw).strip()
            if not kw:
                continue
            if normalize_keyword_token(kw) in excluded_keywords:
                continue
            k_node = f"keyword::{kw}"
            node_info.setdefault(k_node, {"type": "keyword", "keyword": kw})
            adj[e_node].add(k_node)
            adj[k_node].add(e_node)
    return adj, node_info


def shortest_path_between_sets(adj: dict[str, set[str]], starts: set[str], targets: set[str]) -> list[str]:
    if not starts or not targets:
        return []
    inter = starts & targets
    if inter:
        return [next(iter(inter))]

    q: deque[str] = deque()
    prev: dict[str, str | None] = {}
    for s in starts:
        prev[s] = None
        q.append(s)

    target_hit: str | None = None
    while q:
        cur = q.popleft()
        if cur in targets:
            target_hit = cur
            break
        for nxt in adj.get(cur, set()):
            if nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)

    if target_hit is None:
        return []

    path: list[str] = []
    cur: str | None = target_hit
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path



def extract_speaker_names(sample: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    conv = sample.get("conversation", {}) if isinstance(sample, dict) else {}
    if not isinstance(conv, dict):
        return names
    for value in conv.values():
        if not isinstance(value, list):
            continue
        for msg in value:
            if not isinstance(msg, dict):
                continue
            speaker = msg.get("speaker")
            if speaker is None:
                continue
            s = str(speaker).strip()
            if s:
                names.add(normalize_keyword_token(s))
    return names

def evidence_to_turn_ids(evidence_list: list[Any]) -> list[str]:
    out: list[str] = []
    for ev in evidence_list:
        s = str(ev).strip()
        if not s:
            continue
        if ":" in s:
            s = s.split(":", 1)[1].strip()
        out.append(s)
    return out


def main() -> None:
    args = parse_args()
    dataset = load_json(args.dataset_json)
    if not isinstance(dataset, list):
        raise ValueError("dataset-json must be a list")

    sample_mem_files: list[tuple[int, Path]] = []
    for p in sorted(args.input_dir.glob("locomo10_sample_*_memory_entries.json")):
        m = SAMPLE_MEM_RE.search(p.name)
        if m:
            sample_mem_files.append((int(m.group(1)), p))

    details: dict[str, Any] = {"samples": []}
    global_hist: Counter[int] = Counter()
    connected_pairs = 0
    total_pairs = 0

    for sample_idx, mem_file in sample_mem_files:
        filtered_file = mem_file.with_name(f"locomo10_sample_{sample_idx}_longllmlingua_filtered.json")
        if not filtered_file.exists() or sample_idx >= len(dataset):
            continue

        entries = load_json(mem_file)
        filtered = load_json(filtered_file)
        if not isinstance(entries, list):
            continue

        results = filtered.get("results", []) if isinstance(filtered, dict) else []
        if not isinstance(results, list):
            continue

        sample_data = dataset[sample_idx] if isinstance(dataset[sample_idx], dict) else {}
        speaker_names = extract_speaker_names(sample_data)
        adj, node_info = build_graph(entries, excluded_keywords=speaker_names)

        turn_to_entry_nodes: dict[str, set[str]] = defaultdict(set)
        for item in results:
            if not isinstance(item, dict):
                continue
            inserted = item.get("inserted_raw_context_entries", [])
            if not isinstance(inserted, list):
                continue
            for raw_entry in inserted:
                if not isinstance(raw_entry, dict):
                    continue
                md = raw_entry.get("metadata", {})
                if not isinstance(md, dict):
                    continue
                turn_id = md.get("turn_dia_id")
                if turn_id is None:
                    turn_id = md.get("dialogue_id")
                turn_id = str(turn_id).strip() if turn_id is not None else ""
                if not turn_id:
                    continue
                links = raw_entry.get("links", [])
                if not isinstance(links, list):
                    continue
                for link in links:
                    e_node = f"entry::{str(link)}"
                    if e_node in node_info:
                        turn_to_entry_nodes[turn_id].add(e_node)

        sample_qas = sample_data.get("qa", []) if isinstance(sample_data, dict) else []
        sample_out: dict[str, Any] = {
            "sample_idx": sample_idx,
            "memory_file": str(mem_file),
            "filtered_file": str(filtered_file),
            "qa_multievidence_paths": [],
        }

        for qa_idx, qa in enumerate(sample_qas):
            if not isinstance(qa, dict):
                continue
            evidence = qa.get("evidence", [])
            if not isinstance(evidence, list) or len(evidence) <= 1:
                continue
            # turn_ids = evidence_to_turn_ids(evidence)
            turn_ids = evidence
            turn_nodes = {tid: sorted(turn_to_entry_nodes.get(tid, set())) for tid in turn_ids}

            pair_records: list[dict[str, Any]] = []
            for t1, t2 in combinations(turn_ids, 2):
                total_pairs += 1
                pnodes = shortest_path_between_sets(adj, set(turn_nodes.get(t1, [])), set(turn_nodes.get(t2, [])))
                if pnodes:
                    connected_pairs += 1
                    path_len = len(pnodes) - 1
                    global_hist[path_len] += 1
                else:
                    path_len = None

                enriched_path = []
                for nid in pnodes:
                    info = node_info.get(nid, {"type": "unknown"})
                    item = {"node_id": nid, "node_type": info.get("type")}
                    if info.get("type") == "entry":
                        item.update(
                            {
                                "entry_id": info.get("entry_id"),
                                "entry_index": info.get("entry_index"),
                                "lossless_restatement": info.get("lossless_restatement"),
                                "entities": info.get("entities", []),
                            }
                        )
                    elif info.get("type") == "keyword":
                        item["keyword"] = info.get("keyword")
                    enriched_path.append(item)

                pair_records.append(
                    {
                        "turn_a": t1,
                        "turn_b": t2,
                        "turn_a_entry_nodes": turn_nodes.get(t1, []),
                        "turn_b_entry_nodes": turn_nodes.get(t2, []),
                        "is_connected": bool(pnodes),
                        "shortest_path_edge_length": path_len,
                        "shortest_path_nodes": enriched_path,
                    }
                )

            sample_out["qa_multievidence_paths"].append(
                {
                    "qa_index": qa_idx,
                    "question": qa.get("question"),
                    "evidence": evidence,
                    "evidence_turn_ids": turn_ids,
                    "turn_to_entry_nodes": turn_nodes,
                    "pairwise_paths": pair_records,
                }
            )

        details["samples"].append(sample_out)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    details["summary"] = {
        "total_turn_pairs": total_pairs,
        "connected_turn_pairs": connected_pairs,
        "disconnected_turn_pairs": total_pairs - connected_pairs,
        "connected_ratio": (connected_pairs / total_pairs) if total_pairs else 0.0,
        "shortest_path_length_distribution": {str(k): v for k, v in sorted(global_hist.items())},
    }

    args.output_json.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    args.summary_json.write_text(json.dumps(details["summary"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Saved] {args.output_json}")
    print(f"[Saved] {args.summary_json}")


if __name__ == "__main__":
    main()
