#!/usr/bin/env python3
"""Build and visualize bipartite graphs for locomo10 memory-entry JSON files.

Usage:
    python scripts/plot_locomo_bipartite.py --input-dir . --output-dir outputs/bipartite
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from itertools import combinations
import json
from pathlib import Path

import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bipartite keyword-entry graphs for locomo10 files.")
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing locomo10_sample_*_memory_entries.json files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bipartite"), help="Directory to save HTML graphs.")
    return parser.parse_args()


def find_input_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("locomo10_sample_*_memory_entries.json"))


def load_entries(file_path: Path) -> list[dict]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{file_path} is not a list of dict entries.")
    return data


def build_graph_data(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    links: list[dict] = []

    keyword_to_node_id: dict[str, str] = {}

    for idx, entry in enumerate(entries):
        entry_node_id = f"entry::{idx}"
        entry_text = str(entry.get("lossless_restatement", "")).strip()
        short_text = entry_text if len(entry_text) <= 60 else f"{entry_text[:57]}..."
        entry_label = f"E{idx}: {short_text}"

        nodes.append(
            {
                "name": entry_node_id,
                "symbolSize": 28,
                "category": 1,
                "value": entry_text,
                "label": {"show": True, "formatter": entry_label},
            }
        )

        keywords = entry.get("keywords", [])
        if not isinstance(keywords, list):
            continue

        for kw in keywords:
            keyword = str(kw).strip()
            if not keyword:
                continue

            if keyword not in keyword_to_node_id:
                kw_node_id = f"keyword::{keyword}"
                keyword_to_node_id[keyword] = kw_node_id
                nodes.append(
                    {
                        "name": kw_node_id,
                        "symbolSize": 16,
                        "category": 0,
                        "value": keyword,
                        "label": {"show": True, "formatter": keyword},
                    }
                )

            links.append({"source": entry_node_id, "target": keyword_to_node_id[keyword]})

    return nodes, links


def collect_keyword_and_pair_stats(entries: list[dict]) -> tuple[dict[str, list[str]], dict[str, list[str]], Counter, Counter]:
    """Collect keyword-degree and keyword-pair shared-entry statistics.

    Returns:
        keyword_to_entries_texts: keyword -> list of entry text.
        pair_to_shared_entries_texts: "kw1 || kw2" -> list of shared entry text.
        keyword_degree_hist: degree -> number of keywords.
        pair_shared_hist: shared-entry count -> number of keyword pairs.
    """
    keyword_to_entry_idxs: dict[str, set[int]] = defaultdict(set)
    entry_texts: list[str] = []

    for idx, entry in enumerate(entries):
        entry_text = str(entry.get("lossless_restatement", "")).strip()
        entry_texts.append(entry_text)

        keywords = entry.get("keywords", [])
        if not isinstance(keywords, list):
            continue

        dedup_keywords = {str(kw).strip() for kw in keywords if str(kw).strip()}
        for kw in dedup_keywords:
            keyword_to_entry_idxs[kw].add(idx)

    keyword_to_entries_texts: dict[str, list[str]] = {}
    keyword_degree_hist: Counter = Counter()
    for kw, idxs in sorted(keyword_to_entry_idxs.items()):
        sorted_idxs = sorted(idxs)
        keyword_to_entries_texts[kw] = [entry_texts[i] for i in sorted_idxs]
        keyword_degree_hist[len(sorted_idxs)] += 1

    pair_to_shared_entries_texts: dict[str, list[str]] = {}
    pair_shared_hist: Counter = Counter()
    keywords_sorted = sorted(keyword_to_entry_idxs.keys())
    for kw1, kw2 in combinations(keywords_sorted, 2):
        shared_idxs = sorted(keyword_to_entry_idxs[kw1] & keyword_to_entry_idxs[kw2])
        if not shared_idxs:
            continue
        pair_key = f"{kw1} || {kw2}"
        pair_to_shared_entries_texts[pair_key] = [entry_texts[i] for i in shared_idxs]
        pair_shared_hist[len(shared_idxs)] += 1

    return keyword_to_entries_texts, pair_to_shared_entries_texts, keyword_degree_hist, pair_shared_hist


def collect_entity_shortest_paths_stats(entries: list[dict]) -> tuple[dict[str, dict], Counter, int]:
    """Collect all shortest paths between each entity (keyword) pair in the bipartite graph.

    Returns:
        pair_to_paths: "kw1 || kw2" -> {"shortest_length": int, "paths": ["kw1 -> E1 -> kw2", ...]}
        shortest_length_hist: shortest path length -> number of connected keyword pairs
        disconnected_pair_count: number of keyword pairs that are not connected
    """
    keyword_to_entry_idxs: dict[str, set[int]] = defaultdict(set)
    entry_texts: list[str] = []
    for idx, entry in enumerate(entries):
        entry_texts.append(str(entry.get("lossless_restatement", "")).strip())
        keywords = entry.get("keywords", [])
        if not isinstance(keywords, list):
            continue
        dedup_keywords = {str(kw).strip() for kw in keywords if str(kw).strip()}
        for kw in dedup_keywords:
            keyword_to_entry_idxs[kw].add(idx)

    entry_to_keywords: dict[int, set[str]] = defaultdict(set)
    for kw, idxs in keyword_to_entry_idxs.items():
        for idx in idxs:
            entry_to_keywords[idx].add(kw)

    def keyword_neighbors(kw: str) -> list[str]:
        return [f"entry::{idx}" for idx in sorted(keyword_to_entry_idxs.get(kw, set()))]

    def entry_neighbors(entry_node: str) -> list[str]:
        entry_idx = int(entry_node.split("::", 1)[1])
        return [f"keyword::{kw}" for kw in sorted(entry_to_keywords.get(entry_idx, set()))]

    def get_neighbors(node: str) -> list[str]:
        if node.startswith("keyword::"):
            return keyword_neighbors(node.split("::", 1)[1])
        return entry_neighbors(node)

    def format_path(path_nodes: list[str]) -> str:
        formatted_parts: list[str] = []
        for node in path_nodes:
            if node.startswith("keyword::"):
                formatted_parts.append(node.split("::", 1)[1])
            else:
                entry_idx = int(node.split("::", 1)[1])
                entry_text = entry_texts[entry_idx]
                short_text = entry_text if len(entry_text) <= 30 else f"{entry_text[:27]}..."
                formatted_parts.append(f"E{entry_idx}({short_text})")
        return " -> ".join(formatted_parts)

    def all_shortest_paths_between_keywords(src_kw: str, dst_kw: str) -> list[list[str]]:
        src = f"keyword::{src_kw}"
        dst = f"keyword::{dst_kw}"
        dist: dict[str, int] = {src: 0}
        parents: dict[str, list[str]] = defaultdict(list)
        q = deque([src])
        min_dst_dist: int | None = None

        while q:
            node = q.popleft()
            d = dist[node]
            if min_dst_dist is not None and d >= min_dst_dist:
                continue

            for nb in get_neighbors(node):
                nd = d + 1
                if nb not in dist:
                    dist[nb] = nd
                    parents[nb].append(node)
                    if nb == dst:
                        min_dst_dist = nd
                    q.append(nb)
                elif dist[nb] == nd:
                    parents[nb].append(node)

        if dst not in dist:
            return []

        all_paths: list[list[str]] = []

        def backtrack(cur: str, acc: list[str]) -> None:
            if cur == src:
                all_paths.append([src] + list(reversed(acc)))
                return
            for p in parents[cur]:
                backtrack(p, acc + [cur])

        backtrack(dst, [])
        return all_paths

    keywords_sorted = sorted(keyword_to_entry_idxs.keys())
    pair_to_paths: dict[str, dict] = {}
    shortest_length_hist: Counter = Counter()
    disconnected_pair_count = 0

    for kw1, kw2 in combinations(keywords_sorted, 2):
        paths = all_shortest_paths_between_keywords(kw1, kw2)
        if not paths:
            disconnected_pair_count += 1
            continue
        shortest_len = len(paths[0]) - 1
        pair_key = f"{kw1} || {kw2}"
        pair_to_paths[pair_key] = {
            "shortest_length": shortest_len,
            "paths": [format_path(p) for p in paths],
        }
        shortest_length_hist[shortest_len] += 1

    return pair_to_paths, shortest_length_hist, disconnected_pair_count


def render_histogram(hist: Counter, title: str, x_name: str, y_name: str, output_path: Path) -> None:
    x_vals = sorted(hist.keys())
    y_vals = [hist[x] for x in x_vals]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_vals, y_vals, width=0.8, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_xticks(x_vals)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def save_json(data: dict[str, list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def render_bipartite_graph(nodes: list[dict], links: list[dict], title: str, output_path: Path) -> None:
    categories = [{"name": "Keyword"}, {"name": "Entry(lossless_restatement)"}]

    graph = (
        Graph(init_opts=opts.InitOpts(width="1600px", height="1000px"))
        .add(
            series_name="",
            nodes=nodes,
            links=links,
            categories=categories,
            layout="force",
            repulsion=900,
            edge_length=[90, 220],
            is_draggable=True,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            legend_opts=opts.LegendOpts(is_show=True),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.render(str(output_path))


def main() -> None:
    args = parse_args()
    files = find_input_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_dir}/locomo10_sample_*_memory_entries.json")

    for file_path in files:
        entries = load_entries(file_path)
        nodes, links = build_graph_data(entries)
        output_path = args.output_dir / f"{file_path.stem}_bipartite.html"
        render_bipartite_graph(nodes, links, title=f"Bipartite Graph: {file_path.name}", output_path=output_path)
        print(f"[Saved] {output_path}")

        keyword_to_entries, pair_to_entries, keyword_degree_hist, pair_shared_hist = collect_keyword_and_pair_stats(entries)
        pair_to_shortest_paths, shortest_path_len_hist, disconnected_pair_count = collect_entity_shortest_paths_stats(entries)

        keyword_json_path = args.output_dir / f"{file_path.stem}_keyword_to_entries.json"
        pair_json_path = args.output_dir / f"{file_path.stem}_entity_pair_shared_entries.json"
        shortest_paths_json_path = args.output_dir / f"{file_path.stem}_entity_pair_shortest_paths.json"
        shortest_paths_disconnected_json_path = args.output_dir / f"{file_path.stem}_entity_pair_shortest_paths_disconnected_stats.json"
        save_json(keyword_to_entries, keyword_json_path)
        save_json(pair_to_entries, pair_json_path)
        save_json(pair_to_shortest_paths, shortest_paths_json_path)
        save_json({"disconnected_pair_count": disconnected_pair_count}, shortest_paths_disconnected_json_path)
        print(f"[Saved] {keyword_json_path}")
        print(f"[Saved] {pair_json_path}")
        print(f"[Saved] {shortest_paths_json_path}")
        print(f"[Saved] {shortest_paths_disconnected_json_path}")

        keyword_hist_path = args.output_dir / f"{file_path.stem}_keyword_degree_histogram.pdf"
        pair_hist_path = args.output_dir / f"{file_path.stem}_entity_pair_shared_entries_histogram.pdf"
        shortest_path_hist_path = args.output_dir / f"{file_path.stem}_entity_pair_shortest_path_length_histogram.pdf"

        render_histogram(
            keyword_degree_hist,
            title=f"Keyword Degree Distribution: {file_path.name}",
            x_name="Degree (linked entries count)",
            y_name="Keyword Count",
            output_path=keyword_hist_path,
        )
        render_histogram(
            pair_shared_hist,
            title=f"Shared Entries Distribution Between Entity Pairs: {file_path.name}",
            x_name="Shared entries count per entity pair",
            y_name="Entity Pair Count",
            output_path=pair_hist_path,
        )
        render_histogram(
            shortest_path_len_hist,
            title=f"Shortest Path Length Distribution Between Entity Pairs: {file_path.name}",
            x_name="Shortest path length",
            y_name="Connected entity pair count",
            output_path=shortest_path_hist_path,
        )
        print(f"[Saved] {keyword_hist_path}")
        print(f"[Saved] {pair_hist_path}")
        print(f"[Saved] {shortest_path_hist_path}")


if __name__ == "__main__":
    main()
