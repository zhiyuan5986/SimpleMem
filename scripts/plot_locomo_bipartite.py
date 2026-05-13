#!/usr/bin/env python3
"""Build and visualize bipartite graphs for locomo10 memory-entry JSON files.

Usage:
    python scripts/plot_locomo_bipartite.py --input-dir . --output-dir outputs/bipartite
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

        keyword_json_path = args.output_dir / f"{file_path.stem}_keyword_to_entries.json"
        pair_json_path = args.output_dir / f"{file_path.stem}_entity_pair_shared_entries.json"
        save_json(keyword_to_entries, keyword_json_path)
        save_json(pair_to_entries, pair_json_path)
        print(f"[Saved] {keyword_json_path}")
        print(f"[Saved] {pair_json_path}")

        keyword_hist_path = args.output_dir / f"{file_path.stem}_keyword_degree_histogram.pdf"
        pair_hist_path = args.output_dir / f"{file_path.stem}_entity_pair_shared_entries_histogram.pdf"

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
        print(f"[Saved] {keyword_hist_path}")
        print(f"[Saved] {pair_hist_path}")


if __name__ == "__main__":
    main()
