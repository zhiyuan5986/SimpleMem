#!/usr/bin/env python3
"""Build and visualize bipartite graphs for locomo10 memory-entry JSON files.

Usage:
    python scripts/plot_locomo_bipartite.py --input-dir . --output-dir outputs/bipartite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


if __name__ == "__main__":
    main()
