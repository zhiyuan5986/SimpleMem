#!/usr/bin/env python3
"""Generate t-SNE plots for LoCoMo memory entries and query-similarity analysis.

Usage:
    python scripts/plot_locomo_tsne.py \
        --input-dir . \
        --model-path /mnt/sdb/liuqiaoan/all-MiniLM-L6-v2 \
        --output-dir outputs/tsne
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot t-SNE for locomo10 memory entries and query similarity.")
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing locomo10_sample_*_memory_entries.json files.")
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=Path("test_ref/locomo10.json"),
        help="LoCoMo dataset JSON for per-sample query-similarity t-SNE.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/sdb/liuqiaoan/all-MiniLM-L6-v2",
        help="Local sentence-transformers model path.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tsne"), help="Directory to save PNG plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def find_input_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("locomo10_sample_*_memory_entries.json"))


def load_entries(file_path: Path) -> list[dict]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{file_path} is not a list of dict entries.")
    return data


def compute_tsne(embeddings: np.ndarray, seed: int) -> np.ndarray:
    n_samples = embeddings.shape[0]
    if n_samples == 1:
        return np.array([[0.0, 0.0]])

    perplexity = min(30, n_samples - 1)
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def compute_similarity_tsne(embeddings: np.ndarray, seed: int) -> np.ndarray:
    n_samples = embeddings.shape[0]
    if n_samples == 1:
        return np.array([[0.0, 0.0]])

    similarity = cosine_similarity(embeddings)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    # fill negative values with 0.0
    distance[distance < 0.0] = 0.0

    perplexity = min(30, n_samples - 1)
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        metric="precomputed",
    )
    return tsne.fit_transform(distance)


def plot_single_file(entries: list[dict], coordinates: np.ndarray, output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 7))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=45, alpha=0.8)

    for i, (entry, point) in enumerate(zip(entries, coordinates)):
        entry_id = str(entry.get("entry_id", i))[:8]
        plt.annotate(entry_id, (point[0], point[1]), fontsize=7, alpha=0.75)

    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_query_similarity_sample(sample_idx: int, questions: list[str], coordinates: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(11, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=50, alpha=0.82)

    for i, point in enumerate(coordinates):
        plt.annotate(str(i), (point[0], point[1]), fontsize=7, alpha=0.8)

    plt.title(f"Sample {sample_idx} query similarity t-SNE (index labels)")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")

    preview_count = min(10, len(questions))
    preview_lines = [f"{i}: {questions[i][:80]}" for i in range(preview_count)]
    preview = "\n".join(preview_lines)
    if len(questions) > preview_count:
        preview += f"\n... ({len(questions) - preview_count} more queries)"
    plt.gcf().text(0.01, 0.01, preview, fontsize=8, va="bottom", ha="left")

    plt.tight_layout(rect=[0.05, 0.05, 1.0, 1.0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def load_dataset_samples(dataset_json: Path) -> list[dict]:
    with dataset_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{dataset_json} is not a list of samples.")
    return data


def extract_sample_queries(sample: dict) -> list[str]:
    qa_list = sample.get("qa", [])
    if not isinstance(qa_list, list):
        return []

    queries: list[str] = []
    for qa in qa_list:
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question", "")).strip()
        if question:
            queries.append(question)
    return queries


def main() -> None:
    args = parse_args()
    model = SentenceTransformer(args.model_path)
    files = find_input_files(args.input_dir)
    did_work = False

    for file_path in files:
        entries = load_entries(file_path)
        texts = [str(entry.get("lossless_restatement", "")).strip() for entry in entries]
        if not texts:
            print(f"[Skip] {file_path.name}: no entries")
            continue
        print(f"[Loaded] {file_path.name}: {len(texts)} entries")

        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        coords = compute_tsne(embeddings, seed=args.seed)

        output_path = args.output_dir / f"{file_path.stem}_tsne.png"
        plot_single_file(entries, coords, output_path, title=f"t-SNE: {file_path.name}")
        print(f"[Saved] {output_path}")
        did_work = True

    if args.dataset_json.exists():
        dataset_samples = load_dataset_samples(args.dataset_json)
        for sample_idx, sample in enumerate(dataset_samples):
            queries = extract_sample_queries(sample)
            if len(queries) < 2:
                print(f"[Skip] sample {sample_idx}: not enough queries ({len(queries)})")
                continue
            query_embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)
            query_coords = compute_similarity_tsne(query_embeddings, seed=args.seed)
            output_path = args.output_dir / f"locomo10_sample_{sample_idx}_query_similarity_tsne.png"
            plot_query_similarity_sample(sample_idx, queries, query_coords, output_path)
            print(f"[Saved] {output_path}")
            did_work = True
    else:
        print(f"[Warn] dataset JSON not found: {args.dataset_json}")

    if not did_work:
        raise FileNotFoundError(
            "No plots generated. Check memory-entry files and dataset path."
        )


if __name__ == "__main__":
    main()
