#!/usr/bin/env python3
"""Generate one t-SNE plot per locomo10 memory-entry JSON file.

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
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot t-SNE for each locomo10 memory-entry file.")
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing locomo10_sample_*_memory_entries.json files.")
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


def main() -> None:
    args = parse_args()
    files = find_input_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_dir}/locomo10_sample_*_memory_entries.json")

    model = SentenceTransformer(args.model_path)

    for file_path in files:
        entries = load_entries(file_path)
        texts = [str(entry.get("lossless_restatement", "")).strip() for entry in entries]
        if not texts:
            print(f"[Skip] {file_path.name}: no entries")
            continue

        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        coords = compute_tsne(embeddings, seed=args.seed)

        output_path = args.output_dir / f"{file_path.stem}_tsne.png"
        plot_single_file(entries, coords, output_path, title=f"t-SNE: {file_path.name}")
        print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
