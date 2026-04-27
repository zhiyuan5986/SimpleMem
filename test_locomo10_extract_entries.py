"""
LoCoMo10 extraction-only script.

Builds memory entries per sample and saves:
- memory entries JSON
- extraction trace JSON
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from database.vector_store import VectorStore
from main import SimpleMemSystem
from test_locomo10 import load_locomo_dataset


def sample_to_dialogues(sample) -> list:
    dialogues = []
    dialogue_id = 1
    for session_id in sorted(sample.conversation.sessions.keys()):
        session = sample.conversation.sessions[session_id]
        for turn in session.turns:
            dialogues.append(
                {
                    "dialogue_id": dialogue_id,
                    "speaker": turn.speaker,
                    "content": turn.text,
                    "timestamp": session.date_time,
                    "metadata": {
                        "session_id": session_id,
                        "session_date_time": session.date_time,
                        "turn_metadata": turn.metadata,
                    },
                }
            )
            dialogue_id += 1
    return dialogues


def main():
    parser = argparse.ArgumentParser(description="Extract LoCoMo10 memory entries only.")
    parser.add_argument("--dataset", type=str, default="test_ref/locomo10.json")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Directory used to save extraction artifacts.")
    args = parser.parse_args()

    print("Initializing SimpleMem system for extraction...")
    system = SimpleMemSystem(clear_db=True)
    output_dir = Path(args.output_dir or system.llm_client.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_locomo_dataset(args.dataset)
    if args.num_samples is not None:
        samples = samples[: args.num_samples]

    base_db_path = Path(system.vector_store.db_path)
    table_name = system.vector_store.table_name

    for sample_idx, sample in enumerate(samples):
        print(f"\n[Extract] sample={sample_idx}")
        sample_db_path = base_db_path / f"locomo10_sample_{sample_idx}"
        sample_vector_store = VectorStore(
            db_path=str(sample_db_path),
            embedding_model=system.embedding_model,
            table_name=table_name,
        )
        system.vector_store = sample_vector_store
        system.memory_builder.vector_store = sample_vector_store
        system.hybrid_retriever.vector_store = sample_vector_store
        system.memory_builder.previous_entries = []
        system.memory_builder.dialogue_buffer = []
        system.memory_builder.processed_count = 0
        system.memory_builder.reset_extraction_trace()

        dialogues = sample_to_dialogues(sample)
        for d in dialogues:
            system.add_dialogue(
                speaker=d["speaker"],
                content=d["content"],
                timestamp=d["timestamp"],
            )
        system.finalize()

        all_entries = system.vector_store.get_all_entries()
        with open(output_dir / f"locomo10_sample_{sample_idx}_memory_entries.json", "w", encoding="utf-8") as f:
            json.dump([entry.model_dump() for entry in all_entries], f, ensure_ascii=False, indent=2)

        extraction_trace = system.memory_builder.get_extraction_trace()
        with open(output_dir / f"locomo10_sample_{sample_idx}_extraction_trace.json", "w", encoding="utf-8") as f:
            json.dump(extraction_trace, f, ensure_ascii=False, indent=2)

        print(f"[Extract] entries={len(all_entries)} db={sample_db_path}")


if __name__ == "__main__":
    main()
