from __future__ import annotations
import os
import json 
from base import (
    MemoryDataset, 
    Trajectory, 
    Session, 
    QuestionAnswerPair, 
    Message, 
)
from datetime import datetime
from typing import Dict, Any, List, Tuple

def parse_session(session_data: dict) -> Tuple[Session, List[QuestionAnswerPair]]:
    session_identifier = session_data["session_identifier"]
    session_uuid = session_data["session_uuid"]
    date_time = session_data["current_time"]
    timestamp = datetime.strptime(date_time, "%Y-%m-%d (%A)")
    extracted_memory = session_data.get("extracted_memory", [])

    messages = []
    qa_pairs = []
    qa_pair = None
    for message in session_data['dialogue_turns']:
        messages.append(Message(
            role=message["speaker"],
            content=message["content"],
            timestamp=timestamp,
            metadata={
                "is_query": message.get("is_query", False),
                "query_id": message.get("query_id", ""),
                "topic": message.get("topic", ""),
                "category_name": message.get("category_name", ""),
                "session_type": message.get("session_type", ""),
                "memory_used": message.get("memory_used", []),
                "memory_session_uuids": message.get("memory_session_uuids", []),
            },
        ))
        if message.get("is_query", False):
            qa_pair = {
                "role": "user",
                "question": message["content"],
                "answer_list": ("", ), 
                "timestamp": timestamp, 
                "metadata": {
                    "category_name": message.get("category_name", ""),
                    "session_type": message.get("session_type", ""),
                    "id": message.get("query_id", ""),
                    "memory_used": message.get("memory_used", []),
                    "memory_session_uuids": message.get("memory_session_uuids", []),
                }
            }
        else:
            if qa_pair is not None:
                qa_pair["answer_list"] = (message["content"], )
                qa_pair["metadata"]["memory_used"] = message.get("memory_used", [])
                qa_pair["metadata"]["memory_session_uuids"] = message.get("memory_session_uuids", [])
                qa_pair = QuestionAnswerPair(
                    role=qa_pair["role"],
                    question=qa_pair["question"],
                    answer_list=qa_pair["answer_list"],
                    timestamp=qa_pair["timestamp"],
                    metadata=qa_pair["metadata"],
                )
                qa_pairs.append(qa_pair)
                qa_pair = None

    return Session(
        messages=tuple(messages),
        timestamp=timestamp,
        metadata={
            "session_identifier": session_identifier,
            "session_uuid": session_uuid,
            "extracted_memory": extracted_memory,
        }
    ), qa_pairs

def parse_dialogues(dialogues: List[dict]) -> Tuple[Trajectory, List[QuestionAnswerPair]]:
    sessions = []
    qa_list = []
    for session_data in dialogues:
        session, qa_pairs = parse_session(session_data)
        sessions.append(session)
        qa_list.extend(qa_pairs)
    return sessions, qa_list

class RealMem(MemoryDataset):

    @classmethod
    def read_raw_data(cls, path: str) -> RealMem:
        files = [f for f in os.listdir(path) if f.endswith(".json")]
        data = []
        for file in files:
            with open(os.path.join(path, file), 'r') as f:
                data.append(json.load(f))
        trajectories, question_answer_pair_lists = [], []
        for sample in data:
            trajectory, qa_list = parse_dialogues(sample["dialogues"])
            trajectories.append(Trajectory(
                sessions=trajectory,
                metadata={k:v for k,v in sample["_metadata"].items()}
            ))
            question_answer_pair_lists.append(qa_list)
        return cls(
            trajectories=trajectories,
            question_answer_pair_lists=question_answer_pair_lists,
        )
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate the metadata of the dataset."""
        dataset_metadata = {
            "name": "RealMem",
            "paper": "RealMem: Benchmarking LLMs in Real-World Memory-Driven Interaction", 
            "codebase_url": "https://github.com/AvatarMemory/RealMemBench", 
            "total_sessions": 0, 
            "total_messages": 0, 
            "total_questions": 0, 
            "size": len(self)
        } 
        question_type_stats = {}  
        memory_category_stats = {}

        for trajectory, question_answer_pair_list in self: 
            dataset_metadata["total_sessions"] += len(trajectory)
            dataset_metadata["total_messages"] += sum(len(session) for session in trajectory)
            dataset_metadata["total_questions"] += len(question_answer_pair_list)
            for question_answer_pair in question_answer_pair_list: 
                category_name = question_answer_pair.metadata["category_name"]
                memory_category_stats[category_name] = memory_category_stats.get(category_name, 0) + 1
                question_type = question_answer_pair.metadata["session_type"]
                question_type_stats[question_type] = question_type_stats.get(question_type, 0) + 1
            # dataset_metadata["total_questions"] += sum([int(message['metadata']['is_query']) for session in trajectory for message in session])
            # for session in trajectory:
            #     for message in session:
            #         category_name = message['metadata']['category_name']
            #         memory_category_stats[category_name] = memory_category_stats.get(category_name, 0) + 1
            #         question_type = message['metadata']['question_type']
            #         question_type_stats[question_type] = question_type_stats.get(question_type, 0) + 1

        dataset_metadata["question_type_stats"] = question_type_stats
        dataset_metadata["memory_category_stats"] = memory_category_stats
        dataset_metadata["avg_session_per_trajectory"] = dataset_metadata["total_sessions"] / len(self)
        dataset_metadata["avg_message_per_session"] = dataset_metadata["total_messages"] / dataset_metadata["total_sessions"]
        dataset_metadata["avg_question_per_trajectory"] = dataset_metadata["total_questions"] / len(self)

        return dataset_metadata 

if __name__ == "__main__":
    dataset = RealMem.read_raw_data("/home/liuqiaoan/Documents/RealMemBench/dataset")
    print(dataset.metadata)
    for trajectory, question_answer_pair_list in dataset:
        for session in trajectory:
            for message in session:
                if session.metadata['session_uuid'] in message.metadata['memory_session_uuids']:
                    print(message)
