from __future__ import annotations
import json 
from .base import (
    MemoryDataset, 
    Trajectory, 
    Session, 
    QuestionAnswerPair, 
    Message, 
)
from datetime import datetime
from typing import Dict, Any, List
def parse_session(session_data: List[dict], session_id: int, date_time: str) -> Session:
    """Parse a single session's data, including messages with images by using their captions."""
    messages = []
    for message in session_data:
        # For messages with images, combine caption and text
        text = message.get("text", "")
        if "img_url" in message and "blip_caption" in message:
            caption_text = f"[Image: {message['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text
            
        messages.append(Message(
            role=message["speaker"],
            content=text,
            timestamp=datetime.strptime(date_time, "%I:%M %p on %d %B, %Y"),
            metadata={
                "dia_id": message["dia_id"],
            },
        ))
    return Session(
        messages = messages,
        timestamp=datetime.strptime(date_time, "%I:%M %p on %d %B, %Y"),
        metadata={
            "id": str(session_id),
        },
    )

def parse_conversation(conv_data: dict) -> List[Session]:
    """Parse conversation data."""
    sessions = []
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
                # Only add sessions that have turns after filtering
                if session.messages:
                    sessions.append(session)
    
    return sessions

question_category = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversial"
}

class LoCoMo(MemoryDataset):

    @classmethod
    def read_raw_data(cls, path: str) -> LoCoMo:
        with open(path, 'r') as f:
            data = json.load(f)
        
        trajectories, question_answer_pair_lists = [], [] 
        for sample in data:
            # Parse QA data
            qa_list = []
            
            for qa in sample["qa"]:
                answer = qa.get("answer", qa.get("adversarial_answer"))
                if isinstance(answer, int): 
                    answer = str(answer)
                question_answer_pair = QuestionAnswerPair(
                    role="user",
                    question=qa["question"],
                    answer_list=(answer,),
                    timestamp=datetime(2025, 1, 1),
                    metadata={
                        "evidence": qa["evidence"],
                        "question_type": question_category[str(qa["category"])],
                        "sample_id": sample["sample_id"],
                    },
                )
                qa_list.append(question_answer_pair)
            trajectory = parse_conversation(sample["conversation"])
            trajectories.append(Trajectory(
                sessions = trajectory,
                metadata = {
                    "id": f"locomo_{sample['sample_id']}",
                }
            ))
            question_answer_pair_lists.append(qa_list)
        return cls(
            trajectories = trajectories,
            question_answer_pair_lists = question_answer_pair_lists,
        )

    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate the metadata of the dataset."""
        dataset_metadata = {
            "name": "LoCoMo",
            "paper": "Evaluating Very Long-Term Conversational Memory of LLM Agents", 
            "codebase_url": "https://snap-research.github.io/locomo/", 
            "total_sessions": 0, 
            "total_messages": 0, 
            "total_questions": 0, 
            "size": len(self)
        } 
        question_category_stats = {}  

        unique_trajectories = {}
        for trajectory, question_answer_pair_list in self: 
            trajectory_id = trajectory.metadata["id"]
            if trajectory_id not in unique_trajectories:
                unique_trajectories[trajectory_id] = trajectory
                dataset_metadata["total_sessions"] += len(trajectory)
                dataset_metadata["total_messages"] += sum(len(session) for session in trajectory)
            dataset_metadata["total_questions"] += len(question_answer_pair_list)
            for question_answer_pair in question_answer_pair_list: 
                question_category = question_answer_pair.metadata["question_type"]
                question_category_stats[question_category] = question_category_stats.get(question_category, 0) + 1

        dataset_metadata["question_category_stats"] = question_category_stats
        dataset_metadata["avg_session_per_trajectory"] = dataset_metadata["total_sessions"] / len(self)
        dataset_metadata["avg_message_per_session"] = dataset_metadata["total_messages"] / dataset_metadata["total_sessions"]
        dataset_metadata["avg_question_per_trajectory"] = dataset_metadata["total_questions"] / len(self)

        return dataset_metadata