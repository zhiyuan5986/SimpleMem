"""Data model for raw context rows (LLM spans or turns)."""
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class RawContextEntry(BaseModel):
    """Raw context storage row."""

    entry_id: str
    text: str = ""
    links: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
