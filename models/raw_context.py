"""Data model for entry-aligned raw context spans."""
from typing import Any, Dict
from pydantic import BaseModel, Field


class RawContextEntry(BaseModel):
    """Raw context span storage row."""

    entry_id: str
    text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
