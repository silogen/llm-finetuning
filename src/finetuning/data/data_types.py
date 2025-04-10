"""Definitions of the finetuning data types

This mirrors the silocore.datasets.data_types module. The types in that module inherit from DataPoint,
which is a Beanie model. However, in training we simply want the actual data content, not the database-related
metadata. Therefore, we define the same types here, but as Pydantic models.
"""

from typing import Dict, List

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # non emptpy
    content: str  # non empty
    tags: List[str] = []
    labels: dict[str, str] = {}


class ChatConversation(BaseModel):
    dataset: str | None = None
    id: str | None = None
    messages: List[ChatMessage]
    tags: List[str] = []
    labels: dict[str, str] = {}


class DirectPreference(BaseModel):
    """Records for direct preferences by providing rejected versus preferred
    options.
    """

    dataset: str | None = None
    id: str | None = None
    prompt_messages: List[ChatMessage]
    chosen_messages: List[ChatMessage]
    rejected_messages: List[ChatMessage]
    tags: List[str] = []
    labels: Dict[str, str] = {}


data_type_by_method = {"sft": ChatConversation, "dpo": DirectPreference}
data_type_by_name = {cls.__name__: cls for cls in data_type_by_method.values()}
