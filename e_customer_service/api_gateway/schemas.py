from typing import Any, Literal

from pydantic import BaseModel, Field


Intent = Literal["customer_service", "research_report"]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: str | None = None
    session_id: str | None = None
    user_profile: dict[str, Any] | None = None
    order_info: dict[str, Any] | None = None


class SourceDocument(BaseModel):
    title: str | None = None
    content: str
    score: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    intent: Intent
    source: Literal["cache", "llm", "rag"]
    citations: list[SourceDocument] = Field(default_factory=list)

