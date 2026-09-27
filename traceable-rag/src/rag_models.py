from typing import Any

from pydantic import BaseModel


class Source(BaseModel):
    """
    Represents one retrieved source used by the RAG pipeline.
    """

    rank: int
    document: str
    version: str
    page: int | str
    chunk_id: str
    content: str
    retrieval_distance: float


class RAGResponse(BaseModel):
    """
    Complete structured response returned by the RAG pipeline.
    """

    question: str
    answer: str
    sources: list[Source]
    metrics: dict[str, Any]