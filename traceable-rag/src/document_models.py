from typing import Literal

from pydantic import BaseModel, Field


BlockType = Literal[
    "heading",
    "paragraph",
    "table",
    "image",
    "chart",
    "footnote",
    "caption",
    "header",
    "footer",
    "list",
    "unknown",
]


class Provenance(BaseModel):
    """
    Identifies where a block originated in the source.
    """

    page_number: int
    char_start: int | None = None
    char_end: int | None = None
    bbox: list[float] | None = None
    extraction_method: str = "text"
    extraction_confidence: float | None = None


class DocumentBlock(BaseModel):
    """
    One logical content element extracted from a document.
    """

    block_id: str
    block_type: BlockType
    text: str = ""
    order: int
    provenance: Provenance

    section_path: list[str] = Field(
        default_factory=list
    )

    metadata: dict = Field(
        default_factory=dict
    )


class DocumentPage(BaseModel):
    """
    Content and dimensions for one source page.
    """

    page_number: int
    width: float | None = None
    height: float | None = None

    blocks: list[DocumentBlock] = Field(
        default_factory=list
    )


class NormalizedDocument(BaseModel):
    """
    Common representation consumed by downstream
    ingestion, chunking, and retrieval components.
    """

    document_id: str
    document_name: str
    version: str
    content_hash: str | None = None
    source_format: str
    pages: list[DocumentPage] = Field(
        default_factory=list
    )

    metadata: dict = Field(
        default_factory=dict
    )