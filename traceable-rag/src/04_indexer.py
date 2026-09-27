import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
    CHUNKS_DATA_PATH,
)


def load_chunks(input_file: str) -> list[dict]:
    """Load chunks from JSON."""

    with open(input_file, "r", encoding="utf-8") as file:
        return json.load(file)


def create_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model."""

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def index_chunks(
    chunks: list[dict],
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """Create Chroma vector store and index chunks."""

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            **chunk["metadata"],
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIRECTORY),
    )

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )

    return vector_store


if __name__ == "__main__":

    input_file = CHUNKS_DATA_PATH / "PDF-Sample1.json"

    chunks = load_chunks(input_file)

    print(f"Loaded {len(chunks)} chunks")

    embeddings = create_embeddings()

    vector_store = index_chunks(
        chunks,
        embeddings,
    )

    print("\nIndexing completed")
    print(f"Collection: {CHROMA_COLLECTION_NAME}")
    print(f"Vector store: {CHROMA_PERSIST_DIRECTORY}")