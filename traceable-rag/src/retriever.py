from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    RETRIEVAL_TOP_K,
    RETRIEVAL_MAX_DISTANCE,
    RETRIEVAL_MIN_CHUNKS,
)


def create_vector_store() -> Chroma:
    """Load the existing Chroma vector store."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIRECTORY),
    )

def filter_results_by_distance(
    results,
):
    """
    Remove weak retrieval results using the configured
    maximum distance.

    At least RETRIEVAL_MIN_CHUNKS are retained.
    """

    if not results:
        return []

    filtered_results = [
        result
        for result in results
        if result[1] <= RETRIEVAL_MAX_DISTANCE
    ]

    if len(filtered_results) < RETRIEVAL_MIN_CHUNKS:
        filtered_results = results[
            :RETRIEVAL_MIN_CHUNKS
        ]

    return filtered_results

def retrieve_documents(
    vector_store,
    question: str,
    top_k: int = RETRIEVAL_TOP_K,
):
    """
    Retrieve candidate documents and apply deterministic
    distance-based context filtering.
    """

    raw_results = (
        vector_store.similarity_search_with_score(
            question,
            k=top_k,
        )
    )

    filtered_results = filter_results_by_distance(
        raw_results
    )

    return (
        raw_results,
        filtered_results,
    )

def print_results(results) -> None:
    """Display retrieved chunks, scores, and provenance."""

    print("\nRetrieved Chunks")
    print("=" * 70)

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):

        metadata = document.metadata

        print(f"\nResult {index}")
        print("-" * 70)

        print(f"Score      : {score:.4f}")
        print(f"Chunk ID   : {metadata.get('chunk_id')}")
        print(f"Document   : {metadata.get('document_name')}")
        print(f"Version    : {metadata.get('version')}")
        print(f"Page       : {metadata.get('page_number')}")

        print("\nText:")
        print(document.page_content)

if __name__ == "__main__":

    vector_store = create_vector_store()

    print("\nTraceable RAG — Semantic Retrieval")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("Enter your question: ").strip()

        if question.lower() == "exit":
            print("\nExiting...")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        results = retrieve_documents(
            vector_store,
            question,
        )

        print_results(results)
        print("\n" + "=" * 70 + "\n")

