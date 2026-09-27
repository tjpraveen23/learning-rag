import time

from rag_llm import (
    build_context,
    create_llm,
    generate_answer,
)

from rag_metrics import build_metrics

from rag_models import (
    RAGResponse,
    Source,
)

from rag_formatter import print_response

from retriever import (
    create_vector_store,
    retrieve_documents,
)


def build_sources(
    results,
) -> list[Source]:
    """
    Convert retrieval results into structured Source objects.

    Source information comes from retrieval metadata,
    not from the LLM.
    """

    sources = []

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):
        metadata = document.metadata

        source = Source(
            rank=index,
            document=metadata.get(
                "document_name",
                "Unknown document",
            ),
            version=metadata.get(
                "version",
                "Unknown version",
            ),
            page=metadata.get(
                "page_number",
                "Unknown page",
            ),
            chunk_id=metadata.get(
                "chunk_id",
                "Unknown chunk",
            ),
            content=document.page_content.strip(),
            retrieval_distance=round(
                score,
                4,
            ),
        )

        sources.append(source)

    return sources


def run_rag(
    vector_store,
    llm,
    question: str,
) -> RAGResponse:
    """
    Execute the complete RAG pipeline.
    """

    # --------------------------------------------------------
    # 1. Retrieval
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    raw_results, results = retrieve_documents(
        vector_store,
        question,
    )

    retrieval_end = time.perf_counter()

    retrieval_latency_ms = (
        retrieval_end - retrieval_start
    ) * 1000

    # --------------------------------------------------------
    # 2. Build context
    # --------------------------------------------------------

    raw_context = build_context(raw_results)
    context = build_context(results)

    # --------------------------------------------------------
    # 3. Generate answer
    # --------------------------------------------------------

    (
        answer,
        usage_metadata,
        llm_latency_ms,
    ) = generate_answer(
        llm,
        question,
        context,
    )

    # --------------------------------------------------------
    # 4. Build sources
    # --------------------------------------------------------

    sources = build_sources(results)

    # --------------------------------------------------------
    # 5. Build metrics
    # --------------------------------------------------------

    metrics = build_metrics(
    retrieval_latency_ms=retrieval_latency_ms,
    llm_latency_ms=llm_latency_ms,
    raw_results=raw_results,
    results=results,
    raw_context=raw_context,
    context=context,
    usage_metadata=usage_metadata,
    )

    # --------------------------------------------------------
    # 6. Return structured response
    # --------------------------------------------------------

    return RAGResponse(
        question=question,
        answer=answer,
        sources=sources,
        metrics=metrics,
    )


def main():
    """
    Application entry point.
    """

    vector_store = create_vector_store()

    llm = create_llm()

    print("\nTraceable RAG — Groq")
    print("Type 'exit' to stop.\n")

    while True:

        question = input(
            "Enter your question: "
        ).strip()

        if question.lower() == "exit":

            print("\nExiting...")

            break

        if not question:

            print(
                "Please enter a question.\n"
            )

            continue

        response = run_rag(
            vector_store=vector_store,
            llm=llm,
            question=question,
        )

        print_response(response)

        print(
            "\n"
            + "=" * 70
            + "\n"
        )


if __name__ == "__main__":
    main()