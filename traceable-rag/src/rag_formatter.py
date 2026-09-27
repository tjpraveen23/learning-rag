from rag_models import RAGResponse


def print_response(
    response: RAGResponse,
) -> None:
    """
    Display the RAG response in the console.
    """

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    print("\nAnswer")
    print("=" * 70)

    print(response.answer)

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    print("\nSources / Traceability")
    print("=" * 70)

    if not response.sources:

        print("No sources retrieved.")

    else:

        for source in response.sources:

            print(
                f"""
Source {source.rank}
--------------------------------------------------

📄 Document:
{source.document}

📑 Version:
{source.version}

📑 Page:
{source.page}

📝 Relevant content:
{source.content}

📍 Source location:
Chunk ID: {source.chunk_id}

🔎 Retrieval distance:
{source.retrieval_distance}
"""
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print("\nObservability Metrics")
    print("=" * 70)

    retrieval_metrics = response.metrics[
        "retrieval"
    ]

    context_metrics = response.metrics[
        "context"
    ]

    llm_metrics = response.metrics[
        "llm"
    ]

    optimization_metrics = response.metrics[
        "optimization"
    ]

    context_metrics = response.metrics[
    "context"
    ]
    print(
        f"📦 Chunks before filtering: "
        f"{optimization_metrics['chunks_before_filtering']}"
    )

    print(
        f"📦 Chunks after filtering: "
        f"{optimization_metrics['chunks_after_filtering']}"
    )

    print(
        f"📉 Chunk reduction: "
        f"{optimization_metrics['chunk_reduction_percent']}%"
    )

    print(
        f"📝 Context before filtering: "
        f"{context_metrics['characters_before_filtering']} characters"
    )

    print(
        f"📝 Context after filtering: "
        f"{context_metrics['characters_after_filtering']} characters"
    )

    print(
        f"📉 Context reduction: "
        f"{optimization_metrics['context_reduction_percent']}%"
    )

    print(
        f"⏱️ Retrieval latency: "
        f"{retrieval_metrics['latency_ms']} ms"
    )

    print(
        f"📦 Retrieved chunks: "
        f"{retrieval_metrics['retrieved_chunks']}"
    )

    print(
        f"📝 Context size: "
        f"{context_metrics['characters']} characters"
    )

    print(
        f"⏱️ LLM latency: "
        f"{llm_metrics['latency_ms']} ms"
    )

    print(
        f"🔢 Input tokens: "
        f"{llm_metrics['input_tokens']}"
    )

    print(
        f"🔢 Output tokens: "
        f"{llm_metrics['output_tokens']}"
    )

    print(
        f"🔢 Total tokens: "
        f"{llm_metrics['total_tokens']}"
    )