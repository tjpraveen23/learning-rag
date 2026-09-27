from typing import Any


def build_metrics(
    retrieval_latency_ms: float,
    llm_latency_ms: float,
    raw_results,
    results,
    raw_context: str,
    context: str,
    usage_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Build structured RAG observability metrics.

    raw_context:
        Context that would have been sent before filtering.

    context:
        Context actually sent to the LLM after filtering.
    """

    input_tokens = usage_metadata.get(
        "input_tokens",
        0,
    )

    output_tokens = usage_metadata.get(
        "output_tokens",
        0,
    )

    total_tokens = usage_metadata.get(
        "total_tokens",
        0,
    )

    # --------------------------------------------------------
    # Chunk optimization
    # --------------------------------------------------------

    chunks_before = len(raw_results)
    chunks_after = len(results)

    chunk_reduction_percent = 0.0

    if chunks_before > 0:
        chunk_reduction_percent = (
            (chunks_before - chunks_after)
            / chunks_before
        ) * 100

    # --------------------------------------------------------
    # Context optimization
    # --------------------------------------------------------

    context_before = len(raw_context)
    context_after = len(context)

    context_reduction_percent = 0.0

    if context_before > 0:
        context_reduction_percent = (
            (context_before - context_after)
            / context_before
        ) * 100

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    return {
        "retrieval": {
            "latency_ms": round(
                retrieval_latency_ms,
                2,
            ),
            "retrieved_chunks": chunks_after,
        },

        "context": {
            "characters_before_filtering": context_before,
            "characters_after_filtering": context_after,
            "characters": context_after,
        },

        "llm": {
            "latency_ms": round(
                llm_latency_ms,
                2,
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },

        "optimization": {
            "chunks_before_filtering": chunks_before,
            "chunks_after_filtering": chunks_after,
            "chunk_reduction_percent": round(
                chunk_reduction_percent,
                2,
            ),
            "context_reduction_percent": round(
                context_reduction_percent,
                2,
            ),
        },
    }