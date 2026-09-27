import time

from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
)


def create_llm() -> ChatGroq:
    """
    Create the Groq LLM.
    """

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=GROQ_TEMPERATURE,
    )


def build_context(results) -> str:
    """
    Build context for the LLM from retrieved chunks.
    """

    context_parts = []

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):
        metadata = document.metadata

        context_parts.append(
            f"""
SOURCE {index}

Document: {metadata.get('document_name')}
Version: {metadata.get('version')}
Page: {metadata.get('page_number')}
Chunk ID: {metadata.get('chunk_id')}

Content:
{document.page_content}
"""
        )

    return "\n".join(context_parts)


def build_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build a grounded RAG prompt.
    """

    return f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough information,
   say exactly:

"I don't have enough information in the provided document."

4. Give a concise and clear answer.
5. Do NOT generate a Sources section.
6. Do NOT invent document names, page numbers,
   chunk IDs, paragraphs, or line numbers.

User Question:
{question}

Retrieved Context:
{context}

Answer:
"""


def generate_answer(
    llm: ChatGroq,
    question: str,
    context: str,
):
    """
    Generate the answer using Groq.

    Returns:
        answer
        usage metadata
        LLM latency
    """

    prompt = build_prompt(
        question,
        context,
    )

    start_time = time.perf_counter()

    response = llm.invoke(prompt)

    end_time = time.perf_counter()

    llm_latency_ms = (
        end_time - start_time
    ) * 1000

    answer = response.content

    usage_metadata = getattr(
        response,
        "usage_metadata",
        {},
    )

    return (
        answer,
        usage_metadata,
        llm_latency_ms,
    )