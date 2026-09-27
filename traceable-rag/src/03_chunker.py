import json
from pathlib import Path
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_DATA_PATH,
    NORMALIZED_DATA_PATH
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_normalized_pages(input_file: str) -> list[dict]:
    """Load normalized page-level documents."""

    with open(input_file, "r", encoding="utf-8") as file:
        return json.load(file)


def create_chunks(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split page text into smaller chunks while preserving
    document and page-level provenance.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):

            chunk_id = (
                f"{page['document_id']}"
                f"_p{page['page_number']}"
                f"_c{chunk_index}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "document_id": page["document_id"],
                        "document_name": page["document_name"],
                        "version": page["version"],
                        "page_number": page["page_number"],
                    },
                }
            )

    return chunks


def save_chunks(chunks: list[dict], output_file: str) -> None:
    """Save chunks as JSON."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":

    input_file = NORMALIZED_DATA_PATH / "PDF-Sample1.json"
    output_file = CHUNKS_DATA_PATH / "PDF-Sample1.json"

    pages = load_normalized_pages(input_file)

    chunks = create_chunks(
        pages,
        chunk_size=500,
        chunk_overlap=50,
    )

    save_chunks(chunks, output_file)

    print(f"Created {len(chunks)} chunks")
    print(f"Output: {output_file}")

    print("\nChunk summary:")

    for chunk in chunks:
        metadata = chunk["metadata"]

        print(
            f"- {chunk['chunk_id']} "
            f"| page={metadata['page_number']} "
            f"| characters={len(chunk['text'])}"
        )