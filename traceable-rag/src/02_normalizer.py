from pathlib import Path
import json

def normalize_pdf_json(
    input_file: str,
    document_id: str,
    document_name: str,
    version: str = "v1",
) -> list[dict]:

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both:
    # { "pages": [...] }
    # and
    # [ {...}, {...} ]

    if isinstance(data, dict):
        pages = data.get("pages", [])
    elif isinstance(data, list):
        pages = data
    else:
        raise ValueError("Unexpected JSON structure")

    normalized_pages = []

    for page in pages:
        page_number = page.get("page_number")

        page_lines = []

        for block in page.get("blocks", []):
            for line in block.get("lines", []):

                line_text = "".join(
                    span.get("text", "")
                    for span in line.get("spans", [])
                ).strip()

                if line_text:
                    page_lines.append(line_text)

        page_text = "\n".join(page_lines).strip()

        if not page_text:
            continue

        normalized_pages.append(
            {
                "document_id": document_id,
                "document_name": document_name,
                "version": version,
                "page_number": page_number,
                "text": page_text,
            }
        )

    return normalized_pages

if __name__ == "__main__":

    pages = normalize_pdf_json(
        input_file="data/extracted/PDF-Sample1.json",
        document_id="PDF-Sample1",
        document_name="PDF-Sample1.pdf",
        version="v1",
    )

    output_file = Path("data/normalized/PDF-Sample1.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"Normalized {len(pages)} pages")
    print(f"Output: {output_file}")