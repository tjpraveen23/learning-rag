from pathlib import Path
import json
import pymupdf

PDF_PATH = Path("data/original/PDF-Sample1.pdf")
OUTPUT_PATH = Path("data/extracted/PDF-Sample1.json")


def extract_pdf(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path.resolve())

    doc = pymupdf.open(pdf_path)
    pages = []

    for page_index, page in enumerate(doc):
        page_data = {
            "page_number": page_index + 1,
            "width": page.rect.width,
            "height": page.rect.height,
            "blocks": [],
        }

        data = page.get_text("dict")

        for block_index, block in enumerate(data["blocks"]):
            if block.get("type") != 0:
                continue

            block_data = {
                "block_index": block_index,
                "lines": [],
            }

            for line_index, line in enumerate(block["lines"]):
                line_data = {
                    "line_index": line_index,
                    "spans": [],
                }

                for span_index, span in enumerate(line["spans"]):
                    if not span["text"].strip():
                        continue

                    line_data["spans"].append({
                        "span_index": span_index,
                        "text": span["text"],
                        "bbox": [
                            round(x, 2) for x in span["bbox"]
                        ],
                        "font": span["font"],
                        "size": round(span["size"], 2),
                    })

                if line_data["spans"]:
                    block_data["lines"].append(line_data)

            if block_data["lines"]:
                page_data["blocks"].append(block_data)

        pages.append(page_data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(pages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Extracted pages: {len(pages)}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    extract_pdf(PDF_PATH)