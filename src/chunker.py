"""Split document pages into meaningful policy-section chunks."""

import re
from typing import TypedDict


class TextChunk(TypedDict):
    """A text chunk and its PDF page."""

    text: str
    metadata: dict[str, int]


def chunk_pages(
    pages: list[tuple[int, str]],
    max_chars: int = 500,
    source_name: str = "unknown",
) -> list[TextChunk]:
    """Split document text into section-aware chunks."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    chunks: list[TextChunk] = []

    for page_number, page_text in pages:
        lines = [
            line.strip()
            for line in page_text.splitlines()
            if line.strip()
        ]

        current_chunk = ""

        for line in lines:

            # Detect numbered section headings such as:
            # 1. Leave Policy
            # 2. Work From Home Policy
            # 3. IT Security Policy
            # 4. Employee Benefits
            is_heading = bool(
                re.match(r"^\d+\.\s+", line)
            )

            if is_heading and current_chunk:
                chunks.append(
                    {
                        "text": current_chunk,
                       "metadata": {
    "page": page_number,
    "source": source_name,
},
                    }
                )
                current_chunk = line
                continue

            if not current_chunk:
                current_chunk = line

            elif len(current_chunk) + len(line) + 1 <= max_chars:
                current_chunk += " " + line

            else:
                chunks.append(
                    {
                        "text": current_chunk,
                        "metadata": {
    "page": page_number,
    "source": source_name,
},
                    }
                )

                current_chunk = line

        if current_chunk:
            chunks.append(
                {
                    "text": current_chunk,
                    "metadata": {
                        "page": page_number,
                        "source": source_name,
                    },
                }
            )

    return chunks


if __name__ == "__main__":
    example_pages = [
        (
            1,
            "1. Leave Policy\n\n"
            "Employees receive 12 casual leaves per year.\n\n"
            "Employees apply through the HR portal.\n\n"
            "2. Work From Home Policy\n\n"
            "Employees can work remotely two days per week.",
        )
    ]

    chunks = chunk_pages(example_pages)

    for number, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {number}")
        print(f"Page: {chunk['metadata']['page']}")
        print(chunk["text"])