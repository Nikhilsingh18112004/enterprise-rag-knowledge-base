"""Keyword-based retrieval using BM25."""

import re

from rank_bm25 import BM25Okapi


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(chunks):
    """Build a BM25 index from document chunks."""

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def search_bm25(bm25, chunks, question, top_k=3):
    """Return the chunks most relevant to the question."""

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    ranked_positions = sorted(
        range(len(scores)),
        key=lambda position: scores[position],
        reverse=True,
    )[:top_k]

    results = []

    for position in ranked_positions:
        results.append(
            {
                "text": chunks[position]["text"],
                "metadata": chunks[position]["metadata"],
                "score": float(scores[position]),
            }
        )

    return results

if __name__ == "__main__":
    from pathlib import Path

    from chunker import chunk_pages
    from pdf_loader import extract_pdf_text

    project_root = Path(__file__).resolve().parent.parent

    pdf_path = project_root / "documents" / "employee_handbook.pdf"

    pages = extract_pdf_text(pdf_path)
    chunks = chunk_pages(pages)

    bm25 = build_bm25_index(chunks)

    question = input("Ask a question: ")

    results = search_bm25(
        bm25,
        chunks,
        question,
        top_k=3,
    )

    for number, result in enumerate(results, start=1):
        print(f"\nResult {number}")
        print(f"Page: {result['metadata']['page']}")
        print(f"BM25 score: {result['score']:.4f}")
        print(f"Text: {result['text']}")