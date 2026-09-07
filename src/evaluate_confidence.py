"""Evaluate confidence scores for answerable and unanswerable questions."""

import csv
from pathlib import Path

from bm25_retriever import build_bm25_index
from chunker import chunk_pages
from embedder import create_embeddings
from hybrid_retriever import hybrid_search
from pdf_loader import extract_pdf_text
from reranker import rerank_results
from vector_store import build_index


def load_questions(csv_path):
    """Load evaluation questions from CSV."""
    questions = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            questions.append(row)

    return questions


def evaluate_confidence(
    questions,
    chunks,
    faiss_index,
    bm25_index,
):
    """Print confidence scores for all evaluation questions."""

    for item in questions:
        question = item["question"]
        answerable = item["answerable"]

        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print(f"Expected answer: {answerable}")

        # Step 1: Hybrid retrieval
        hybrid_results = hybrid_search(
            question,
            chunks,
            faiss_index,
            bm25_index,
            top_k=10,
        )

        # Step 2: Cross-Encoder reranking
        reranked_results = rerank_results(
            question,
            hybrid_results,
            top_k=3,
        )

        if not reranked_results:
            print("No results found.")
            continue

        best_result = reranked_results[0]

        print("\nBest retrieved result:")
        print(
            f"Reranker score: "
            f"{best_result['reranker_score']:.4f}"
        )

        print(
            f"Hybrid score: "
            f"{best_result.get('score', 0.0):.4f}"
        )

        print(
            f"Vector score: "
            f"{best_result.get('vector_score', 0.0):.4f}"
        )

        print(
            f"BM25 score: "
            f"{best_result.get('bm25_score', 0.0):.4f}"
        )

        print(
            f"Retrieved text: "
            f"{best_result['text']}"
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    pdf_path = (
        project_root
        / "documents"
        / "employee_handbook.pdf"
    )

    csv_path = (
        project_root
        / "tests"
        / "evaluation_questions.csv"
    )

    print("Loading document...")

    pages = extract_pdf_text(pdf_path)

    chunks = chunk_pages(
        pages,
        source_name=pdf_path.name,
    )

    print(
        f"Created {len(chunks)} document chunks."
    )

    print("Creating embeddings...")

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(chunk_texts)

    faiss_index = build_index(embeddings)

    print("Building BM25 index...")

    bm25_index = build_bm25_index(chunks)

    print("Loading evaluation questions...")

    questions = load_questions(csv_path)

    print(
        f"Running confidence evaluation "
        f"on {len(questions)} questions..."
    )

    evaluate_confidence(
        questions,
        chunks,
        faiss_index,
        bm25_index,
    )