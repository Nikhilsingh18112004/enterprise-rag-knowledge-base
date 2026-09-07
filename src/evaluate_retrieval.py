"""Evaluate the RAG retrieval pipeline."""

import csv
from pathlib import Path

from bm25_retriever import build_bm25_index
from chunker import chunk_pages
from embedder import create_embeddings
from hybrid_retriever import hybrid_search
from pdf_loader import extract_pdf_text
from reranker import rerank_results
from vector_store import build_index


def load_evaluation_questions(csv_path):
    """Load evaluation questions from a CSV file."""

    questions = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            questions.append(row)

    return questions


def evaluate_retrieval(
    questions,
    chunks,
    faiss_index,
    bm25_index,
):
    """Calculate Recall@1, Recall@3, and Recall@5."""

    correct_at_1 = 0
    correct_at_3 = 0
    correct_at_5 = 0

    for item in questions:

        question = item["question"]
        expected_section = item["expected_section"]

        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print(f"Expected section: {expected_section}")

        # --------------------------------------------------
        # Stage 1: Hybrid retrieval
        # --------------------------------------------------

        hybrid_results = hybrid_search(
            question,
            chunks,
            faiss_index,
            bm25_index,
            top_k=10,
        )

        # --------------------------------------------------
        # Stage 2: Cross-Encoder reranking
        # --------------------------------------------------

        reranked_results = rerank_results(
            question,
            hybrid_results,
            top_k=5,
        )

        # --------------------------------------------------
        # Display retrieved chunks
        # --------------------------------------------------

        print("\nRetrieved chunks:")

        for position, result in enumerate(
            reranked_results,
            start=1,
        ):
            metadata = result.get(
                "metadata",
                {},
            )

            print(f"\nPosition {position}")

            print(
                f"Reranker score: "
                f"{result['reranker_score']:.4f}"
            )

            print(
                f"Hybrid score: "
                f"{result.get('score', 0.0):.4f}"
            )

            print(
                f"Source: "
                f"{metadata.get('source', 'unknown')}"
            )

            print(
                f"Page: "
                f"{metadata.get('page', 'unknown')}"
            )

            print(
                f"Text: {result['text']}"
            )

        # --------------------------------------------------
        # Find expected section
        # --------------------------------------------------

        found_positions = []

        for index, result in enumerate(
            reranked_results
        ):
            text = result["text"].lower()

            if expected_section.lower() in text:
                found_positions.append(index)

        # --------------------------------------------------
        # Calculate Recall@K
        # --------------------------------------------------

        if found_positions:

            first_position = found_positions[0]

            print(
                f"\nCorrect section found at "
                f"position {first_position + 1}"
            )

            if first_position < 1:
                correct_at_1 += 1

            if first_position < 3:
                correct_at_3 += 1

            if first_position < 5:
                correct_at_5 += 1

        else:

            print(
                "\nCorrect section NOT found "
                "in top 5."
            )

    # ------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------

    total = len(questions)

    if total == 0:
        print("\nNo evaluation questions found.")
        return

    recall_at_1 = correct_at_1 / total
    recall_at_3 = correct_at_3 / total
    recall_at_5 = correct_at_5 / total

    print("\n")
    print("=" * 40)
    print("Retrieval Evaluation Results")
    print("=" * 40)

    print(f"Total questions: {total}")
    print(f"Recall@1: {recall_at_1:.2%}")
    print(f"Recall@3: {recall_at_3:.2%}")
    print(f"Recall@5: {recall_at_5:.2%}")

    print("=" * 40)


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

    pages = extract_pdf_text(
        pdf_path
    )

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

    embeddings = create_embeddings(
        chunk_texts
    )

    faiss_index = build_index(
        embeddings
    )

    print("Building BM25 index...")

    bm25_index = build_bm25_index(
        chunks
    )

    print("Loading evaluation questions...")

    questions = load_evaluation_questions(
        csv_path
    )

    print(
        f"Running evaluation on "
        f"{len(questions)} questions..."
    )

    evaluate_retrieval(
        questions,
        chunks,
        faiss_index,
        bm25_index,
    )