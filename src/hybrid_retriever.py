"""Hybrid retrieval using FAISS semantic search and BM25 keyword search."""

from bm25_retriever import build_bm25_index, search_bm25
from retriever import search_documents


def normalize_scores(results):
    """Normalize retrieval scores to the range 0 to 1."""

    if not results:
        return results

    scores = [result["score"] for result in results]

    minimum = min(scores)
    maximum = max(scores)

    # If all scores are identical, give every result a score of 1.
    if maximum == minimum:
        for result in results:
            result["normalized_score"] = 1.0
        return results

    for result in results:
        result["normalized_score"] = (
            (result["score"] - minimum)
            / (maximum - minimum)
        )

    return results


def hybrid_search(
    question,
    chunks,
    faiss_index,
    bm25_index,
    top_k=3,
):
    """Combine FAISS and BM25 retrieval results."""

    # Get semantic/vector results.
    vector_results = search_documents(
        question,
        chunks,
        faiss_index,
        top_k=top_k,
    )

    # Get keyword results.
    bm25_results = search_bm25(
        bm25_index,
        chunks,
        question,
        top_k=top_k,
    )

    # Normalize both score types independently.
    normalize_scores(vector_results)
    normalize_scores(bm25_results)

    combined = {}

    # Add vector-search results.
    for result in vector_results:
        key = result["text"]

        combined[key] = {
            "text": result["text"],
            "metadata": result["metadata"],
            "vector_score": result["normalized_score"],
            "bm25_score": 0.0,
        }

    # Add BM25 results or update existing results.
    for result in bm25_results:
        key = result["text"]

        if key not in combined:
            combined[key] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "vector_score": 0.0,
                "bm25_score": result["normalized_score"],
            }
        else:
            combined[key]["bm25_score"] = result["normalized_score"]

    # Calculate the final hybrid score.
    results = []

    for result in combined.values():
        result["score"] = (
            0.6 * result["vector_score"]
            + 0.4 * result["bm25_score"]
        )

        results.append(result)

    # Highest hybrid score first.
    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:top_k]

if __name__ == "__main__":
    from pathlib import Path

    from chunker import chunk_pages
    from embedder import create_embeddings
    from pdf_loader import extract_pdf_text
    from vector_store import build_index

    project_root = Path(__file__).resolve().parent.parent

    pdf_path = project_root / "documents" / "employee_handbook.pdf"

    # Load and chunk the document.
    pages = extract_pdf_text(pdf_path)
    chunks = chunk_pages(pages)

    # Build FAISS index.
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(chunk_texts)
    faiss_index = build_index(embeddings)

    # Build BM25 index.
    bm25_index = build_bm25_index(chunks)

    question = input("Ask a question: ")

    results = hybrid_search(
        question,
        chunks,
        faiss_index,
        bm25_index,
        top_k=3,
    )

    print("\nHybrid Search Results:")

    for number, result in enumerate(results, start=1):
        print(f"\nResult {number}")
        print(f"Page: {result['metadata']['page']}")
        print(f"Vector score: {result['vector_score']:.4f}")
        print(f"BM25 score: {result['bm25_score']:.4f}")
        print(f"Hybrid score: {result['score']:.4f}")
        print(f"Text: {result['text']}")