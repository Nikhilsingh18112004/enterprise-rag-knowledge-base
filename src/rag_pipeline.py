"""End-to-end RAG pipeline using hybrid retrieval."""
from reranker import rerank_results 


from pathlib import Path

from bm25_retriever import build_bm25_index
from chunker import chunk_pages
from embedder import create_embeddings
from langchain_generator import generate_answer
from hybrid_retriever import hybrid_search
from pdf_loader import extract_pdf_text
from vector_store import build_index


def build_rag_system(pdf_path):
    """Load a PDF and prepare FAISS and BM25 indexes."""

    pages = extract_pdf_text(pdf_path)

    chunks = chunk_pages(
    pages,
    source_name=Path(pdf_path).name,
)

    chunk_texts = [chunk["text"] for chunk in chunks]

    # Build FAISS vector index.
    embeddings = create_embeddings(chunk_texts)
    faiss_index = build_index(embeddings)

    # Build BM25 keyword index.
    bm25_index = build_bm25_index(chunks)

    return chunks, faiss_index, bm25_index


def ask_question(
    question,
    chunks,
    faiss_index,
    bm25_index,
):
    """Retrieve, rerank, and generate an answer."""

    # Stage 1: Hybrid retrieval
    retrieved_chunks = hybrid_search(
        question,
        chunks,
        faiss_index,
        bm25_index,
        top_k=10,
    )

    if not retrieved_chunks:
        return "The information is not available in the provided documents."

    # Stage 2: Cross-Encoder reranking
    reranked_chunks = rerank_results(
        question,
        retrieved_chunks,
        top_k=3,
    )

    if not reranked_chunks:
        return "The information is not available in the provided documents."

    # Display reranker score.
    best_score = reranked_chunks[0]["reranker_score"]

    print(f"\nBest reranker score: {best_score:.4f}")

    CONFIDENCE_THRESHOLD = 0.0

    if best_score < CONFIDENCE_THRESHOLD:
        return (
            "The information is not available "
            "in the provided documents."
        )

    answer = generate_answer(
        question,
        reranked_chunks,
    )

    return answer


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    pdf_path = (
        project_root
        / "documents"
        / "employee_handbook.pdf"
    )

    print("Loading document and building RAG system...")

    chunks, faiss_index, bm25_index = build_rag_system(
        pdf_path
    )

    print("RAG system is ready!")

    question = input("\nAsk a question: ")

    answer = ask_question(
        question,
        chunks,
        faiss_index,
        bm25_index,
    )

    print("\nAnswer:")
    print(answer)