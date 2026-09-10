"""FastAPI backend for the Enterprise RAG system."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bm25_retriever import build_bm25_index
from chunker import chunk_pages
from embedder import create_embeddings
from hybrid_retriever import hybrid_search
from langchain_generator import generate_answer
from pdf_loader import extract_pdf_text
from reranker import rerank_results
from vector_store import build_index


app = FastAPI(
    title="Enterprise Knowledge Base API",
    description="RAG-based enterprise document question answering API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    document: str
    page: int


class QuestionResponse(BaseModel):
    answer: str
    sources: list[Source]


print("Loading document and building RAG system...")


project_root = Path(__file__).resolve().parent.parent
pdf_path = project_root / "documents" / "employee_handbook.pdf"


pages = extract_pdf_text(pdf_path)

chunks = chunk_pages(
    pages,
    source_name=pdf_path.name,
)

chunk_texts = [chunk["text"] for chunk in chunks]

embeddings = create_embeddings(chunk_texts)

faiss_index = build_index(embeddings)

bm25_index = build_bm25_index(chunks)


print("RAG system is ready!")


CONFIDENCE_THRESHOLD = 0.0


def ask_question(question: str) -> QuestionResponse:
    """Retrieve context, check confidence, and generate an answer."""

    retrieved_chunks = hybrid_search(
        question,
        chunks,
        faiss_index,
        bm25_index,
        top_k=10,
    )

    if not retrieved_chunks:
        return QuestionResponse(
            answer="The information is not available in the provided documents.",
            sources=[],
        )

    reranked_chunks = rerank_results(
        question,
        retrieved_chunks,
        top_k=3,
    )

    if not reranked_chunks:
        return QuestionResponse(
            answer="The information is not available in the provided documents.",
            sources=[],
        )

    best_score = reranked_chunks[0]["reranker_score"]

    print(f"Best reranker score: {best_score:.4f}")

    if best_score < CONFIDENCE_THRESHOLD:
        return QuestionResponse(
            answer="The information is not available in the provided documents.",
            sources=[],
        )

    answer = generate_answer(
        question,
        reranked_chunks,
    )

    sources = []

    for chunk in reranked_chunks:
        metadata = chunk.get("metadata", {})

        document = metadata.get("source", "unknown")
        page = metadata.get("page", 0)

        source = Source(
            document=document,
            page=int(page),
        )

        if source not in sources:
            sources.append(source)

    return QuestionResponse(
        answer=answer,
        sources=sources,
    )


@app.get("/")
def root():
    return {
        "message": "Enterprise RAG API is running"
    }


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    return ask_question(request.question)