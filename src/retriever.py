"""Retrieve the most relevant employee-handbook chunks for a question."""

from pathlib import Path

from chunker import chunk_pages
from embedder import create_embeddings
from pdf_loader import extract_pdf_text
from vector_store import build_index, search_index


def search_documents(question, chunks, index, top_k=3):
	"""Search the FAISS index and return the matching document chunks.

	Args:
		question: The user's question as a string.
		chunks: The chunk dictionaries created by ``chunk_pages``.
		index: The FAISS index created by ``build_index``.
		top_k: Maximum number of chunks to return.

	Returns:
		A list of dictionaries containing each matching chunk's text,
		metadata, and similarity score.
	"""
	# The embedder expects a list, so wrap the single question in one.
	question_embedding = create_embeddings([question])[0]
	positions, scores = search_index(index, question_embedding, top_k)

	results = []
	for position, score in zip(positions, scores):
		# FAISS can use -1 for a missing result in some search scenarios.
		if position == -1:
			continue

		matching_chunk = chunks[int(position)]
		results.append(
			{
				"text": matching_chunk["text"],
				"metadata": matching_chunk["metadata"],
				"score": float(score),
			}
		)

	return results


if __name__ == "__main__":
	# Build the retrieval data once when this script starts.
	project_root = Path(__file__).resolve().parent.parent
	pdf_path = project_root / "documents" / "employee_handbook.pdf"

	pages = extract_pdf_text(pdf_path)
	chunks = chunk_pages(pages)
	chunk_texts = [chunk["text"] for chunk in chunks]
	embeddings = create_embeddings(chunk_texts)
	index = build_index(embeddings)

	question = input("Ask a question about the employee handbook: ")
	results = search_documents(question, chunks, index)

	for number, result in enumerate(results, start=1):
		print(f"\nResult {number}")
		print(f"Page: {result['metadata']['page']}")
		print(f"Similarity score: {result['score']:.4f}")
		print(f"Text: {result['text']}")
