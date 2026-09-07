"""Store and search text embeddings with FAISS."""

import faiss
import numpy as np


def build_index(embeddings: list[list[float]]) -> faiss.IndexFlatIP:
	"""Create a FAISS inner-product index and add the embeddings to it.

	Args:
		embeddings: A list containing one embedding vector per text item.

	Returns:
		A FAISS index. The position of each vector in the index matches its
		position in the original embeddings list.
	"""
	# FAISS requires a two-dimensional NumPy array with float32 values.
	embeddings_array = np.asarray(embeddings, dtype="float32")
	if embeddings_array.ndim != 2 or embeddings_array.shape[0] == 0:
		raise ValueError("embeddings must contain at least one vector")

	# IndexFlatIP compares vectors using inner product (dot product).
	# For normalized embeddings, inner product is also cosine similarity.
	index = faiss.IndexFlatIP(embeddings_array.shape[1])
	index.add(embeddings_array)
	return index


def search_index(
	index: faiss.IndexFlatIP,
	query_embedding: list[float],
	top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
	"""Find the closest stored vectors to one query embedding.

	Args:
		index: The FAISS index created by ``build_index``.
		query_embedding: One embedding vector to search for.
		top_k: Maximum number of matching vectors to return.

	Returns:
		A tuple containing matching vector positions and similarity scores.
	"""
	if top_k <= 0:
		raise ValueError("top_k must be greater than zero")

	# FAISS search expects a two-dimensional float32 array, even for one query.
	query_array = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
	if query_array.shape[1] != index.d:
		raise ValueError("query_embedding has a different size than the index")

	# FAISS returns scores first and matching vector positions second.
	scores, positions = index.search(query_array, min(top_k, index.ntotal))
	return positions[0], scores[0]


if __name__ == "__main__":
	# Import here so importing this module does not load the embedding model.
	from embedder import create_embeddings

	example_texts = [
		"Employees receive 12 casual leaves per year.",
		"Employees must use the HR portal to apply for leave.",
	]

	# Create embeddings, store them, and search using the first embedding.
	embeddings = create_embeddings(example_texts)
	index = build_index(embeddings)
	positions, scores = search_index(index, embeddings[0], top_k=2)

	print(f"Matching vector positions: {positions}")
	print(f"Similarity scores: {scores}")
