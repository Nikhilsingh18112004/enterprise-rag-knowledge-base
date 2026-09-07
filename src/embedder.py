"""Create vector embeddings for text using sentence-transformers."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer


# This is a small English embedding model that works well for semantic search.
MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
	"""Load the embedding model once and reuse it for later calls."""
	return SentenceTransformer(MODEL_NAME)


def create_embeddings(texts: list[str]) -> list[list[float]]:
	"""Convert a list of text strings into numerical embeddings.

	Args:
		texts: The text strings to convert into vectors.

	Returns:
		One embedding vector for each input string. Every vector has the
		same length, determined by the selected model.
	"""
	model = _get_model()

	# convert_to_numpy=False makes the result a list of Python lists instead
	# of a NumPy array, which is easier to inspect and store as JSON later.
	embeddings = model.encode(texts, convert_to_numpy=False)
	return [embedding.tolist() for embedding in embeddings]


if __name__ == "__main__":
	# These example sentences have related meanings, so their vectors can be
	# compared later to find relevant information during retrieval.
	example_texts = [
		"Employees receive 12 casual leaves per year.",
		"Employees must use the HR portal to apply for leave.",
	]

	embeddings = create_embeddings(example_texts)
	print(f"Number of embeddings: {len(embeddings)}")
	print(f"Length of one embedding: {len(embeddings[0])}")
