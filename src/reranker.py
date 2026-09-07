"""Rerank retrieved chunks using a Cross-Encoder."""

from functools import lru_cache

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_model():
    """Load the Cross-Encoder model once."""
    return CrossEncoder(MODEL_NAME)


def rerank_results(question, results, top_k=3):
    """Rerank retrieved chunks based on question-document relevance."""

    if not results:
        return []

    model = _get_model()

    pairs = [
        (question, result["text"])
        for result in results
    ]

    scores = model.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):
        updated_result = result.copy()
        updated_result["reranker_score"] = float(score)
        reranked.append(updated_result)

    reranked.sort(
        key=lambda result: result["reranker_score"],
        reverse=True,
    )

    return reranked[:top_k]


if __name__ == "__main__":
    question = "How many casual leaves do employees get?"

    example_results = [
        {
            "text": "Employees receive 12 casual leaves per year.",
            "metadata": {"page": 1},
            "score": 0.85,
        },
        {
            "text": "Employees must use strong passwords and enable two-factor authentication.",
            "metadata": {"page": 1},
            "score": 0.60,
        },
    ]

    results = rerank_results(
        question,
        example_results,
        top_k=2,
    )

    for number, result in enumerate(results, start=1):
        print(f"\nResult {number}")
        print(f"Reranker score: {result['reranker_score']:.4f}")
        print(f"Text: {result['text']}")