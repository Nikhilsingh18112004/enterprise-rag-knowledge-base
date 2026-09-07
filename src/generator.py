"""Generate answers from retrieved document chunks with Gemini."""

import os

from dotenv import load_dotenv
from google import genai


# Read variables from the project's .env file.
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in the .env file")

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(question, retrieved_chunks):
    """Generate a concise answer using retrieved document chunks."""

    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):

        metadata = chunk.get("metadata", {})

        page_number = metadata.get("page", "unknown")
        source_name = metadata.get("source", "unknown")

        context_parts.append(
            f"Source {number} "
            f"(document: {source_name}, page: {page_number}):\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    if not context:
        context = "No context was retrieved."

    prompt = f"""You answer questions about documents.

Answer the user's question ONLY using the provided context.

If the context does not contain enough information, say exactly:
"The information is not available in the provided documents."

Keep the answer concise.

Include the relevant document filename and page number in your answer.

Do not use outside knowledge or make up details.

User question:
{question}

Provided context:
{context}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text


if __name__ == "__main__":

    question = "How many casual leaves do employees get?"

    retrieved_chunks = [
        {
            "text": "Employees are entitled to 12 casual leaves per year.",
            "metadata": {
                "page": 1,
                "source": "employee_handbook.pdf",
            },
            "score": 0.7811,
        }
    ]

    answer = generate_answer(
        question,
        retrieved_chunks,
    )

    print("\nAnswer:")
    print(answer)