"""Generate answers using LangChain and Gemini."""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file"
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GEMINI_API_KEY,
    
)


prompt = ChatPromptTemplate.from_template(
    """You answer questions about enterprise documents.

Answer the user's question ONLY using the provided context.

If the context does not contain enough information, say exactly:

"The information is not available in the provided documents."

Keep the answer concise.

Include the relevant document filename and page number.

Do not use outside knowledge or make up details.

User question:
{question}

Provided context:
{context}
"""
)


def generate_answer(question, retrieved_chunks):
    """Generate an answer using LangChain and Gemini."""

    context_parts = []

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = chunk.get("metadata", {})

        source_name = metadata.get(
            "source",
            "unknown",
        )

        page_number = metadata.get(
            "page",
            "unknown",
        )

        context_parts.append(
            f"Source {number} "
            f"(document: {source_name}, "
            f"page: {page_number}):\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    if not context:
        context = "No context was retrieved."

    messages = prompt.format_messages(
        question=question,
        context=context,
    )

    response = llm.invoke(messages)

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))

        return "".join(text_parts)

    return str(content)

    


if __name__ == "__main__":
    question = (
        "How many casual leaves do employees get?"
    )

    retrieved_chunks = [
        {
            "text": (
                "Employees are entitled to "
                "12 casual leaves per year."
            ),
            "metadata": {
                "page": 1,
                "source": "employee_handbook.pdf",
            },
        }
    ]

    answer = generate_answer(
        question,
        retrieved_chunks,
    )

    print("\nAnswer:")
    print(answer)