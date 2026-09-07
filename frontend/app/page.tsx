"use client";

import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  async function askQuestion() {
    if (!question.trim()) {
      return;
    }

    setLoading(true);
    setAnswer("");

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get answer");
      }

      const data = await response.json();

      setAnswer(data.answer);
    } catch (error) {
      console.error(error);
      setAnswer(
        "Unable to connect to the RAG backend. Please make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-100 px-6 py-12">
      <div className="mx-auto max-w-3xl">
        <div className="rounded-2xl bg-white p-8 shadow-lg">
          <h1 className="text-3xl font-bold text-gray-900">
            Enterprise Knowledge Assistant
          </h1>

          <p className="mt-2 text-gray-600">
            Ask questions about the company documents.
          </p>

          <div className="mt-8">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question..."
              className="h-32 w-full rounded-lg border border-gray-300 p-4 text-gray-900 outline-none focus:border-blue-500"
            />
          </div>

          <button
            onClick={askQuestion}
            disabled={loading}
            className="mt-4 rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-400"
          >
            {loading ? "Thinking..." : "Ask Question"}
          </button>

          {answer && (
            <div className="mt-8 rounded-lg border border-gray-200 bg-gray-50 p-6">
              <h2 className="text-lg font-semibold text-gray-900">
                Answer
              </h2>

              <p className="mt-3 whitespace-pre-wrap text-gray-700">
                {answer}
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}