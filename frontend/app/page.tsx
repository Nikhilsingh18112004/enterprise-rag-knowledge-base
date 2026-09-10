"use client";

import { useState } from "react";

const exampleQuestions = [
  "How many casual leaves do employees get?",
  "How many days can employees work from home?",
  "What should employees use for sensitive systems?",
  "What benefits are provided to employees?",
];

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
    <main className="min-h-screen bg-slate-100">

      {/* Header */}
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">

          <div>
            <h1 className="text-xl font-bold text-slate-900">
              Enterprise Knowledge Assistant
            </h1>

            <p className="mt-1 text-sm text-slate-500">
              AI-powered enterprise document search
            </p>
          </div>

          <div className="rounded-full bg-blue-50 px-4 py-2 text-sm font-medium text-blue-700">
            RAG • Gemini
          </div>

        </div>
      </header>

      {/* Main Content */}
      <section className="px-6 py-12">
        <div className="mx-auto max-w-4xl">

          {/* Introduction */}
          <div className="mb-8 text-center">

            <h2 className="text-4xl font-bold tracking-tight text-slate-900">
              Ask your company documents
            </h2>

            <p className="mx-auto mt-4 max-w-2xl text-lg text-slate-600">
              Get accurate answers from your enterprise knowledge base using
              hybrid search, semantic retrieval, and AI-powered generation.
            </p>

          </div>

          {/* Question Card */}
          <div className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">

            <label className="mb-3 block text-sm font-semibold text-slate-700">
              Your question
            </label>

            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask something about company policies, benefits, security, or other documents..."
              className="h-36 w-full resize-none rounded-xl border border-slate-300 bg-slate-50 p-4 text-slate-900 outline-none transition focus:border-blue-500 focus:bg-white focus:ring-2 focus:ring-blue-100"
            />

            {/* Example Questions */}
            <div className="mt-6">

              <p className="mb-3 text-sm font-semibold text-slate-700">
                Try an example
              </p>

              <div className="flex flex-wrap gap-2">

                {exampleQuestions.map((example) => (
                  <button
                    key={example}
                    onClick={() => setQuestion(example)}
                    className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm text-slate-600 transition hover:border-blue-400 hover:bg-blue-50 hover:text-blue-700"
                  >
                    {example}
                  </button>
                ))}

              </div>

            </div>

            {/* Ask Button */}
            <button
              onClick={askQuestion}
              disabled={loading}
              className="mt-7 rounded-xl bg-blue-600 px-7 py-3 font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              {loading ? "Searching documents..." : "Ask Question"}
            </button>

          </div>

          {/* Answer */}
          {answer && (
            <div className="mt-8 rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">

              <div className="flex items-center gap-3">

                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-100 text-blue-700">
                  AI
                </div>

                <div>
                  <h3 className="font-semibold text-slate-900">
                    Answer
                  </h3>

                  <p className="text-sm text-slate-500">
                    Generated from your enterprise documents
                  </p>
                </div>

              </div>

              <div className="mt-6 border-t border-slate-100 pt-6">

                <p className="whitespace-pre-wrap leading-7 text-slate-700">
                  {answer}
                </p>

              </div>

            </div>
          )}

        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-6">

        <p className="text-center text-sm text-slate-500">
          Enterprise Knowledge Base • Retrieval-Augmented Generation
        </p>

      </footer>

    </main>
  );
}