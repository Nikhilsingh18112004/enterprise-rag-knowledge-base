\# Enterprise Knowledge Base with Retrieval-Augmented Generation (RAG)



An AI-powered enterprise knowledge assistant that answers questions from internal company documents using Retrieval-Augmented Generation (RAG).



The system combines semantic search, keyword search, hybrid retrieval, Cross-Encoder reranking, confidence-based abstention, and Gemini-powered answer generation. Answers are accompanied by document and page-level sources.



\---



\## 📌 Project Information



| Field | Details |

|---|---|

| Project | Enterprise Knowledge Base with RAG |

| Project Type | Individual Project |

| Institution | Parul University |

| Student | Nikhil Singh Ravindra |

| Technology Area | Artificial Intelligence / RAG / NLP |



\---



\## 🎯 Problem Statement



Enterprise organizations store important information across documents such as employee handbooks, security policies, and other internal knowledge sources.



Traditional keyword-based document searching can make it difficult to quickly find relevant information and understand it in context.



This project develops an AI-powered enterprise knowledge assistant that retrieves relevant information from enterprise documents and generates concise answers grounded only in the retrieved content.



\---



\## 🎯 Objectives



The main objectives of this project are to:



\- Ingest enterprise PDF documents.

\- Extract and intelligently chunk document content.

\- Generate semantic embeddings for document chunks.

\- Perform semantic vector search.

\- Perform keyword-based BM25 search.

\- Combine both approaches using hybrid retrieval.

\- Rerank retrieved results using a Cross-Encoder.

\- Generate answers using Gemini.

\- Prevent unsupported answers using confidence-based abstention.

\- Provide document and page-level sources.

\- Evaluate retrieval performance.

\- Provide a web-based user interface.

\- Expose the RAG system through a FastAPI backend.



\---



\# 🏗️ System Architecture



```text

&#x20;                   ┌─────────────────────────┐

&#x20;                   │     Enterprise PDFs     │

&#x20;                   └────────────┬────────────┘

&#x20;                                │

&#x20;                                ▼

&#x20;                   ┌─────────────────────────┐

&#x20;                   │      PDF Extraction     │

&#x20;                   │         pypdf           │

&#x20;                   └────────────┬────────────┘

&#x20;                                │

&#x20;                                ▼

&#x20;                   ┌─────────────────────────┐

&#x20;                   │   Section-aware Chunking│

&#x20;                   └────────────┬────────────┘

&#x20;                                │

&#x20;                                ▼

&#x20;                   ┌─────────────────────────┐

&#x20;                   │     Embedding Model     │

&#x20;                   │ BAAI/bge-small-en-v1.5  │

&#x20;                   └────────────┬────────────┘

&#x20;                                │

&#x20;                   ┌────────────┴────────────┐

&#x20;                   ▼                         ▼

&#x20;         ┌──────────────────┐      ┌──────────────────┐

&#x20;         │   FAISS Search   │      │   BM25 Search    │

&#x20;         │ Semantic Search  │      │ Keyword Search   │

&#x20;         └────────┬─────────┘      └────────┬─────────┘

&#x20;                  │                         │

&#x20;                  └───────────┬─────────────┘

&#x20;                              ▼

&#x20;                   ┌─────────────────────────┐

&#x20;                   │    Hybrid Retrieval     │

&#x20;                   │  Vector + BM25 Ranking  │

&#x20;                   └────────────┬────────────┘

&#x20;                                │

&#x20;                                ▼

&#x20;                   ┌─────────────────────────┐

&#x20;                   │   Cross-Encoder         │

&#x20;                   │       Reranking         │

&#x20;                   └────────────┬────────────┘

&#x20;                                │

&#x20;                                ▼

&#x20;                   ┌─────────────────────────┐

&#x20;                   │ Confidence Check /      │

&#x20;                   │      Abstention         │

&#x20;                   └────────────┬────────────┘

&#x20;                                │

&#x20;                   ┌────────────┴────────────┐

&#x20;                   │                         │

&#x20;                Confident              Low confidence

&#x20;                   │                         │

&#x20;                   ▼                         ▼

&#x20;         ┌──────────────────┐      ┌────────────────────┐

&#x20;         │ Gemini +         │      │ Information not    │

&#x20;         │ LangChain        │      │ available response │

&#x20;         └────────┬─────────┘      └────────────────────┘

&#x20;                  │

&#x20;                  ▼

&#x20;         ┌──────────────────┐

&#x20;         │ Answer + Sources │

&#x20;         └────────┬─────────┘

&#x20;                  │

&#x20;                  ▼

&#x20;         ┌──────────────────┐

&#x20;         │   FastAPI API    │

&#x20;         └────────┬─────────┘

&#x20;                  │

&#x20;                  ▼

&#x20;         ┌──────────────────┐

&#x20;         │   Next.js UI     │

&#x20;         └──────────────────┘

