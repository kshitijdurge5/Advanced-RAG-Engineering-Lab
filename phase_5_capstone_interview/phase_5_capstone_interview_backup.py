"""
PHASE 5: CAPSTONE - AGENTIC RAG AI ASSISTANT
============================================

BEGINNER GUIDE:

This project combines concepts from the previous phases.

Architecture:

Documents
    |
    v
Text extraction
    |
    v
Chunking
    |
    v
Keyword retrieval
    |
    v
Tool-enabled agent
    |
    v
Answer with source references

This is a runnable educational capstone.
It uses local documents and does not require a paid API.

To use your own documents:
Place TXT files inside the existing documents folder.

Run:
python phase_5_capstone_interview.py

Run tests:
python phase_5_capstone_interview.py --test
"""

import re
import sys
import unittest
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = ROOT / "documents"


# ------------------------------------------------------------
# 1. LOAD DOCUMENTS
# ------------------------------------------------------------

def load_documents():

    DOCUMENTS.mkdir(exist_ok=True)

    documents = []

    for file in DOCUMENTS.rglob("*.txt"):

        try:
            text = file.read_text(encoding="utf-8")

            if text.strip():
                documents.append({
                    "source": str(file.relative_to(ROOT)),
                    "content": text
                })

        except (OSError, UnicodeError) as error:
            print("Could not read:", file.name, error)

    if not documents:

        documents.append({
            "source": "built_in_learning_document.txt",
            "content": """
            Retrieval Augmented Generation combines document retrieval
            with language model generation.

            AI agents can call tools to complete tasks.

            ChromaDB is a vector database used to store embeddings.

            LangGraph supports stateful agent workflows.

            MCP connects AI applications with external tools and context.

            BM25 is a keyword retrieval algorithm.

            FastAPI is a Python framework for building APIs.
            """
        })

    return documents


# ------------------------------------------------------------
# 2. CHUNK DOCUMENTS
# ------------------------------------------------------------

def chunk_documents(documents, chunk_size=100, overlap=20):

    chunks = []

    for document in documents:

        words = document["content"].split()

        for start in range(0, len(words), chunk_size - overlap):

            content = words[start:start + chunk_size]

            if not content:
                break

            chunks.append({
                "content": " ".join(content),
                "source": document["source"]
            })

            if start + chunk_size >= len(words):
                break

    return chunks


# ------------------------------------------------------------
# 3. RETRIEVAL
# ------------------------------------------------------------

def retrieve(question, chunks, top_k=3):

    query_words = set(
        re.findall(r"\w+", question.lower())
    )

    results = []

    for chunk in chunks:

        words = re.findall(
            r"\w+",
            chunk["content"].lower()
        )

        frequencies = Counter(words)

        score = sum(
            frequencies[word]
            for word in query_words
        )

        if score > 0:

            results.append({
                **chunk,
                "score": score
            })

    return sorted(
        results,
        key=lambda item: item["score"],
        reverse=True
    )[:top_k]


# ------------------------------------------------------------
# 4. TOOL CALLING
# ------------------------------------------------------------

def search_documents(question, chunks):

    results = retrieve(question, chunks)

    if not results:
        return "No relevant documents found."

    output = []

    for result in results:

        output.append(
            "SOURCE: " + result["source"] +
            "\nCONTENT: " + result["content"]
        )

    return "\n\n".join(output)


# ------------------------------------------------------------
# 5. AGENT
# ------------------------------------------------------------

class RAGAgent:

    def __init__(self, chunks):
        self.chunks = chunks
        self.memory = []

    def run(self, question):

        print("\nAGENT: Understanding your question...")

        print("AGENT: Calling document retrieval tool...")

        context = search_documents(
            question,
            self.chunks
        )

        self.memory.append({
            "question": question,
            "context": context
        })

        if context == "No relevant documents found.":
            answer = (
                "I could not find relevant information "
                "in the available documents."
            )

        else:
            answer = (
                "Retrieved information:\n\n" + context
            )

        return answer


# ------------------------------------------------------------
# 6. EVALUATION
# ------------------------------------------------------------

def evaluate_retrieval(chunks):

    test_question = "What is BM25?"

    results = retrieve(
        test_question,
        chunks,
        top_k=3
    )

    return {
        "test_question": test_question,
        "retrieval_results": len(results),
        "passed": len(results) > 0
    }


# ------------------------------------------------------------
# 7. SMOKE TESTS
# ------------------------------------------------------------
# A smoke test checks whether the main application works at a basic level.

class CapstoneTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.documents = load_documents()
        cls.chunks = chunk_documents(cls.documents)
        cls.agent = RAGAgent(cls.chunks)

    def test_documents_loaded(self):
        self.assertGreater(len(self.documents), 0)

    def test_chunks_created(self):
        self.assertGreater(len(self.chunks), 0)

    def test_retrieval(self):
        results = retrieve(
            "What is BM25?",
            self.chunks
        )
        self.assertGreater(len(results), 0)

    def test_agent_response(self):
        answer = self.agent.run("What is BM25?")
        self.assertTrue(answer)

    def test_evaluation(self):
        result = evaluate_retrieval(self.chunks)
        self.assertTrue(result["passed"])


# ------------------------------------------------------------
# 8. MAIN APPLICATION
# ------------------------------------------------------------

def main():

    print("=" * 55)
    print("AGENTIC RAG CAPSTONE")
    print("=" * 55)

    documents = load_documents()

    chunks = chunk_documents(documents)

    print("Documents loaded:", len(documents))
    print("Chunks created:", len(chunks))

    agent = RAGAgent(chunks)

    print("\nAsk a question about your documents.")
    print("Type exit to stop.")

    while True:

        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        answer = agent.run(question)

        print("\nFINAL ANSWER")
        print(answer)

    print("\nCAPSTONE SESSION COMPLETED")


if __name__ == "__main__":

    if "--test" in sys.argv:

        unittest.main(argv=[sys.argv[0]])

    else:

        main()


# INTERVIEW QUESTIONS
#
# 1. Explain your complete RAG architecture.
# 2. Why do you use chunking?
# 3. How does retrieval work?
# 4. What is the difference between RAG and fine-tuning?
# 5. How would you improve retrieval quality?
# 6. How would you introduce Gemini into this application?
# 7. How would you implement tool calling?
# 8. How would you evaluate hallucinations?
# 9. How would you secure this application?
# 10. How would you deploy it using FastAPI and Docker?
#
# IMPORTANT:
# This is a local educational capstone.
# It does not yet include a live LLM, production authentication,
# vector database or deployed multi-agent orchestration.
# Those are advanced extensions to implement and test.