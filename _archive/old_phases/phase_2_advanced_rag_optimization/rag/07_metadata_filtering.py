"""
LESSON 07: Metadata Filtering
Area: RAG
Current recorded status: NOT STARTED

LEARNING TARGET
---------------
Understand what Metadata Filtering means, where it fits in an AI Engineer system,
and what you should be able to explain or demonstrate in an interview.

HOW TO STUDY
------------
1. Read the short explanation in this file.
2. Run the example with the project virtual environment.
3. Change the input and predict the output before running again.
4. Explain the result aloud in your own words.
5. Only then update your progress tracker to COMPLETED.

This lesson is a focused learning starter. Advanced services (cloud providers,
GPU serving, managed databases, or external APIs) may require extra setup.
Never paste API keys into source code.
"""

def demo():
    # Small transparent example for the Metadata Filtering idea.
    # Production implementations use a retrieval library or vector database.
    documents = [
        {"id": "D1", "text": "RAG retrieves relevant documents before answering."},
        {"id": "D2", "text": "Embeddings represent text as numeric vectors."},
        {"id": "D3", "text": "BM25 ranks documents using keyword statistics."},
    ]
    query = "How does retrieval find relevant documents?"
    query_terms = set(query.lower().split())
    scored = []
    for doc in documents:
        terms = set(doc["text"].lower().split())
        score = len(query_terms & terms)
        scored.append((score, doc["id"], doc["text"]))
    scored.sort(reverse=True)
    print("Query:", query)
    print("Illustrative ranked results:")
    for score, doc_id, text in scored:
        print(score, doc_id, text)
    print("\nStudy focus: explain how Metadata Filtering differs from this simplified baseline.")

if __name__ == "__main__":
    demo()
