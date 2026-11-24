"""
LESSON 03: Chunking Strategies
Area: RAG
Current recorded status: COMPLETED

LEARNING TARGET
---------------
Understand what Chunking Strategies means, where it fits in an AI Engineer system,
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

def chunk_text(text, chunk_size=8, overlap=2):
    # This small word-based example demonstrates chunk size and overlap.
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks

if __name__ == "__main__":
    sample = " ".join(f"word{i}" for i in range(1, 22))
    for number, chunk in enumerate(chunk_text(sample), 1):
        print(f"Chunk {number}: {chunk}")
