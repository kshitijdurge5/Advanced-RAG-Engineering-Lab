"""
LESSON 02: Cosine Similarity
Area: RAG
Current recorded status: COMPLETED

LEARNING TARGET
---------------
Understand what Cosine Similarity means, where it fits in an AI Engineer system,
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

import math

def cosine_similarity(a, b):
    # Dot product measures alignment between two vectors.
    dot = sum(x * y for x, y in zip(a, b))
    # Vector lengths normalize the score.
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

if __name__ == "__main__":
    query_vector = [1, 2, 1]
    document_vector = [2, 1, 1]
    print("Cosine similarity:", round(cosine_similarity(query_vector, document_vector), 3))
