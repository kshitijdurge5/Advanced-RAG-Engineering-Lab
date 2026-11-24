"""
LESSON 01: Tokenization
Area: LLM
Current recorded status: NOT STARTED

LEARNING TARGET
---------------
Understand what Tokenization means, where it fits in an AI Engineer system,
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
    text = "RAG helps an AI answer using documents."
    # A simple whitespace split is NOT a real model tokenizer.
    approximate_tokens = text.split()
    print("Text:", text)
    print("Approximate word pieces:", approximate_tokens)
    print("Approximate count:", len(approximate_tokens))
    print("Real LLM tokenizers may split words differently.")

if __name__ == "__main__":
    demo()
