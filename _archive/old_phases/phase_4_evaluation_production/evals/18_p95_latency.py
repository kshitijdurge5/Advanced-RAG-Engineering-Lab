"""
LESSON 18: P95 Latency
Area: Evals
Current recorded status: NOT STARTED

LEARNING TARGET
---------------
Understand what P95 Latency means, where it fits in an AI Engineer system,
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

def evaluate_example():
    # A tiny evaluation record. Real evaluation needs labelled test examples.
    test_case = {
        "question": "What does RAG do?",
        "expected": "It retrieves relevant context and uses it to answer.",
        "actual": "RAG retrieves context and grounds the answer.",
    }
    # This is only a placeholder similarity-free check, not a real LLM metric.
    passed = "retrieves" in test_case["actual"].lower()
    print("Evaluation concept:", "P95 Latency")
    print("Question:", test_case["question"])
    print("Expected:", test_case["expected"])
    print("Actual:", test_case["actual"])
    print("Illustrative check:", "PASS" if passed else "REVIEW")
    print("Important: build a proper dataset and metric for real evaluation.")

if __name__ == "__main__":
    evaluate_example()
