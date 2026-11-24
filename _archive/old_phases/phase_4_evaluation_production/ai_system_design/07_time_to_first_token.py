"""
LESSON 07: Time to First Token
Area: AI System Design
Current recorded status: NOT STARTED

LEARNING TARGET
---------------
Understand what Time to First Token means, where it fits in an AI Engineer system,
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

def choose_model(task_complexity, budget):
    # A tiny rule-based example to introduce system trade-offs.
    if task_complexity == "high" and budget == "flexible":
        return "higher-capability-model"
    return "lower-cost-model"

if __name__ == "__main__":
    print("System design topic: Time to First Token")
    print("Selected route:", choose_model("high", "flexible"))
    print("Discuss quality, latency, reliability and cost before choosing a design.")
