"""
LESSON 06: Short-term Memory
Area: AI Agents
Current recorded status: NOT STARTED

LEARNING TARGET
---------------
Understand what Short-term Memory means, where it fits in an AI Engineer system,
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

def short_term_memory_demo():
    # An agent combines a model's decision with a controlled action.
    # This offline example demonstrates the control-flow idea without an API key.
    user_request = "Find the total of 12 and 30"
    available_tool = "add_numbers"
    print("User request:", user_request)
    print("Available tool:", available_tool)
    print("Agent decision: call the addition tool")
    result = 12 + 30
    print("Tool result:", result)
    print("Agent response: The total is", result)
    print("Now study the specific concept: Short-term Memory.")

if __name__ == "__main__":
    short_term_memory_demo()
