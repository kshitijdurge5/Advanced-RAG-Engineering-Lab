"""
LESSON 20: Logging and Monitoring
Area: Backend
Current recorded status: NOT STARTED

LEARNING TARGET
---------------
Understand what Logging and Monitoring means, where it fits in an AI Engineer system,
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

from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    # Pydantic validates incoming data types and constraints.
    question: str = Field(min_length=3, max_length=500)

def demo():
    request = QuestionRequest(question="Explain retrieval augmented generation")
    print("Backend topic:", "Logging and Monitoring")
    print("Validated question:", request.question)
    print("Input validation succeeded.")

if __name__ == "__main__":
    demo()
