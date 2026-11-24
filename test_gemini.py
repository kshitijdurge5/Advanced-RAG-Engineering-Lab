import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load existing .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

print("Connecting to Gemini...")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0,
    timeout=30,
    max_retries=0
)

# Send a test question
response = llm.invoke(
    "Reply with exactly: Gemini connection successful"
)

print("\nGemini response:")
print(response.content)
