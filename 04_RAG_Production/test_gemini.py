
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

print("Connecting to Gemini 3.5 Flash-Lite...")

# Initialize the lighter Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0,
    timeout=60,
    max_retries=0
)

try:

    response = llm.invoke(
        "Reply with exactly: Gemini connection successful"
    )

    print("\nSUCCESS!")
    print(response.content)

except Exception as error:

    print("\nGEMINI CONNECTION FAILED")

    print(type(error).__name__)
    print(str(error))
