
# ============================================================
# PROJECT 4: PRODUCTION RAG APPLICATION
# PDF -> CHUNKING -> EMBEDDINGS -> CHROMADB
# -> RETRIEVAL -> GEMINI -> CITATIONS -> EVALUATION -> FASTAPI
# ============================================================

# -------------------- IMPORT LIBRARIES --------------------

import os  # Reads environment variables.
import argparse  # Handles command-line options.
from pathlib import Path  # Handles file paths.

from dotenv import load_dotenv  # Loads API keys from .env.
from langchain_community.document_loaders import PyPDFLoader  # Loads PDF documents.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits documents into chunks.
from langchain_huggingface import HuggingFaceEmbeddings  # Converts text into vectors.
from langchain_chroma import Chroma  # Stores vectors in ChromaDB.
from langchain_google_genai import ChatGoogleGenerativeAI  # Connects to Gemini.

from fastapi import FastAPI, HTTPException  # Creates API endpoints.
from pydantic import BaseModel  # Validates API input.
import uvicorn  # Runs the API server.


# -------------------- PROJECT CONFIGURATION --------------------

ROOT = Path(__file__).resolve().parents[1]  # Finds the main project folder.

load_dotenv(ROOT / ".env")  # Loads environment variables from .env.

PDF_PATH = ROOT / "documents" / "TechVision_AI_RAG_Learning_Dataset.pdf"  # PDF file location.

CHROMA_PATH = ROOT / "chroma_db"  # ChromaDB storage folder.

COLLECTION_NAME = "production_rag_v1"  # Existing ChromaDB collection.

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Hugging Face embedding model.

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")  # Reads Gemini model from .env or uses default.

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Reads Gemini API key.

TOP_K = 4  # Retrieves four relevant document chunks.

vector_db = None  # Stores ChromaDB connection.

embedding_model = None  # Stores embedding model.


# ============================================================
# 1. LOAD PDF DOCUMENT
# ============================================================

def load_pdf():

    if not PDF_PATH.exists():  # Checks whether the PDF exists.

        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")  # Stops if PDF is missing.

    pages = PyPDFLoader(str(PDF_PATH)).load()  # Loads all PDF pages.

    print(f"PDF pages loaded: {len(pages)}")  # Displays page count.

    return pages  # Returns loaded pages.


# ============================================================
# 2. SPLIT PDF INTO CHUNKS
# ============================================================

def split_pdf(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Maximum chunk size is 500 characters.
        chunk_overlap=50  # Keeps 50 common characters between chunks.
    )

    chunks = splitter.split_documents(pages)  # Splits pages into chunks.

    print(f"Text chunks created: {len(chunks)}")  # Displays number of chunks.

    return chunks  # Returns chunks.


# ============================================================
# 3. CREATE OR REUSE VECTOR DATABASE
# ============================================================

def get_vector_database(chunks):

    global vector_db, embedding_model  # Makes these variables available globally.

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL  # Loads the embedding model.
    )

    vector_db = Chroma(
        persist_directory=str(CHROMA_PATH),  # Stores vectors permanently.
        embedding_function=embedding_model,  # Converts text into vectors.
        collection_name=COLLECTION_NAME  # Connects to existing collection.
    )

    existing_count = vector_db._collection.count()  # Counts existing vectors.

    if existing_count == 0:  # Checks whether the database is empty.

        vector_db.add_documents(chunks)  # Adds document chunks.

        print(f"Added {len(chunks)} chunks to ChromaDB.")  # Confirms insertion.

    else:

        print(f"Reusing existing ChromaDB data: {existing_count} chunks.")  # Avoids duplicate insertion.

    return vector_db  # Returns database connection.


# ============================================================
# 4. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(question):

    results = vector_db.similarity_search(
        question,  # Searches using the user's question.
        k=TOP_K  # Retrieves four matching chunks.
    )

    return results  # Returns relevant documents.


# ============================================================
# 5. GENERATE SOURCE CITATIONS
# ============================================================

def get_source_names(documents):

    sources = []  # Creates an empty citation list.

    for document in documents:  # Loops through retrieved documents.

        filename = Path(
            document.metadata.get("source", PDF_PATH.name)
        ).name  # Gets PDF filename.

        page_number = int(
            document.metadata.get("page", 0)
        ) + 1  # Converts zero-based page index to page number.

        source_name = f"{filename}, page {page_number}"  # Creates citation.

        if source_name not in sources:  # Checks for duplicate citations.

            sources.append(source_name)  # Adds citation to list.

    return sources  # Returns citations.


# ============================================================
# 6. GENERATE ANSWER USING GEMINI
# ============================================================

def generate_answer(question, documents):

    context_parts = []  # Stores retrieved text.

    for number, document in enumerate(documents, start=1):  # Numbers each retrieved chunk.

        page_number = int(
            document.metadata.get("page", 0)
        ) + 1  # Gets page number.

        context_parts.append(
            f"[Source {number} | page {page_number}] {document.page_content}"
        )  # Adds text with source label.

    context = "\n\n".join(context_parts)  # Combines retrieved chunks.

    if not GEMINI_API_KEY:  # Checks whether Gemini key exists.

        return "Gemini API key missing.\n\n" + context  # Returns retrieved context.

    try:

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,  # Selects Gemini model.
            google_api_key=GEMINI_API_KEY,  # Provides API key.
            timeout=45,  # Sets request timeout.
            max_retries=1  # Retries once after a temporary failure.
        )

        prompt = f"""
        Answer the question using only the information below.

        If the answer is not available, say that clearly.

        Include source labels such as [Source 1].

        CONTEXT:
        {context}

        QUESTION:
        {question}
        """  # Creates the RAG prompt.

        response = llm.invoke(prompt)  # Sends prompt to Gemini.

        content = response.content  # Extracts Gemini response content.

        if isinstance(content, list):  # Checks whether content is a list of blocks.

            text_parts = [
                item["text"]
                for item in content
                if isinstance(item, dict) and "text" in item
            ]  # Extracts only text blocks.

            return "\n".join(text_parts)  # Returns clean readable text.

        return content  # Returns content if it is already a string.

    except Exception as error:  # Handles Gemini errors.

        print(f"Gemini error: {type(error).__name__}")  # Displays error type.

        return (
            "Gemini is temporarily unavailable. "
            "Here is the retrieved information:\n\n" + context
        )  # Provides fallback information instead of crashing.


# ============================================================
# 7. COMPLETE RAG PIPELINE
# ============================================================

def ask_rag(question):

    documents = retrieve_documents(question)  # Retrieves relevant chunks.

    if not documents:  # Checks whether results were found.

        return {
            "question": question,
            "answer": "No relevant information found.",
            "sources": []
        }  # Returns message when nothing is retrieved.

    answer = generate_answer(question, documents)  # Generates answer.

    sources = get_source_names(documents)  # Generates citations.

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }  # Returns complete RAG response.


# ============================================================
# 8. RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval():

    test_questions = [
        "What is the purpose of TechVision AI?",
        "What services does the company provide?",
        "What security practices are described?"
    ]  # Creates sample test questions.

    print("\nRETRIEVAL EVALUATION")  # Displays heading.

    successful = 0  # Counts successful tests.

    for question in test_questions:  # Tests every question.

        results = retrieve_documents(question)  # Retrieves relevant chunks.

        found = len(results) > 0  # Checks whether results exist.

        successful += int(found)  # Adds one for successful retrieval.

        print(
            f"{'PASS' if found else 'FAIL'}: "
            f"{question} | chunks returned: {len(results)}"
        )  # Displays test result.

    print(
        f"\nRetrieval success: {successful}/{len(test_questions)}"
    )  # Displays total successful tests.

    print(
        "Note: This is basic retrieval evaluation, not a full RAGAS score."
    )  # Explains evaluation limitation.


# ============================================================
# 9. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="TechVision Production RAG API",
    version="1.0"
)  # Creates FastAPI application.


class QuestionRequest(BaseModel):  # Defines request data structure.

    question: str  # Requires a question as text.


# -------------------- HEALTH CHECK --------------------

@app.get("/health")  # Creates health-check endpoint.

def health_check():

    return {
        "status": "running",
        "project": "Production RAG"
    }  # Returns API status.


# -------------------- ASK QUESTION API --------------------

@app.post("/ask")  # Creates question-answer endpoint.

def ask_endpoint(request: QuestionRequest):

    if not request.question.strip():  # Checks whether question is empty.

        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )  # Returns bad request error.

    try:

        return ask_rag(request.question)  # Runs RAG pipeline.

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )  # Returns application error.


# ============================================================
# 10. MAIN APPLICATION
# ============================================================

def main():

    global vector_db  # Uses the global database connection.

    pages = load_pdf()  # Step 1: Loads PDF.

    chunks = split_pdf(pages)  # Step 2: Splits PDF.

    get_vector_database(chunks)  # Step 3: Creates or reuses ChromaDB.

    evaluate_retrieval()  # Step 4: Tests retrieval.

    parser = argparse.ArgumentParser()  # Creates command-line parser.

    parser.add_argument(
        "--api",
        action="store_true",
        help="Start FastAPI server"
    )  # Adds API mode option.

    args = parser.parse_args()  # Reads command-line arguments.

    if args.api:  # Checks whether API mode was selected.

        print("\nStarting FastAPI server...")  # Displays startup message.

        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000
        )  # Starts FastAPI server.

    else:

        print("\nRAG application is ready.")  # Displays ready message.

        print("Type 'exit' to stop.")  # Displays exit instruction.

        while True:  # Keeps accepting user questions.

            question = input("\nEnter your question: ").strip()  # Takes user input.

            if question.lower() == "exit":  # Checks whether user wants to exit.

                print("Application closed.")  # Displays exit message.

                break  # Stops the loop.

            if not question:  # Checks for empty input.

                print("Please enter a question.")  # Displays warning.

                continue  # Starts next loop.

            result = ask_rag(question)  # Runs complete RAG pipeline.

            print("\nANSWER:\n")  # Displays answer heading.

            print(result["answer"])  # Displays clean answer text.

            print("\nSOURCES:")  # Displays citations heading.

            for source in result["sources"]:  # Loops through citations.

                print("-", source)  # Displays each citation.


# ============================================================
# 11. RUN APPLICATION
# ============================================================

if __name__ == "__main__":  # Checks whether this file runs directly.

    main()  # Starts the application.
