# ============================================================
# PROJECT 4: PRODUCTION RAG - FINAL SINGLE-FILE VERSION
# PDF -> CHUNKS -> EMBEDDINGS -> CHROMADB -> RETRIEVAL
# -> GEMINI -> CITATIONS -> EVALUATION -> FASTAPI
#
# Run normal RAG:
#   python .\04_RAG_Production\01_production_rag_app.py
#
# Run automated checks:
#   python .\04_RAG_Production\01_production_rag_app.py --check
#
# Run API:
#   python .\04_RAG_Production\01_production_rag_app.py --api
#
# Optional RAGAS evaluation:
#   python .\04_RAG_Production\01_production_rag_app.py --ragas
# ============================================================

import os  # Reads environment variables.
import argparse  # Reads options typed in PowerShell.
from pathlib import Path  # Builds reliable file paths.
from typing import Any  # Allows flexible response types.

from dotenv import load_dotenv  # Loads settings from the .env file.
from langchain_community.document_loaders import PyPDFLoader  # Loads PDF pages.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits pages into chunks.
from langchain_huggingface import HuggingFaceEmbeddings  # Creates text embeddings.
from langchain_chroma import Chroma  # Stores and searches vectors.
from langchain_google_genai import ChatGoogleGenerativeAI  # Connects to Gemini.

from fastapi import FastAPI, HTTPException  # Creates API endpoints and API errors.
from pydantic import BaseModel  # Validates incoming API data.
import uvicorn  # Runs the API server.


# -------------------- SETTINGS --------------------

ROOT = Path(__file__).resolve().parents[1]  # Finds the project root folder.
load_dotenv(ROOT / ".env")  # Loads the project's .env file.

PDF_PATH = ROOT / "documents" / "TechVision_AI_RAG_Learning_Dataset.pdf"  # Sets the source PDF.
CHROMA_PATH = ROOT / "chroma_db"  # Uses the existing persistent ChromaDB folder.
COLLECTION_NAME = "production_rag_v1"  # Uses the existing collection name.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Uses the existing embedding model.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")  # Reads model name or uses the default.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Reads the Gemini key without displaying it.
TOP_K = 4  # Retrieves four relevant chunks per question.

vector_db = None  # Holds the ChromaDB connection.
embedding_model = None  # Holds the embedding model.
llm = None  # Holds the Gemini model after it is initialized.


# -------------------- PDF LOADING --------------------

def load_pdf():  # Loads the source PDF.
    if not PDF_PATH.exists():  # Checks that the PDF file exists.
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")  # Stops with a clear error.
    pages = PyPDFLoader(str(PDF_PATH)).load()  # Loads the PDF into page documents.
    print(f"PDF pages loaded: {len(pages)}")  # Displays the number of pages.
    return pages  # Returns the page documents.


# -------------------- CHUNKING --------------------

def split_pdf(pages):  # Splits the PDF pages into smaller text pieces.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Sets the maximum chunk size.
        chunk_overlap=50  # Repeats some text to preserve context between chunks.
    )
    chunks = splitter.split_documents(pages)  # Creates chunks and keeps page metadata.
    print(f"Text chunks created: {len(chunks)}")  # Displays the number of chunks.
    return chunks  # Returns the chunks.


# -------------------- VECTOR DATABASE --------------------

def connect_vector_database(chunks):  # Creates embeddings and connects to ChromaDB.
    global vector_db, embedding_model  # Allows the function to set shared variables.
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL  # Loads the existing local embedding model.
    )
    vector_db = Chroma(
        persist_directory=str(CHROMA_PATH),  # Uses persistent local storage.
        embedding_function=embedding_model,  # Tells Chroma how to embed text.
        collection_name=COLLECTION_NAME  # Opens the existing collection.
    )
    current_count = vector_db._collection.count()  # Counts vectors already saved.
    if current_count == 0:  # Adds chunks only if the collection is empty.
        vector_db.add_documents(chunks)  # Embeds and stores the chunks.
        print(f"Stored {len(chunks)} chunks in ChromaDB.")  # Confirms storage.
    else:  # Reuses the collection if it already contains vectors.
        print(f"Reusing ChromaDB: {current_count} existing vectors.")  # Avoids duplicate insertion.
    return vector_db  # Returns the database connection.


# -------------------- RETRIEVAL --------------------

def retrieve_documents(question: str):  # Finds chunks relevant to a question.
    if vector_db is None:  # Checks whether the database has been initialized.
        raise RuntimeError("Vector database is not initialized.")  # Prevents search before setup.
    return vector_db.similarity_search(question, k=TOP_K)  # Returns the most similar chunks.


# -------------------- CITATIONS --------------------

def get_sources(documents):  # Creates readable citations from document metadata.
    sources = []  # Creates an empty citation list.
    for document in documents:  # Checks every retrieved chunk.
        filename = Path(document.metadata.get("source", PDF_PATH.name)).name  # Gets the PDF filename.
        page = int(document.metadata.get("page", 0)) + 1  # Converts page index to human page number.
        citation = f"{filename}, page {page}"  # Creates the citation text.
        if citation not in sources:  # Avoids duplicate page citations.
            sources.append(citation)  # Saves the unique citation.
    return sources  # Returns all unique citations.


# -------------------- GEMINI CONNECTION --------------------

def get_gemini():  # Creates the Gemini model only when an answer is requested.
    global llm  # Allows this function to reuse the same model object.
    if llm is not None:  # Checks whether Gemini was already initialized.
        return llm  # Reuses the existing Gemini object.
    if not GEMINI_API_KEY:  # Checks whether the API key is available.
        raise ValueError("GEMINI_API_KEY is missing from the project .env file.")  # Gives a clear setup error.
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,  # Selects the Gemini model.
        google_api_key=GEMINI_API_KEY,  # Supplies the API key.
        timeout=60,  # Waits up to 60 seconds for a response.
        max_retries=2  # Retries temporary API failures twice.
    )
    return llm  # Returns the Gemini connection.


# -------------------- ANSWER GENERATION --------------------

def generate_answer(question: str, documents):  # Uses Gemini to answer from retrieved context.
    context_blocks = []  # Stores text with source labels.
    for index, document in enumerate(documents, start=1):  # Numbers each retrieved chunk.
        page = int(document.metadata.get("page", 0)) + 1  # Reads the PDF page number.
        context_blocks.append(
            f"[Source {index} | page {page}]\n{document.page_content}"
        )  # Adds the chunk and its source label.
    context = "\n\n".join(context_blocks)  # Combines all chunks into one context.
    prompt = f"""
You are a document question-answering assistant.
Answer using only the CONTEXT below.
If the answer is not in the context, say that the document does not provide it.
Cite supporting chunks using labels such as [Source 1 | page 2].
Do not invent facts.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""  # Creates a grounded RAG prompt.
    response = get_gemini().invoke(prompt)  # Sends the prompt to Gemini.
    content: Any = response.content  # Gets the content returned by Gemini.
    if isinstance(content, str):  # Checks whether Gemini returned plain text.
        return content  # Returns the text directly.
    if isinstance(content, list):  # Checks whether Gemini returned content blocks.
        text_parts = []  # Creates a list for text blocks only.
        for item in content:  # Checks each content block.
            if isinstance(item, dict) and isinstance(item.get("text"), str):  # Keeps only dictionary text blocks.
                text_parts.append(item["text"])  # Adds readable text and ignores signatures or other metadata.
        return "\n".join(text_parts)  # Returns clean readable text.
    return str(content)  # Converts any other response type into readable text.


# -------------------- COMPLETE RAG FUNCTION --------------------

def ask_rag(question: str):  # Runs retrieval, generation and citation formatting.
    if not question.strip():  # Checks for an empty question.
        raise ValueError("Please enter a question.")  # Rejects empty input.
    documents = retrieve_documents(question)  # Retrieves relevant document chunks.
    if not documents:  # Checks whether retrieval found anything.
        return {
            "question": question,
            "answer": "No relevant information was found in the PDF.",
            "sources": []
        }  # Returns a safe no-results response.
    sources = get_sources(documents)  # Creates page citations.
    try:  # Handles Gemini service or connection errors.
        answer = generate_answer(question, documents)  # Generates the answer.
        status = "generated_by_gemini"  # Records that Gemini generated the answer.
    except Exception as error:  # Catches errors such as temporary 503 or 504 responses.
        print(f"Gemini generation failed: {type(error).__name__}: {error}")  # Shows the error for troubleshooting.
        answer = "Gemini is temporarily unavailable. Retrieved document excerpts:\n\n" + "\n\n".join(
            document.page_content for document in documents
        )  # Returns retrieved text instead of crashing.
        status = "retrieval_fallback"  # Records that fallback text was returned.
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "status": status
    }  # Returns answer, citations and generation status.


# -------------------- BASIC RETRIEVAL EVALUATION --------------------

def evaluate_retrieval():  # Runs simple checks that the retriever returns results.
    questions = [
        "What is the purpose of TechVision AI?",
        "What services does the company provide?",
        "What security practices are described?"
    ]  # Defines three questions based on the learning PDF.
    print("\nBASIC RETRIEVAL EVALUATION")  # Displays the evaluation heading.
    passed = 0  # Counts successful retrieval checks.
    for question in questions:  # Tests each question.
        results = retrieve_documents(question)  # Retrieves relevant chunks.
        success = len(results) > 0  # Checks whether at least one chunk was found.
        passed += int(success)  # Adds one for a successful check.
        print(f"{'PASS' if success else 'FAIL'} | {question} | chunks: {len(results)}")  # Prints the test result.
    print(f"Retrieval checks passed: {passed}/{len(questions)}")  # Displays the total.
    return passed == len(questions)  # Returns True only when all checks pass.


# -------------------- OPTIONAL RAGAS EVALUATION --------------------

def run_ragas_evaluation():  # Runs RAGAS if it has been installed and configured.
    try:  # Checks whether the optional evaluation packages are available.
        from datasets import Dataset  # Creates the evaluation dataset.
        from ragas import evaluate  # Runs RAGAS evaluation.
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall  # Imports common RAG metrics.
    except ImportError as error:  # Handles missing optional packages.
        print("RAGAS dependencies are not installed.")  # Explains why evaluation cannot run.
        print("Install them with: pip install ragas datasets")  # Gives the installation command.
        print(f"Details: {error}")  # Shows the missing dependency.
        return  # Stops this optional evaluation without stopping the RAG app.

    evaluation_questions = [  # Defines questions for the evaluation dataset.
        "What is the purpose of TechVision AI?",
        "What services does the company provide?",
        "What security practices are described?"
    ]
    rows = []  # Stores question, answer, context and reference data.
    print("\nPreparing RAGAS evaluation examples...")  # Displays progress.
    for question in evaluation_questions:  # Builds one evaluation row per question.
        documents = retrieve_documents(question)  # Retrieves context for the question.
        if not documents:  # Skips questions with no retrieved context.
            continue  # Moves to the next question.
        context_list = [document.page_content for document in documents]  # Converts retrieved documents into context strings.
        try:  # Generates an answer for this evaluation question.
            answer = generate_answer(question, documents)  # Calls Gemini.
        except Exception as error:  # Handles generation failures.
            print(f"Skipping RAGAS row because Gemini failed: {error}")  # Explains why the row is skipped.
            continue  # Moves to the next question.
        rows.append({
            "question": question,
            "answer": answer,
            "contexts": context_list,
            "ground_truth": "Use the supplied TechVision AI PDF to answer this question accurately."
        })  # Adds one row to the evaluation dataset.

    if not rows:  # Checks whether any evaluation examples were created.
        print("No RAGAS evaluation rows were created.")  # Explains the issue.
        return  # Stops evaluation.

    dataset = Dataset.from_list(rows)  # Converts rows into a Hugging Face Dataset.
    print("\nRunning RAGAS metrics...")  # Displays evaluation progress.
    try:  # Runs RAGAS metrics.
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )  # Measures faithfulness, relevance and context quality.
        print("\nRAGAS RESULTS")  # Displays the results heading.
        print(results)  # Prints the metric scores.
    except Exception as error:  # Handles RAGAS version or model integration errors.
        print("RAGAS could not complete with the installed package configuration.")  # Explains the failure.
        print(f"Details: {type(error).__name__}: {error}")  # Prints the technical reason.
        print("The normal RAG application and retrieval checks are still available.")  # Clarifies that the app remains usable.


# -------------------- FASTAPI --------------------

app = FastAPI(
    title="TechVision Production RAG API",
    version="1.0",
    description="Ask questions about the TechVision AI PDF and receive cited answers."
)  # Creates the FastAPI application.


class QuestionRequest(BaseModel):  # Defines the structure of a question request.
    question: str  # Requires a question string.


@app.get("/health")  # Creates the API health-check route.
def health_check():  # Returns the API status.
    return {
        "status": "running",
        "project": "Production RAG"
    }  # Confirms that the server is running.


@app.post("/ask")  # Creates the RAG question endpoint.
def ask_endpoint(request: QuestionRequest):  # Receives a validated question.
    try:  # Handles errors from the RAG application.
        return ask_rag(request.question)  # Returns answer, sources and status.
    except ValueError as error:  # Handles empty or invalid questions.
        raise HTTPException(status_code=400, detail=str(error))  # Returns a client error.
    except Exception as error:  # Handles unexpected server errors.
        raise HTTPException(status_code=500, detail=str(error))  # Returns a server error.


# -------------------- AUTOMATED CHECKS --------------------

def run_checks():  # Runs the final local project checks.
    print("\nPROJECT 4 FINAL CHECKS")  # Displays the checks heading.
    print(f"PDF exists: {PDF_PATH.exists()}")  # Checks the PDF path.
    print(f"Environment file exists: {(ROOT / '.env').exists()}")  # Checks the .env file path.
    print(f"ChromaDB folder exists: {CHROMA_PATH.exists()}")  # Checks persistent storage.
    count = vector_db._collection.count()  # Counts vectors in the collection.
    print(f"ChromaDB vector count: {count}")  # Displays the vector count.
    retrieval_passed = evaluate_retrieval()  # Runs the retrieval checks.
    print(f"Final retrieval check: {'PASS' if retrieval_passed else 'CHECK FAILED'}")  # Displays final retrieval status.
    print("API routes available: GET /health and POST /ask")  # Lists available endpoints.
    print("OpenAPI documentation: http://127.0.0.1:8000/docs")  # Shows the interactive API documentation URL.
    print("Note: API routes are listed here; start with --api to test them in a browser.")  # Explains that the API server must be started separately.


# -------------------- INTERACTIVE MODE --------------------

def run_chat():  # Starts the terminal question-answer loop.
    print("\nRAG application is ready. Type 'exit' to stop.")  # Displays instructions.
    while True:  # Continues until the user exits.
        question = input("\nEnter your question: ").strip()  # Reads a question.
        if question.lower() == "exit":  # Checks for the exit command.
            print("RAG application closed.")  # Displays the closing message.
            break  # Stops the loop.
        if not question:  # Checks for blank input.
            print("Please enter a question.")  # Asks for a question.
            continue  # Starts the next loop.
        result = ask_rag(question)  # Runs the RAG pipeline.
        print("\nANSWER:\n")  # Displays the answer heading.
        print(result["answer"])  # Displays clean answer text.
        print("\nSOURCES:")  # Displays the source heading.
        for source in result["sources"]:  # Loops through citations.
            print("-", source)  # Prints one citation.
        print(f"\nSTATUS: {result['status']}")  # Shows Gemini or fallback status.


# -------------------- MAIN PROGRAM --------------------

def main():  # Starts the selected application mode.
    pages = load_pdf()  # Loads the PDF pages.
    chunks = split_pdf(pages)  # Splits the PDF into chunks.
    connect_vector_database(chunks)  # Connects to the persistent vector database.

    parser = argparse.ArgumentParser(description="Production RAG application")  # Creates the command-line parser.
    parser.add_argument("--api", action="store_true", help="Start the FastAPI server")  # Adds API mode.
    parser.add_argument("--check", action="store_true", help="Run final project checks")  # Adds check mode.
    parser.add_argument("--ragas", action="store_true", help="Run optional RAGAS evaluation")  # Adds RAGAS mode.
    args = parser.parse_args()  # Reads the command-line options.

    if args.api:  # Checks whether API mode was requested.
        print("\nStarting API at http://127.0.0.1:8000")  # Displays server address.
        uvicorn.run(app, host="127.0.0.1", port=8000)  # Starts the API server.
    elif args.check:  # Checks whether final check mode was requested.
        run_checks()  # Runs project checks.
    elif args.ragas:  # Checks whether RAGAS mode was requested.
        run_ragas_evaluation()  # Runs optional RAGAS evaluation.
    else:  # Runs normal interactive mode.
        evaluate_retrieval()  # Checks retrieval before accepting questions.
        run_chat()  # Starts question-answer mode.


if __name__ == "__main__":  # Checks whether this file was run directly.
    main()  # Starts the application.
