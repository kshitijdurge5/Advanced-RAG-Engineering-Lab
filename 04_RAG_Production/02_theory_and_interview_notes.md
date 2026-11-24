# Project 4 — Production RAG Engineering

**Code:** `01_production_rag_app.py`  
**Status:** In progress. The ingestion pipeline is completed. Retrieval, Gemini generation, citations, API development, and evaluation are planned next.

> This guide separates implemented features from planned features. Do not describe planned features as completed in an interview.

## 1. What does production-oriented RAG mean?

A learning script may load a document, retrieve a passage, and print a response. A production-oriented application also needs reliable ingestion, persistent storage, retrieval quality, grounded generation, source traceability, API access, validation, error handling, evaluation, security, and monitoring.

Using FastAPI or a persistent vector store alone does not automatically make an application production-ready.

## 2. Current implementation

The current file implements the ingestion foundation:

1. Load environment variables.
2. Construct paths relative to the Python file.
3. Load the TechVision AI PDF.
4. Split pages into chunks.
5. Initialize a Hugging Face embedding model.
6. Connect to a persistent Chroma collection.
7. Insert chunks if the collection is empty.
8. Skip insertion if the collection already contains IDs.

The successful run printed `DOCUMENT INGESTION PIPELINE COMPLETED!`.

## 3. Libraries and components currently used

### `os`

Python's `os` module provides operating-system functions.

- `os.path.abspath(__file__)` resolves the script path.
- `os.path.dirname(...)` gets the containing directory.
- `os.path.join(...)` joins path components.
- `os.path.exists(...)` checks whether a path exists.

Resolving paths relative to the script makes the project less dependent on the terminal's current working directory.

### `python-dotenv`

`load_dotenv()` reads variables from a local `.env` file and makes them available through the process environment.

The Gemini API key is stored locally as `GEMINI_API_KEY`. Never commit the real key or paste it into chat.

### `PyPDFLoader`

Loads the PDF and returns LangChain Document objects, commonly one per page, with source/page metadata.

### `RecursiveCharacterTextSplitter`

Splits loaded documents using:
- `chunk_size=500`
- `chunk_overlap=50`

These are project settings, not universal best values. Evaluate them against the document type and retrieval questions.

### `HuggingFaceEmbeddings`

The project uses `sentence-transformers/all-MiniLM-L6-v2`. It creates 384-dimensional embeddings. Use a compatible model and configuration for both document indexing and query embedding.

### `langchain_chroma.Chroma`

Connects LangChain to Chroma for vector storage and retrieval.

Current collection: `production_rag_v1`  
Persistent directory: `chroma_db`

The new collection is separate from Project 3's `techvision_rag`, so the project does not intentionally mix the two collections.

## 4. Persistent storage and duplicate prevention

Persistent storage saves vectors so they can be reused after the Python process exits.

The project configures a collection name, an embedding function, and a persistence directory.

The current duplicate-prevention check reads existing IDs:
- If the collection is empty, it adds the chunks.
- If it already contains IDs, it skips insertion.

**Limitation:** This simple check does not detect whether the source PDF changed. A robust ingestion system should track stable document/chunk IDs and source versions or hashes, and support updates and deletions.

## 5. Current execution flow

```text
Python script starts
       |
Load .env variables
       |
Resolve PDF and Chroma paths
       |
Check that PDF exists
       |
PyPDFLoader reads PDF pages
       |
Recursive splitter creates chunks
       |
Hugging Face model is initialized
       |
Connect to production_rag_v1
       |
Check existing collection IDs
       |
Insert chunks if collection is empty
       |
Print completion summary
```

## 6. Planned features — not yet completed

### A. Retrieval

A retriever will accept a user question and find relevant chunks in Chroma. The query is embedded using the same embedding model used for indexing. Chroma searches the stored vectors and returns matching documents and metadata.

Design choices include result count (`k`), metadata filters, similarity thresholds where supported, and retrieval evaluation.

### B. Gemini answer generation

A chat model will receive the user's question, retrieved context, and prompt instructions. It will generate a natural-language response.

A grounding instruction can tell the model to use supplied evidence and say when the evidence does not contain the answer. This reduces unsupported answers but cannot guarantee hallucinations are impossible.

The Gemini chat model is separate from the Hugging Face embedding model.

### C. Prompt template

A prompt template provides a consistent structure for model input. Conceptual example:

```text
Use only the context below to answer the question.
If the context does not contain the answer, say you cannot find it.

Context:
{context}

Question:
{question}
```

The placeholders are filled at runtime.

### D. Source citations

Retrieved documents carry metadata such as source file and page number. The application can return those details alongside the answer.

Citations must point to evidence actually retrieved; never invent page numbers or sources.

### E. Pydantic

Pydantic validates and parses data against declared Python models and types. It can validate API request fields and define structured response models containing an answer and source list.

Validation does not prove that an AI-generated answer is factually correct.

### F. FastAPI

FastAPI is a Python framework for creating HTTP APIs. A future endpoint can accept a JSON question and return an answer with source metadata.

FastAPI defines the API routes and request/response behaviour; it is not itself the web server.

### G. Uvicorn

Uvicorn is an ASGI server that can run the FastAPI application.

- FastAPI: API application and route definitions.
- Uvicorn: server process that serves the application.

### H. Evaluation and testing

Evaluate retrieval and generation separately.

Retrieval evaluation asks whether the right evidence was retrieved. Answer evaluation asks whether the answer is correct, grounded, relevant, and appropriately cited.

Test:
- Questions with clear answers in the PDF.
- Questions whose answers are absent.
- Questions using synonyms.
- Questions requiring multiple passages.
- Questions likely to retrieve similar but misleading content.

### I. Error handling and reliability

Handle missing/unreadable documents, empty questions, embedding failures, vector database errors, Gemini API errors/rate limits, empty retrieval results, invalid requests, timeouts, and unexpected exceptions.

Log errors safely. Do not write API keys or sensitive document contents to logs.

## 7. Security and production considerations

- Keep API keys in environment variables or a secrets manager.
- Do not commit `.env`.
- Apply authentication and authorization for private documents.
- Ensure users retrieve only documents they are allowed to access.
- Validate uploaded files and limit file size/type.
- Support document update and deletion workflows.
- Monitor latency, errors, token usage, and retrieval quality.
- Add tests for ingestion and API behaviour.

## 8. Interview questions

**What makes a RAG application production-oriented?** Reliability, persistent data, grounded answers, source traceability, APIs, validation, security, evaluation, and monitoring.

**Why use persistent Chroma storage?** Indexed vectors can be reused after the process ends.

**Why use a separate collection?** To isolate this application's vectors from Project 3 and avoid accidental mixing.

**Why use the same embedding model for documents and queries?** Their vectors need to be in a compatible embedding space for meaningful similarity comparisons.

**What is the limitation of the current duplicate check?** It skips insertion into a non-empty collection but does not detect changed source files or update individual chunks.

**What does FastAPI do?** Defines HTTP API routes and request/response handling.

**What does Uvicorn do?** Runs the ASGI application server.

**What does Pydantic do?** Validates and parses structured data such as API request and response models.

**How can RAG reduce unsupported answers?** Retrieve evidence, provide it in the prompt, instruct the model to abstain when evidence is missing, and evaluate results. This reduces risk but does not guarantee correctness.

**What should RAG evaluation measure?** Retrieval quality and answer quality, including relevance, correctness, grounding, and citation support.

## Key takeaway

Project 4 combines the concepts from Projects 1–3 into an application. The ingestion and persistent-storage foundation is complete. Retrieval, Gemini generation, citations, API development, evaluation, and hardening remain to be implemented and tested.
