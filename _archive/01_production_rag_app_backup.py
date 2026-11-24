
# ============================================================
# ADVANCED RAG ENGINEERING LAB
# PROJECT 4 - PRODUCTION RAG APPLICATION
# ============================================================

import os
from dotenv import load_dotenv

# LangChain document loader
from langchain_community.document_loaders import PyPDFLoader

# LangChain text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hugging Face embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# ChromaDB vector database
from langchain_chroma import Chroma

# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Prompt template
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Find the main project folder.
PROJECT_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Load environment variables from the project root.
load_dotenv(os.path.join(PROJECT_PATH, ".env"))

# Read Gemini API key from .env.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini model (can be changed in .env).
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

# PDF file location.
PDF_PATH = os.path.join(
    PROJECT_PATH,
    "documents",
    "TechVision_AI_RAG_Learning_Dataset.pdf"
)

# Persistent ChromaDB location.
CHROMA_PATH = os.path.join(
    PROJECT_PATH,
    "chroma_db"
)

# Separate collection for Project 4.
COLLECTION_NAME = "production_rag_v1"

# Number of document chunks to retrieve.
RETRIEVAL_K = 4


# ============================================================
# 2. LOAD DOCUMENTS
# ============================================================

def load_documents():

    print("\n1. LOADING DOCUMENT...")

    if not os.path.exists(PDF_PATH):

        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}"
        )

    # Load PDF pages.
    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    if not documents:

        raise ValueError("No documents were loaded.")

    print(f"PDF pages loaded: {len(documents)}")

    return documents


# ============================================================
# 3. SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

def split_documents(documents):

    print("\n2. SPLITTING DOCUMENT...")

    # Divide documents into smaller chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    if not chunks:

        raise ValueError("No document chunks generated.")

    print(f"Total chunks generated: {len(chunks)}")

    return chunks


# ============================================================
# 4. INITIALIZE HUGGING FACE EMBEDDINGS
# ============================================================

def create_embedding_model():

    print("\n3. INITIALIZING EMBEDDING MODEL...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model ready!")

    return embedding_model


# ============================================================
# 5. CREATE / CONNECT TO CHROMADB
# ============================================================

def create_vector_database(chunks, embedding_model):

    print("\n4. CONNECTING TO CHROMADB...")

    # Connect to persistent ChromaDB.
    vector_database = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH
    )

    # Check existing vector IDs.
    existing_ids = vector_database.get()["ids"]

    if len(existing_ids) == 0:

        print("New collection detected.")

        print("Storing document chunks...")

        vector_database.add_documents(chunks)

        print("Documents successfully stored!")

    else:

        print(
            f"Existing collection found: {len(existing_ids)} chunks."
        )

        print("Skipping ingestion to prevent duplicates.")

    # Check total stored vectors.
    total_vectors = len(
        vector_database.get()["ids"]
    )

    print(f"Total vectors stored: {total_vectors}")

    return vector_database


# ============================================================
# 6. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_relevant_documents(vector_database, question):

    print("\n5. RETRIEVING RELEVANT DOCUMENTS...")

    # Search ChromaDB using the question.
    retrieved_documents = vector_database.similarity_search(
        question,
        k=RETRIEVAL_K
    )

    if not retrieved_documents:

        raise ValueError("No relevant documents were retrieved.")

    print(
        f"\nRetrieved {len(retrieved_documents)} document chunks."
    )

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        print(f"\nDocument {index}:")

        print(document.page_content[:250])

        print(
            "Source:",
            document.metadata.get("source", "Unknown")
        )

        # PyPDFLoader uses zero-based page numbering.
        page = document.metadata.get("page")

        if page is not None:
            print("PDF Page:", page + 1)
        else:
            print("PDF Page: Unknown")

    return retrieved_documents


# ============================================================
# 7. GENERATE SOURCE INFORMATION
# ============================================================

def format_sources(retrieved_documents):

    sources = []

    seen_sources = set()

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        source_path = document.metadata.get(
            "source",
            "Unknown"
        )

        # Display only the PDF filename.
        source_name = os.path.basename(source_path)

        page = document.metadata.get("page")

        # Convert zero-based page to normal PDF page number.
        page_number = page + 1 if page is not None else "Unknown"

        source_key = (source_name, page_number)

        if source_key not in seen_sources:

            seen_sources.add(source_key)

            sources.append(
                f"[Source {index}] {source_name} - Page {page_number}"
            )

    return sources


# ============================================================
# 8. GEMINI ANSWER GENERATION
# ============================================================

def generate_answer(question, retrieved_documents):

    print("\n6. GENERATING ANSWER USING GEMINI...")

    if not GEMINI_API_KEY:

        raise ValueError(
            "GEMINI_API_KEY is missing from the .env file."
        )

    # Initialize Gemini.
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

    # Prepare context from retrieved document chunks.
    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        source_name = os.path.basename(
            document.metadata.get("source", "Unknown")
        )

        page = document.metadata.get("page")

        page_number = page + 1 if page is not None else "Unknown"

        context_parts.append(
            f"""
            [Source {index}]
            File: {source_name}
            Page: {page_number}

            Content:
            {document.page_content}
            """
        )

    # Combine all retrieved chunks.
    context = "\n\n".join(context_parts)

    # Create Gemini prompt template.
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful AI research assistant.

        Answer the user's question using ONLY the
        information provided in the context.

        Instructions:

        1. Give a clear and accurate answer.
        2. Do not invent information.
        3. If the answer is unavailable in the context,
           clearly say that you could not find it.
        4. Include source references such as [Source 1]
           wherever appropriate.
        5. Use simple and understandable language.

        CONTEXT:

        {context}

        USER QUESTION:

        {question}

        ANSWER:
        """
    )

    # Format the prompt.
    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    # Send the prompt to Gemini.
    response = llm.invoke(formatted_prompt)

    answer = response.content

    print("\nGEMINI ANSWER:")
    print("=" * 60)

    print(answer)

    return answer


# ============================================================
# 9. DISPLAY SOURCE CITATIONS
# ============================================================

def display_sources(retrieved_documents):

    print("\n7. SOURCE CITATIONS")

    print("=" * 60)

    sources = format_sources(retrieved_documents)

    for source in sources:

        print(source)

    print("=" * 60)


# ============================================================
# 10. RUN COMPLETE RAG PIPELINE
# ============================================================

def run_rag(question, vector_database):

    # Retrieve relevant chunks.
    retrieved_documents = retrieve_relevant_documents(
        vector_database,
        question
    )

    # Generate answer using Gemini.
    answer = generate_answer(
        question,
        retrieved_documents
    )

    # Display PDF source references.
    display_sources(retrieved_documents)

    return answer


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 65)

    print("       ADVANCED RAG ENGINEERING LAB")

    print("              PRODUCTION RAG")

    print("=" * 65)

    # Step 1: Load PDF.
    documents = load_documents()

    # Step 2: Split into chunks.
    chunks = split_documents(documents)

    # Step 3: Initialize embeddings.
    embedding_model = create_embedding_model()

    # Step 4: Connect to ChromaDB.
    vector_database = create_vector_database(
        chunks,
        embedding_model
    )

    print("\n" + "=" * 65)

    print("PRODUCTION RAG SYSTEM READY!")

    print("=" * 65)

    # Interactive question-answering loop.
    while True:

        question = input(
            "\nAsk a question about TechVision AI (or type 'exit'): "
        ).strip()

        if question.lower() in ["exit", "quit"]:

            print("\nThank you for using Production RAG!")

            break

        if not question:

            print("Please enter a question.")

            continue

        try:

            answer = run_rag(
                question,
                vector_database
            )

        except Exception as error:

            print("\nERROR WHILE PROCESSING QUESTION:")

            print(str(error))

            print(
                "\nPlease check your configuration and try again."
            )


# ============================================================
# 12. START APPLICATION
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\nAPPLICATION STARTUP ERROR:")

        print(str(error))
