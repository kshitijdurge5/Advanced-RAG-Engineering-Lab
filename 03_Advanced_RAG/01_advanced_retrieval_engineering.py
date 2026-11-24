
# ============================================================
# PROJECT 3: ADVANCED RETRIEVAL ENGINEERING
# ============================================================
#
# TECHNIQUES:
# 1. BM25 Keyword Search
# 2. ChromaDB Vector Search
# 3. Hybrid Search
# 4. Reciprocal Rank Fusion (RRF)
# 5. Maximum Marginal Relevance (MMR)
# 6. CrossEncoder Reranking
# 7. Multi-Query Retrieval
# 8. Contextual Compression
#
# MODEL PROVIDER: Google Gemini
# EMBEDDINGS: Hugging Face
#
# ============================================================


# STEP 1: IMPORT LIBRARIES

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langchain_classic.retrievers import (
    MultiQueryRetriever,
    ContextualCompressionRetriever
)

from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor
)

from langchain_google_genai import ChatGoogleGenerativeAI

from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder


# STEP 2: PROJECT CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_FOLDER = PROJECT_ROOT / "documents"

CHROMA_FOLDER = PROJECT_ROOT / "chroma_db"

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

PDF_PATH = DOCUMENTS_FOLDER / "TechVision_AI_RAG_Learning_Dataset.pdf"

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

COLLECTION_NAME = "techvision_rag"

print("\n" + "=" * 65)
print("PROJECT 3: ADVANCED RETRIEVAL ENGINEERING")
print("=" * 65)


# STEP 3: LOAD PDF DOCUMENTS

# PyPDFLoader extracts PDF pages into LangChain Documents.

pdf_loader = PyPDFLoader(str(PDF_PATH))

documents = pdf_loader.load()

print("\n1. PDF LOADING")
print("Total PDF pages:", len(documents))


# STEP 4: SPLIT DOCUMENTS INTO CHUNKS

# RecursiveCharacterTextSplitter creates smaller text pieces.
# Metadata from the original pages is preserved.

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("\n2. DOCUMENT CHUNKING")
print("Total chunks:", len(chunks))


# STEP 5: LOAD HUGGING FACE EMBEDDINGS

# Embeddings convert text into numerical vectors.
# The same embedding model must be used for indexing
# and querying an existing ChromaDB collection.

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\n3. EMBEDDING MODEL READY")


# STEP 6: CONNECT TO EXISTING CHROMADB

# Connect to the existing collection.
# Do not insert the PDF chunks again.
# This preserves the existing stored vectors.

vector_database = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=str(CHROMA_FOLDER)
)

stored_count = vector_database._collection.count()

if stored_count == 0:
    raise ValueError(
        "The ChromaDB collection is empty. "
        "Check your existing Project 2 database."
    )

print("\n4. CHROMADB CONNECTION")
print("Collection:", COLLECTION_NAME)
print("Stored vectors:", stored_count)


# STEP 7: DEFINE USER QUERY

query = "What is machine learning and how is it used?"

print("\n5. USER QUERY")
print(query)


# STEP 8: BM25 KEYWORD SEARCH

# BM25 searches for matching words.
# It does not need embeddings.

tokenized_documents = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_documents)

tokenized_query = query.lower().split()

bm25_scores = bm25.get_scores(tokenized_query)

bm25_indexes = sorted(
    range(len(bm25_scores)),
    key=lambda index: bm25_scores[index],
    reverse=True
)[:5]

bm25_results = [
    chunks[index]
    for index in bm25_indexes
]

print("\n6. BM25 KEYWORD SEARCH")

for index, document in enumerate(bm25_results, start=1):
    print(f"\nBM25 Result {index}")
    print(document.page_content[:250])


# STEP 9: CHROMADB VECTOR SEARCH

# Vector search finds documents based on semantic meaning.

vector_results = vector_database.similarity_search(
    query,
    k=5
)

print("\n7. VECTOR SEARCH")

for index, document in enumerate(vector_results, start=1):
    print(f"\nVector Result {index}")
    print(document.page_content[:250])


# STEP 10: HYBRID SEARCH USING RRF

# RRF combines ranked results.
#
# RRF score = 1 / (60 + rank)
#
# Rank starts at 1 in this implementation.

combined_results = {}

for result_list in [bm25_results, vector_results]:

    for rank, document in enumerate(result_list, start=1):

        document_id = (
            document.metadata.get("source", ""),
            document.metadata.get("page", ""),
            document.page_content
        )

        if document_id not in combined_results:

            combined_results[document_id] = {
                "document": document,
                "score": 0
            }

        combined_results[document_id]["score"] += (
            1 / (60 + rank)
        )

hybrid_results = sorted(
    combined_results.values(),
    key=lambda item: item["score"],
    reverse=True
)

hybrid_documents = [
    item["document"]
    for item in hybrid_results[:5]
]

print("\n8. HYBRID SEARCH: RRF")

for index, item in enumerate(hybrid_results[:5], start=1):

    print(f"\nHybrid Result {index}")
    print("RRF Score:", round(item["score"], 5))
    print(item["document"].page_content[:250])


# STEP 11: MAXIMUM MARGINAL RELEVANCE (MMR)

# MMR balances:
# 1. Relevance to the query.
# 2. Diversity among retrieved documents.
#
# This helps reduce repetitive search results.

mmr_results = vector_database.max_marginal_relevance_search(
    query,
    k=5,
    fetch_k=15,
    lambda_mult=0.5
)

print("\n9. MAXIMUM MARGINAL RELEVANCE")

for index, document in enumerate(mmr_results, start=1):

    print(f"\nMMR Result {index}")
    print(document.page_content[:250])


# STEP 12: CROSS-ENCODER RERANKING

# CrossEncoder evaluates query and document together.
# It assigns a relevance score to each pair.

print("\n10. CROSS-ENCODER RERANKING")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

rerank_pairs = [
    (query, document.page_content)
    for document in hybrid_documents
]

rerank_scores = reranker.predict(rerank_pairs)

reranked_results = sorted(
    zip(hybrid_documents, rerank_scores),
    key=lambda item: item[1],
    reverse=True
)

for index, (document, score) in enumerate(
    reranked_results,
    start=1
):

    print(f"\nReranked Result {index}")
    print("Relevance score:", round(float(score), 4))
    print(document.page_content[:250])


# STEP 13: INITIALIZE GEMINI MODEL

# Gemini will generate alternative search queries
# and help extract relevant content.

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
    google_api_key=API_KEY,
    temperature=0
)

print("\n11. GEMINI MODEL READY")


# STEP 14: MULTI-QUERY RETRIEVAL

# MultiQueryRetriever generates alternative versions
# of the original query.
#
# This can retrieve relevant documents that a single
# search query might miss.

base_retriever = vector_database.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

print("\n12. MULTI-QUERY RETRIEVAL")

multi_query_results = multi_query_retriever.invoke(query)

print("Total retrieved documents:", len(multi_query_results))

for index, document in enumerate(
    multi_query_results[:5],
    start=1
):

    print(f"\nMulti-Query Result {index}")
    print(document.page_content[:250])


# STEP 15: CONTEXTUAL COMPRESSION

# Contextual compression filters retrieved documents.
#
# It uses an LLM to extract the relevant portions
# instead of passing entire retrieved chunks forward.

print("\n13. CONTEXTUAL COMPRESSION")

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

compressed_results = compression_retriever.invoke(query)

print("Compressed documents:", len(compressed_results))

for index, document in enumerate(
    compressed_results[:5],
    start=1
):

    print(f"\nCompressed Result {index}")
    print(document.page_content[:400])


# STEP 16: FINAL SUMMARY

print("\n" + "=" * 65)
print("PROJECT 3 RETRIEVAL SUMMARY")
print("=" * 65)

print("Original PDF pages:", len(documents))
print("Newly generated chunks:", len(chunks))
print("ChromaDB stored vectors:", stored_count)
print("BM25 results:", len(bm25_results))
print("Vector results:", len(vector_results))
print("Hybrid results:", len(hybrid_documents))
print("MMR results:", len(mmr_results))
print("Reranked results:", len(reranked_results))
print("Multi-query results:", len(multi_query_results))
print("Compressed results:", len(compressed_results))

print("\nPROJECT 3 COMPLETED SUCCESSFULLY!")
