
# ============================================================
# PROJECT 15: HYBRID SEARCH AND RERANKING
#
# LEARNING METHOD:
# Library -> Explanation -> Practical Task
#
# OBJECTIVE:
# Build a retrieval system using:
# PDF Loading
# Text Chunking
# Embeddings
# ChromaDB Vector Search
# BM25 Keyword Search
# Hybrid Search
# Cross-Encoder Reranking
# ============================================================


# ============================================================
# LIBRARY 1: PyPDFLoader
# ============================================================

# EXPLANATION:
# PyPDFLoader is a LangChain document loader.
# It reads PDF files and converts their content into
# LangChain Document objects.
#
# A Document object contains:
# 1. page_content = extracted text
# 2. metadata = page number and source information
#
# IMPORTANT METHODS:
# PyPDFLoader() = creates the PDF loader.
# load() = reads the PDF and returns Document objects.


from langchain_community.document_loaders import PyPDFLoader


# PRACTICAL TASK 1: LOAD THE PDF

# Define the location of our PDF file.
pdf_path = "documents/TechVision_AI_RAG_Learning_Dataset.pdf"

# Create the PDF loader object.
loader = PyPDFLoader(pdf_path)

# Read the PDF.
documents = loader.load()

# Display the total number of pages.
print("\n========== LIBRARY 1: PDF LOADER ==========")

print("TOTAL PDF PAGES:", len(documents))

# Display the text from the first page.
print("\nFIRST PAGE CONTENT:")
print(documents[0].page_content[:500])

# Display the metadata of the first page.
print("\nFIRST PAGE METADATA:")
print(documents[0].metadata)


# ============================================================
# LIBRARY 2: RecursiveCharacterTextSplitter
# ============================================================

# EXPLANATION:
# This library splits large documents into smaller chunks.
#
# Why do we need chunking?
# Large documents are difficult to retrieve efficiently.
# RAG systems retrieve smaller relevant pieces of information.
#
# IMPORTANT PARAMETERS:
#
# chunk_size = maximum approximate chunk length in characters.
# chunk_overlap = characters shared between neighbouring chunks.
#
# IMPORTANT METHOD:
# split_documents() = splits Document objects into smaller
# Document objects while preserving their metadata.


from langchain_text_splitters import RecursiveCharacterTextSplitter


# PRACTICAL TASK 2: CHUNK THE PDF

# Create the text splitter.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split all PDF pages into smaller chunks.
chunks = text_splitter.split_documents(documents)

print("\n========== LIBRARY 2: TEXT CHUNKING ==========")

print("TOTAL ORIGINAL PAGES:", len(documents))

print("TOTAL CHUNKS:", len(chunks))

# Display the first chunk.
print("\nFIRST CHUNK:")
print(chunks[0].page_content)

# Display the first chunk's metadata.
print("\nFIRST CHUNK METADATA:")
print(chunks[0].metadata)


# ============================================================
# LIBRARY 3: HuggingFaceEmbeddings
# ============================================================

# EXPLANATION:
# Embeddings convert text into numerical vectors.
#
# These vectors represent semantic information.
#
# Similar meanings can have similar vector representations.
#
# We are using:
# all-MiniLM-L6-v2
#
# This model produces 384-dimensional embeddings.
#
# IMPORTANT METHOD:
# embed_query() = converts a single query into an embedding.
# embed_documents() = converts multiple documents into embeddings.
#
# LangChain's Chroma integration can call the embedding model
# automatically during storage and retrieval.


from langchain_huggingface import HuggingFaceEmbeddings


# PRACTICAL TASK 3: GENERATE EMBEDDINGS

# Load the embedding model.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Select the text from the first chunk.
sample_text = chunks[0].page_content

# Convert the text into a numerical vector.
sample_embedding = embedding_model.embed_query(sample_text)

print("\n========== LIBRARY 3: EMBEDDINGS ==========")

print("EMBEDDING DIMENSIONS:", len(sample_embedding))

# Display the first 10 numerical values.
print("\nFIRST 10 EMBEDDING VALUES:")
print(sample_embedding[:10])


# ============================================================
# LIBRARY 4: ChromaDB
# ============================================================

# EXPLANATION:
# ChromaDB is a vector database.
#
# It stores:
# 1. Document text
# 2. Embeddings
# 3. Metadata
#
# It allows us to search documents based on semantic similarity.
#
# IMPORTANT PARAMETERS:
#
# collection_name = name of the database collection.
# embedding_function = model used to generate embeddings.
# persist_directory = location where database is saved.
#
# IMPORTANT METHODS:
#
# add_documents() = stores documents and their embeddings.
# similarity_search() = retrieves semantically similar documents.
#
# We are reusing the ChromaDB database from Project 14.
# We will not insert the same PDF chunks again.


from langchain_chroma import Chroma


# PRACTICAL TASK 4: CONNECT TO CHROMADB

vector_database = Chroma(
    collection_name="techvision_rag",
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)

print("\n========== LIBRARY 4: CHROMADB ==========")

# Display the number of stored vectors.
print(
    "TOTAL STORED VECTORS:",
    vector_database._collection.count()
)

# Define our search question.
query = "What is machine learning?"

# Retrieve the top 5 semantically relevant documents.
vector_results = vector_database.similarity_search(
    query,
    k=5
)

print("\nVECTOR SEARCH RESULTS:")

for i, document in enumerate(vector_results):

    print("\nRESULT:", i + 1)

    print(document.page_content[:300])


# ============================================================
# LIBRARY 5: rank-bm25
# ============================================================

# EXPLANATION:
# BM25 is a keyword-based information retrieval algorithm.
#
# It finds documents based on matching words in the query.
#
# Unlike vector search, BM25 does not require embeddings.
#
# IMPORTANT CLASS:
# BM25Okapi() = creates the BM25 search model.
#
# IMPORTANT METHOD:
# get_scores() = calculates relevance scores for documents.
#
# Higher scores generally indicate stronger keyword relevance.


from rank_bm25 import BM25Okapi


# PRACTICAL TASK 5: KEYWORD SEARCH USING BM25

# Convert every chunk into lowercase words.
tokenized_documents = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

# Create the BM25 model using our document collection.
bm25 = BM25Okapi(tokenized_documents)

# Convert the question into lowercase words.
tokenized_query = query.lower().split()

# Calculate a relevance score for every chunk.
bm25_scores = bm25.get_scores(tokenized_query)

# Sort chunk indexes by their relevance scores.
# reverse=True means highest score comes first.
# [:5] selects the top 5 results.

bm25_indexes = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:5]

# Retrieve the actual documents using their indexes.
bm25_results = [
    chunks[i]
    for i in bm25_indexes
]

print("\n========== LIBRARY 5: BM25 SEARCH ==========")

for i, document in enumerate(bm25_results):

    print("\nRESULT:", i + 1)

    print(document.page_content[:300])

    print("BM25 SCORE:", round(float(bm25_scores[bm25_indexes[i]]), 4))


# ============================================================
# PRACTICAL TASK 6: HYBRID SEARCH USING RRF
# ============================================================

# EXPLANATION:
# Hybrid search combines:
#
# 1. BM25 keyword search
# 2. ChromaDB vector search
#
# Why combine them?
#
# BM25 is good at matching exact keywords.
# Vector search is good at finding similar meanings.
#
# RRF = Reciprocal Rank Fusion.
#
# RRF combines rankings instead of directly adding
# BM25 scores and vector similarity scores.
#
# Formula:
#
# RRF score = 1 / (60 + rank)
#
# A document appearing near the top of both searches
# receives contributions from both rankings.


# Create an empty dictionary to store combined results.
combined_results = {}

# Process both search result lists.
for result_list in [bm25_results, vector_results]:

    # Enumerate each document and its position.
    for rank, document in enumerate(result_list):

        # Use document text as its unique key.
        document_id = document.page_content

        # If the document has not been stored yet,
        # create a new dictionary entry.
        if document_id not in combined_results:

            combined_results[document_id] = {
                "document": document,
                "score": 0
            }

        # Add the RRF contribution for this ranking.
        # rank starts at zero, so we add 1.
        combined_results[document_id]["score"] += (
            1 / (60 + rank + 1)
        )


# Sort documents by their combined RRF scores.
hybrid_results = sorted(
    combined_results.values(),
    key=lambda item: item["score"],
    reverse=True
)

# Extract the top 5 actual documents.
hybrid_documents = [
    item["document"]
    for item in hybrid_results[:5]
]

print("\n========== HYBRID SEARCH RESULTS ==========")

for i, document in enumerate(hybrid_documents):

    print("\nRESULT:", i + 1)

    print(document.page_content[:300])


# ============================================================
# LIBRARY 6: CrossEncoder
# ============================================================

# EXPLANATION:
# CrossEncoder is used for reranking.
#
# It evaluates the question and document together.
#
# It gives each question-document pair a relevance score.
#
# A higher score generally means stronger relevance.
#
# IMPORTANT CLASS:
# CrossEncoder() = loads the pretrained reranking model.
#
# IMPORTANT METHOD:
# predict() = calculates scores for query-document pairs.
#
# MODEL:
# cross-encoder/ms-marco-MiniLM-L-6-v2
#
# This model is trained for relevance scoring.


from sentence_transformers import CrossEncoder


# PRACTICAL TASK 7: RERANK HYBRID RESULTS

# Load the pretrained CrossEncoder model.
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Create pairs of:
# (user question, retrieved document text)

pairs = [
    (query, document.page_content)
    for document in hybrid_documents
]

# Calculate relevance scores for all document pairs.
rerank_scores = reranker.predict(pairs)

# Combine each document with its reranking score.
# Sort using the score, highest first.

reranked_results = sorted(
    zip(hybrid_documents, rerank_scores),
    key=lambda item: item[1],
    reverse=True
)


# ============================================================
# FINAL OUTPUT: RERANKED DOCUMENTS
# ============================================================

print("\n========== FINAL RERANKED RESULTS ==========")

for i, (document, score) in enumerate(reranked_results):

    print("\nRESULT:", i + 1)

    print("RERANK SCORE:", round(float(score), 4))

    print("CONTENT:")

    print(document.page_content)


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n============================================")
print("PROJECT 15 COMPLETED SUCCESSFULLY!")
print("============================================")
