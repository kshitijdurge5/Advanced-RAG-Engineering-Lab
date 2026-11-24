
# ============================================================
# PROJECT 2: CHUNKING, EMBEDDINGS AND VECTOR SEARCH
# ============================================================
#
# CONCEPTS COVERED:
# 1. Character Text Splitting
# 2. Recursive Character Splitting
# 3. Markdown Header Splitting
# 4. Token-based Splitting
# 5. Semantic Chunking
# 6. Text Embeddings
# 7. Cosine Similarity
# 8. ChromaDB Vector Search
#
# EMBEDDING MODELS:
# - Hugging Face: Local embedding model
# - Gemini: Google API embedding model (optional test)
#
# ============================================================


# STEP 1: IMPORT LIBRARIES

import os
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter
)

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_experimental.text_splitter import SemanticChunker


# STEP 2: LOAD ENVIRONMENT VARIABLES

# Load Gemini API key from the main project's .env file.
# Never print or share your API key.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


# STEP 3: LOAD SOURCE DOCUMENTS

# We will use the existing TXT and PDF documents.
# Both are already present in the documents folder.

DOCUMENTS_FOLDER = PROJECT_ROOT / "documents"

text_file = DOCUMENTS_FOLDER / "company_information.txt"

pdf_file = DOCUMENTS_FOLDER / "TechVision_AI_RAG_Learning_Dataset.pdf"

if not text_file.exists() or not pdf_file.exists():
    raise FileNotFoundError("Required TXT or PDF document is missing.")

text_documents = TextLoader(
    str(text_file),
    encoding="utf-8"
).load()

pdf_documents = PyPDFLoader(
    str(pdf_file)
).load()

# Combine the text extracted from both sources.
source_text = "\n\n".join(
    [doc.page_content for doc in text_documents + pdf_documents]
)

print("\n" + "=" * 60)
print("PROJECT 2: CHUNKING AND EMBEDDINGS")
print("=" * 60)

print("\nSOURCE DOCUMENTS")
print("Total characters:", len(source_text))
print("TXT documents:", len(text_documents))
print("PDF pages:", len(pdf_documents))


 

# ============================================================
# STEP 4: CHARACTER TEXT SPLITTER
# ============================================================

# CharacterTextSplitter divides text using a chosen separator.
# Here, we use paragraphs as the separator.
#
# chunk_size = maximum chunk size
# chunk_overlap = repeated characters between neighbouring chunks

character_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=100
)

character_chunks = character_splitter.create_documents(
    [source_text]
)

print("\n1. CHARACTER TEXT SPLITTER")
print("Total chunks:", len(character_chunks))

if character_chunks:
    print("\nFirst chunk:")
    print(character_chunks[0].page_content[:300])


# ============================================================
# STEP 5: RECURSIVE CHARACTER TEXT SPLITTER
# ============================================================

# RecursiveCharacterTextSplitter tries different separators.
#
# It first looks for paragraphs.
# Then it tries lines.
# Then sentences and words.
#
# This usually preserves meaning better than basic splitting.

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

recursive_chunks = recursive_splitter.create_documents(
    [source_text]
)

print("\n2. RECURSIVE CHARACTER SPLITTER")
print("Total chunks:", len(recursive_chunks))

if recursive_chunks:
    print("\nFirst chunk:")
    print(recursive_chunks[0].page_content[:300])


# ============================================================
# STEP 6: CHUNK SIZE AND OVERLAP COMPARISON
# ============================================================

# Different chunk sizes produce different numbers of chunks.
# We compare small, medium and large chunk sizes.

print("\n3. CHUNK SIZE COMPARISON")

chunk_sizes = [200, 500, 1000]

for size in chunk_sizes:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=50
    )

    chunks = splitter.create_documents([source_text])

    print(
        f"Chunk size: {size} | "
        f"Number of chunks: {len(chunks)}"
    )


# ============================================================
# STEP 7: MARKDOWN HEADER TEXT SPLITTER
# ============================================================

# MarkdownHeaderTextSplitter separates content using
# headings such as #, ## and ###.
#
# This is useful for structured documents.

markdown_text = """
# Artificial Intelligence

Artificial Intelligence allows machines to perform tasks
that normally require human intelligence.

## Machine Learning

Machine Learning allows computers to learn from data.

### Supervised Learning

Supervised learning uses labelled training data.

### Unsupervised Learning

Unsupervised learning discovers patterns in unlabelled data.

## Deep Learning

Deep Learning uses neural networks with multiple layers.
"""

headers_to_split_on = [
    ("#", "Main Topic"),
    ("##", "Sub Topic"),
    ("###", "Section")
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

markdown_chunks = markdown_splitter.split_text(
    markdown_text
)

print("\n4. MARKDOWN HEADER SPLITTER")
print("Total chunks:", len(markdown_chunks))

for index, chunk in enumerate(markdown_chunks, start=1):

    print(f"\nMarkdown chunk {index}:")
    print("Content:", chunk.page_content)
    print("Metadata:", chunk.metadata)


# ============================================================
# STEP 8: TOKEN TEXT SPLITTER
# ============================================================

# TokenTextSplitter divides text based on tokens,
# rather than ordinary characters.
#
# Tokens are the smaller text units used by language models.
#
# This example uses the cl100k_base tokenizer.

token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=100,
    chunk_overlap=20
)

token_chunks = token_splitter.create_documents(
    [source_text]
)

print("\n5. TOKEN TEXT SPLITTER")
print("Total chunks:", len(token_chunks))

if token_chunks:
    print("\nFirst token chunk:")
    print(token_chunks[0].page_content[:300])


# ============================================================
# STEP 9: LOAD HUGGING FACE EMBEDDING MODEL
# ============================================================

# Embeddings convert text into numerical vectors.
#
# Similar meanings should generally produce vectors
# that are closer together in the embedding space.
#
# This model runs locally after it has been downloaded.

print("\n6. LOADING HUGGING FACE EMBEDDING MODEL")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Hugging Face embedding model loaded successfully!")


# ============================================================
# STEP 10: SEMANTIC CHUNKING
# ============================================================

# SemanticChunker groups sentences based on their meaning.
#
# Instead of splitting after a fixed number of characters,
# it looks for changes in semantic similarity.
#
# It uses an embedding model to compare sentences.

print("\n7. SEMANTIC CHUNKING")

try:

    semantic_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile"
    )

    semantic_chunks = semantic_splitter.create_documents(
        [source_text]
    )

    print("Total semantic chunks:", len(semantic_chunks))

    if semantic_chunks:
        print("\nFirst semantic chunk:")
        print(semantic_chunks[0].page_content[:300])

except Exception as error:

    semantic_chunks = []

    print("Semantic chunking could not complete.")
    print("Error:", error)


# ============================================================
# STEP 11: GENERATE TEXT EMBEDDINGS
# ============================================================

# Let's convert a few chunks into numerical vectors.

sample_texts = [
    chunk.page_content
    for chunk in recursive_chunks[:5]
]

if len(sample_texts) < 2:
    raise ValueError("Not enough text chunks for embedding comparison.")

document_embeddings = embedding_model.embed_documents(
    sample_texts
)

print("\n8. HUGGING FACE TEXT EMBEDDINGS")

print("Number of embedded texts:", len(document_embeddings))

print("Embedding dimensions:", len(document_embeddings[0]))

print("\nFirst embedding vector:")
print(document_embeddings[0][:10])


# ============================================================
# STEP 12: EMBED A SEARCH QUERY
# ============================================================

# A user query must also be converted into a vector
# before it can be compared with document embeddings.

query = "What is artificial intelligence and machine learning?"

query_embedding = embedding_model.embed_query(query)

print("\n9. QUERY EMBEDDING")

print("Query:", query)

print("Query embedding dimensions:", len(query_embedding))


# ============================================================
# STEP 13: COSINE SIMILARITY
# ============================================================

# Cosine similarity compares the direction of two vectors.
#
# A value closer to 1 generally means greater similarity.
# A value around 0 means little directional similarity.
# A value closer to -1 means opposite directions.
#
# Embedding similarity is a measure, not proof of meaning.

def cosine_similarity(vector_a, vector_b):

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    numerator = np.dot(vector_a, vector_b)

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


print("\n10. COSINE SIMILARITY")

similarity_results = []

for index, embedding in enumerate(document_embeddings):

    similarity = cosine_similarity(
        query_embedding,
        embedding
    )

    similarity_results.append(
        (index, similarity, sample_texts[index])
    )

similarity_results.sort(
    key=lambda item: item[1],
    reverse=True
)

for index, score, content in similarity_results:

    print("\nChunk:", index + 1)
    print("Similarity score:", round(float(score), 4))
    print("Content:", content[:200])


# ============================================================
# STEP 14: OPTIONAL GEMINI EMBEDDING TEST
# ============================================================

# Gemini embeddings use Google's embedding API.
# This section is optional.
#
# The API key is read from the .env file.
# We do not print the API key.

print("\n11. GEMINI EMBEDDING TEST")

gemini_api_key = os.getenv("GEMINI_API_KEY")

if gemini_api_key:

    try:

        from langchain_google_genai import (
            GoogleGenerativeAIEmbeddings
        )

        gemini_model_name = os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "models/gemini-embedding-001"
        )

        gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model=gemini_model_name,
            google_api_key=gemini_api_key
        )

        gemini_vector = gemini_embeddings.embed_query(query)

        print("Gemini embeddings working!")
        print("Gemini model:", gemini_model_name)
        print("Gemini vector dimensions:", len(gemini_vector))
        print("First 10 values:", gemini_vector[:10])

    except Exception as error:

        print("Gemini embedding test was unsuccessful.")
        print("Check the model name, API access and package compatibility.")
        print("Error:", error)

else:

    print("Gemini API key not found in .env.")
    print("Skipping optional Gemini embedding test.")


# ============================================================
# STEP 15: CHROMADB VECTOR DATABASE
# ============================================================

# ChromaDB stores document embeddings in a vector database.
#
# We use recursive_chunks and Hugging Face embeddings.
#
# No persist_directory is specified, so this demonstration
# does not overwrite your existing chroma_db folder.

print("\n12. CHROMADB VECTOR DATABASE")

vector_database = Chroma.from_documents(
    documents=recursive_chunks,
    embedding=embedding_model
)

print("Documents added to ChromaDB:", len(recursive_chunks))


# ============================================================
# STEP 16: SIMILARITY SEARCH
# ============================================================

# Convert the query into a vector.
# ChromaDB finds the most similar document chunks.

search_query = "Explain artificial intelligence and machine learning."

search_results = vector_database.similarity_search(
    search_query,
    k=3
)

print("\n13. CHROMADB SIMILARITY SEARCH")

print("Search query:", search_query)

for index, result in enumerate(search_results, start=1):

    print(f"\nResult {index}:")
    print(result.page_content[:400])
    print("Metadata:", result.metadata)


# ============================================================
# STEP 17: FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("Character chunks:", len(character_chunks))
print("Recursive chunks:", len(recursive_chunks))
print("Markdown chunks:", len(markdown_chunks))
print("Token chunks:", len(token_chunks))
print("Semantic chunks:", len(semantic_chunks))
print("Embedding dimensions:", len(document_embeddings[0]))
print("ChromaDB search results:", len(search_results))

print("\nYou have completed:")
print("1. Five chunking techniques")
print("2. Chunk size and overlap comparison")
print("3. Semantic chunking")
print("4. Hugging Face embeddings")
print("5. Optional Gemini embeddings")
print("6. Cosine similarity")
print("7. ChromaDB vector search")

print("\nPROJECT 2 COMPLETED SUCCESSFULLY!")
