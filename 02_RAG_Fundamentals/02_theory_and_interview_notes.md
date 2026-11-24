# Project 2 — RAG Fundamentals: Chunking and Embeddings

**Code:** `01_chunking_and_embeddings.py`  
**Status:** Completed

## 1. What is RAG?

RAG means **Retrieval-Augmented Generation**:

- **Retrieval:** find information relevant to a question.
- **Augmented:** provide that information as context to a language model.
- **Generation:** generate a response using the question and retrieved context.

A company assistant may not know a private company policy. A RAG application can search company documents, retrieve the relevant passage, and provide it to a language model. RAG retrieves external information at query time; it does not retrain the model on every document.

## 2. Basic RAG stages

### Indexing stage

1. Load documents.
2. Split them into chunks.
3. Convert chunks into embeddings.
4. Store vectors, text, and metadata in a vector store.

### Query stage

1. Receive a question.
2. Embed the question using the compatible embedding model.
3. Search stored vectors.
4. Retrieve matching text chunks.
5. Provide the context and question to a language model.
6. Return an answer, ideally with sources.

This project focused on chunking, embeddings, similarity, and vector search.

## 3. Chunking

Chunking divides long documents into smaller pieces. It allows retrieval to find relevant passages rather than always passing an entire document.

Chunking can make retrieval more focused and help manage context-window limits. Very small chunks may lose context; very large chunks may include unrelated information.

### Methods explored

**CharacterTextSplitter**  
Splits using configured character separators and a target chunk size. Depending on the separator and input, chunks can exceed the target size.

**RecursiveCharacterTextSplitter**  
Tries a sequence of separators, commonly paragraph breaks, newlines, spaces, and smaller boundaries. It attempts to preserve natural text structure. It is a useful general-purpose splitter.

**MarkdownHeaderTextSplitter**  
Uses Markdown headings to divide content and can preserve heading context in metadata. Useful when document hierarchy matters.

**TokenTextSplitter**  
Splits based on token counts rather than character counts. Models process tokens, so token-aware splitting helps manage context windows and token budgets. A token is not necessarily a word or character.

**Semantic chunking**  
Attempts to find boundaries based on changes in meaning, often using embeddings to compare neighbouring text units. Its behaviour depends on the embedding model and breakpoint configuration; it does not guarantee perfect topic boundaries.

## 4. Chunk size and overlap

- `chunk_size` is a target size measured in the splitter's unit.
- `chunk_overlap` repeats some content from one chunk in the next.

Overlap can preserve context near a boundary. Too much overlap increases duplicate text, storage, and retrieval redundancy.

## 5. Embeddings

An embedding model maps text to a numerical vector. Similar meanings may be represented by vectors that are close under a chosen similarity or distance measure.

Illustrative vector only:

```python
[0.12, -0.34, 0.56, ...]
```

The dimensions are learned numerical features; do not assume each dimension corresponds to a simple human-readable concept.

## 6. Embedding models tested

### `sentence-transformers/all-MiniLM-L6-v2`

A sentence-transformer embedding model. In our experiment it produced **384-dimensional** vectors. It is useful for sentence and passage similarity and can run locally.

### `models/gemini-embedding-001`

Google's embedding model, tested through the Gemini integration. Our experiment used **3072-dimensional** embeddings.

An embedding model is different from a chat model:

- Embedding model: converts text into vectors.
- Chat/generative model: generates text responses.

A larger vector dimension does not automatically mean better retrieval. The model, data, task, and evaluation matter.

**Compatibility:** use a compatible embedding model and configuration for both indexing documents and embedding queries. Do not mix unrelated embedding spaces in one collection.

## 7. Cosine similarity

Cosine similarity compares the direction of two vectors:

\[
\operatorname{cosine}(A,B)=\frac{A\cdot B}{\|A\|\|B\|}
\]

A larger similarity often indicates a closer semantic match in common embedding-search setups. The exact interpretation depends on the model and vector-store configuration.

## 8. ChromaDB

Chroma is a vector database/vector store. It stores embeddings along with associated document text and metadata.

Typical flow:

1. Add chunks.
2. Chroma calls the configured embedding function.
3. Store vectors and associated content.
4. Embed a query using the same embedding function.
5. Search for nearest vectors.
6. Return matching documents and metadata.

A vector store retrieves information; it does not itself generate the final answer.

## 9. Project results

| Experiment | Result |
|---|---:|
| Character chunks | 9 |
| Recursive chunks | 40 |
| Markdown chunks | 5 |
| Token chunks | 37 |
| Semantic chunks | 10 |
| Hugging Face dimensions | 384 |
| Gemini dimensions | 3072 |
| ChromaDB search results | 3 |

The ChromaDB search was an in-memory demonstration and did not replace the separate persistent database used later.

## 10. Engineering considerations

- Tune chunk size using representative retrieval questions.
- Preserve source metadata.
- Use the same embedding model for indexing and querying.
- Evaluate retrieval quality rather than assuming a configuration is best.
- Consider scanned PDFs, tables, images, and multilingual content separately.
- Monitor embedding latency, quotas, and costs.

## 11. Interview questions

**What is RAG?** Retrieving external information and supplying it to a language model as context for generation.

**Why chunk documents?** To make retrieval more focused and manage the amount of context.

**What is overlap?** Repeated text shared between adjacent chunks to preserve context at boundaries.

**What is an embedding?** A numerical vector representation produced by a model.

**Embedding model vs LLM?** An embedding model creates vectors; a generative LLM produces responses.

**What does 384 dimensions mean?** Each embedding contains 384 numerical values.

**What is cosine similarity?** A measure comparing vector directions, often used for semantic similarity.

**What is a vector database?** A system that stores vectors and associated data and supports similarity search.

**Is a larger dimension always better?** No; evaluate model and retrieval performance for the task.

**Why use RecursiveCharacterTextSplitter?** It tries multiple separators to split text while preserving natural structure.

## Key takeaway

Project 2 explains how documents become searchable: split text into chunks, embed them, store the vectors, and retrieve semantically similar content.
