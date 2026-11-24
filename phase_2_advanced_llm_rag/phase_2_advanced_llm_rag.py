"""
PHASE 2: ADVANCED LLM AND RAG ENGINEERING
=========================================

BEGINNER GUIDE:
This project teaches how LLMs work and how to improve RAG retrieval.

Topics:
1. Tokens and tokenization
2. Transformer and attention concepts
3. Context windows and KV cache
4. Temperature and Top-P
5. Prompt engineering
6. Structured outputs
7. Embeddings and cosine similarity
8. Chunking strategies
9. BM25 keyword retrieval
10. Hybrid search and RRF
11. Reranking concepts
12. Query rewriting and HyDE
13. GraphRAG concepts
14. Recall, MRR and NDCG

Run:
python phase_2_advanced_llm_rag.py
"""

import math
import re
from collections import Counter


# ------------------------------------------------------------
# 1. TOKENIZATION
# ------------------------------------------------------------
# A token is a piece of text processed by an LLM.
# This is a simple educational tokenizer, not a model tokenizer.

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def token_demo():
    text = "Generative AI helps build intelligent applications."

    tokens = tokenize(text)

    print("\n1. TOKENIZATION")
    print("Text:", text)
    print("Tokens:", tokens)
    print("Approximate token count:", len(tokens))


# ------------------------------------------------------------
# 2. CHUNKING
# ------------------------------------------------------------
# Large documents are divided into smaller pieces called chunks.
# This helps retrieval find relevant information.

def chunk_text(text, chunk_size=40, overlap=10):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size > overlap >= 0")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]

        if not chunk:
            break

        chunks.append(" ".join(chunk))

        if start + chunk_size >= len(words):
            break

    return chunks


# ------------------------------------------------------------
# 3. COSINE SIMILARITY
# ------------------------------------------------------------
# Embeddings are numerical representations of text.
# Cosine similarity compares the direction of two vectors.

def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have equal dimensions.")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ------------------------------------------------------------
# 4. EDUCATIONAL KEYWORD EMBEDDING
# ------------------------------------------------------------
# This is NOT a neural embedding model.
# It creates simple vectors for learning similarity calculations.
# For real projects use Hugging Face or Gemini embeddings.

def keyword_vector(text, vocabulary):
    words = tokenize(text)

    return [
        words.count(term)
        for term in vocabulary
    ]


# ------------------------------------------------------------
# 5. BM25 KEYWORD RETRIEVAL
# ------------------------------------------------------------
# BM25 ranks documents according to keyword relevance.
# Unlike dense retrieval, it does not require embeddings.

def bm25_search(query, documents, k1=1.5, b=0.75):

    tokenized = [tokenize(doc) for doc in documents]
    query_terms = set(tokenize(query))

    total_docs = len(documents)

    if total_docs == 0:
        return []

    avg_length = sum(map(len, tokenized)) / total_docs
    scores = []

    for index, words in enumerate(tokenized):

        frequencies = Counter(words)
        score = 0.0
        length = len(words)

        for term in query_terms:

            document_frequency = sum(
                term in doc for doc in tokenized
            )

            if document_frequency == 0:
                continue

            idf = math.log(
                1 + (
                    total_docs - document_frequency + 0.5
                ) / (document_frequency + 0.5)
            )

            frequency = frequencies[term]

            denominator = frequency + k1 * (
                1 - b + b * length / (avg_length or 1)
            )

            score += idf * (
                frequency * (k1 + 1)
            ) / denominator

        scores.append((index, score))

    return sorted(scores, key=lambda item: item[1], reverse=True)


# ------------------------------------------------------------
# 6. RECIPROCAL RANK FUSION
# ------------------------------------------------------------
# RRF combines rankings from different retrieval methods.
# It is useful in hybrid search.

def reciprocal_rank_fusion(rankings, k=60):

    scores = {}

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = (
                scores.get(document_id, 0)
                + 1 / (k + rank)
            )

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


# ------------------------------------------------------------
# 7. RECALL@K
# ------------------------------------------------------------
# Recall measures how many relevant documents were retrieved.

def recall_at_k(retrieved, relevant, k):

    if not relevant:
        return 0.0

    found = set(retrieved[:k]) & set(relevant)

    return len(found) / len(relevant)


# ------------------------------------------------------------
# 8. MEAN RECIPROCAL RANK
# ------------------------------------------------------------
# MRR measures how high the first relevant result appears.

def reciprocal_rank(retrieved, relevant):

    for rank, item in enumerate(retrieved, start=1):

        if item in relevant:
            return 1 / rank

    return 0.0


# ------------------------------------------------------------
# 9. NDCG@K
# ------------------------------------------------------------
# NDCG evaluates the ordering of search results.

def ndcg_at_k(retrieved, relevance, k):

    def dcg(values):
        return sum(
            (2 ** value - 1) / math.log2(index + 2)
            for index, value in enumerate(values)
        )

    actual = [
        relevance.get(item, 0)
        for item in retrieved[:k]
    ]

    ideal = sorted(relevance.values(), reverse=True)[:k]

    ideal_score = dcg(ideal)

    if ideal_score == 0:
        return 0.0

    return dcg(actual) / ideal_score


# ------------------------------------------------------------
# 10. PROMPT ENGINEERING
# ------------------------------------------------------------

def build_prompt(question, context):

    return f"""
SYSTEM:
You are a helpful AI research assistant.
Answer using the supplied context.
If the answer is unavailable, say you do not know.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# ------------------------------------------------------------
# 11. STRUCTURED OUTPUT
# ------------------------------------------------------------
# Structured output means returning data in a predictable format.

def make_structured_answer(question, answer, source):

    return {
        "question": question,
        "answer": answer,
        "source": source,
        "verified": bool(answer and source)
    }


# ------------------------------------------------------------
# 12. LLM CONCEPTS - EXPLANATIONS
# ------------------------------------------------------------

LLM_CONCEPTS = {
    "Transformer": "Neural architecture used by modern LLMs.",
    "Self Attention": "Helps tokens understand relationships with other tokens.",
    "Context Window": "Maximum input and output context a model can process.",
    "KV Cache": "Stores attention keys and values to speed up generation.",
    "Temperature": "Controls randomness in generated responses.",
    "Top-P": "Samples from a probability distribution nucleus.",
    "Fine Tuning": "Further trains a pretrained model for a task.",
    "LoRA": "Efficient fine-tuning using small trainable low-rank matrices.",
    "QLoRA": "Combines quantization and LoRA.",
    "Quantization": "Uses lower precision numbers to reduce model memory.",
    "Distillation": "Transfers knowledge from a larger teacher model to a smaller model.",
    "HyDE": "Uses a hypothetical answer to improve retrieval.",
    "GraphRAG": "Uses relationships between entities for retrieval.",
    "Reranking": "Reorders retrieved documents by relevance."
}


def show_concepts():

    print("\nLLM AND RAG CONCEPTS")

    for name, explanation in LLM_CONCEPTS.items():
        print(f"{name}: {explanation}")


# ------------------------------------------------------------
# 13. MAIN DEMONSTRATION
# ------------------------------------------------------------

def main():

    token_demo()

    print("\n2. CHUNKING")

    text = (
        "Artificial intelligence helps companies automate tasks. "
        "Retrieval augmented generation connects language models "
        "with external knowledge. Vector databases store embeddings."
    )

    chunks = chunk_text(text, chunk_size=8, overlap=2)

    for number, chunk in enumerate(chunks, start=1):
        print(f"Chunk {number}: {chunk}")

    print("\n3. COSINE SIMILARITY")

    vocabulary = ["python", "ai", "rag"]

    vector_a = keyword_vector("Python AI", vocabulary)
    vector_b = keyword_vector("AI RAG", vocabulary)

    print(cosine_similarity(vector_a, vector_b))

    print("\n4. BM25 SEARCH")

    documents = [
        "Python is used for AI engineering.",
        "RAG uses document retrieval and language models.",
        "Databases store structured information."
    ]

    results = bm25_search("Python AI", documents)

    for index, score in results:
        print(f"{score:.4f}: {documents[index]}")

    print("\n5. RECALL, MRR AND NDCG")

    retrieved = ["doc2", "doc1", "doc3"]
    relevant = {"doc1", "doc2"}

    print("Recall@3:", recall_at_k(retrieved, relevant, 3))
    print("MRR:", reciprocal_rank(retrieved, relevant))

    relevance = {
        "doc1": 3,
        "doc2": 2,
        "doc3": 0
    }

    print("NDCG@3:", round(ndcg_at_k(retrieved, relevance, 3), 4))

    print("\n6. PROMPT ENGINEERING")

    print(build_prompt(
        "What is RAG?",
        "RAG combines retrieval with language generation."
    ))

    print("\n7. STRUCTURED OUTPUT")

    print(make_structured_answer(
        "What is RAG?",
        "Retrieval Augmented Generation.",
        "learning_dataset"
    ))

    show_concepts()

    print("\nPHASE 2 DEMONSTRATION COMPLETED")


if __name__ == "__main__":
    main()


# INTERVIEW QUESTIONS
#
# 1. What is tokenization?
# 2. Explain self-attention.
# 3. What is a context window?
# 4. What is the difference between dense and sparse retrieval?
# 5. Explain BM25.
# 6. What is hybrid search?
# 7. Explain Reciprocal Rank Fusion.
# 8. What is reranking?
# 9. What is the difference between LoRA and QLoRA?
# 10. Explain Recall, MRR and NDCG.