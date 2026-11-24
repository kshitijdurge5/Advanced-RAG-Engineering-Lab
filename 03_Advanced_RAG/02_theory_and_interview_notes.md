# Project 3 — Advanced Retrieval Engineering

**Code:** `01_advanced_retrieval_engineering.py`  
**Status:** Completed successfully

## 1. Why advanced retrieval?

Basic vector search can miss exact keywords, return repetitive passages, or rank a less useful passage above a better one. Advanced methods address different weaknesses:

- Keyword search helps with exact terms.
- Vector search helps with semantic similarity.
- Hybrid search combines retrieval signals.
- MMR encourages diversity.
- Reranking improves candidate ordering.
- Multi-query retrieval explores alternative phrasings.
- Contextual compression reduces irrelevant content.

These methods are complementary. They do not have to form one fixed serial chain in every application.

## 2. Project setup

The script loaded the seven-page TechVision AI learning PDF, split it into chunks, created Hugging Face embeddings, and used a persistent Chroma collection named `techvision_rag`.

Libraries/components used:
- `PyPDFLoader`
- `RecursiveCharacterTextSplitter`
- `HuggingFaceEmbeddings`
- `Chroma`
- `rank_bm25.BM25Okapi`
- `sentence_transformers.CrossEncoder`
- Gemini through `ChatGoogleGenerativeAI`
- LangChain classic retrieval components for multi-query and contextual compression

The project reused the `techvision_rag` collection and was designed not to add duplicate chunks when data already existed.

## 3. BM25 keyword retrieval

**Library:** `rank_bm25`  
**Class:** `BM25Okapi`

BM25 is a ranking function used in information retrieval. It scores documents based on query-term matches. Its scoring considers term frequency, inverse document frequency, and document-length normalization.

It is useful for exact terminology, identifiers, names, and technical phrases. It may not recognize that different phrases have similar meanings.

Example: a query for "access control" can match a document containing "Access control policies restrict access to confidential records."

## 4. Vector similarity retrieval

The project used Chroma with `all-MiniLM-L6-v2` embeddings. Vector retrieval embeds a query and compares it with stored document vectors.

It can retrieve semantically related passages even when wording differs. For example, "How do we restrict access to private information?" may match "Access control policies determine who can view confidential material."

BM25 and vector search solve different problems.

## 5. Hybrid search

Hybrid search combines lexical/keyword retrieval and semantic/vector retrieval.

In this project:
1. BM25 returned a ranked list.
2. Chroma vector search returned another ranked list.
3. Reciprocal Rank Fusion combined the rankings.

BM25 and vector scores are not necessarily on the same scale, so adding their raw scores directly may be inappropriate.

## 6. Reciprocal Rank Fusion (RRF)

RRF combines ranked lists using each result's rank:

\[
RRF(d)=\sum_{r\in R}\frac{1}{k+\operatorname{rank}_r(d)}
\]

Where `d` is a document, `R` is the set of result lists, `k` is a constant, and `rank` is the document's position in a list.

Our implementation used `k=60` and one-based ranks in the contribution calculation.

Illustration: a document ranked first in one list and third in another receives:

\[
\frac{1}{60+1}+\frac{1}{60+3}
\]

RRF combines rankings, not the original raw scores.

## 7. MMR — Maximum Marginal Relevance

**Chroma method:** `max_marginal_relevance_search`

MMR balances:
1. Relevance to the query.
2. Diversity among selected documents.

Without diversity, retrieval may return several nearly identical passages. MMR can select relevant passages that cover more distinct information.

Conceptually:

\[
MMR=\lambda\cdot Relevance-(1-\lambda)\cdot Redundancy
\]

Higher `lambda` emphasizes relevance; lower `lambda` gives more weight to diversity.

## 8. CrossEncoder reranking

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**Library:** `sentence_transformers`  
**Class:** `CrossEncoder`

A CrossEncoder processes a query and candidate document together and produces a relevance score.

| Embedding / bi-encoder retrieval | CrossEncoder |
|---|---|
| Encodes query and documents separately | Processes query-document pairs together |
| Efficient for large-scale candidate search | More expensive per candidate |
| Useful for first-stage retrieval | Useful for reranking a smaller set |

A common two-stage design retrieves candidates first, scores query-document pairs with the CrossEncoder, sorts by score, and keeps the strongest results.

Our output contained negative scores such as `-11.0425`. These are raw model scores, not probabilities. A negative score is not automatically an error; compare relative ordering and do not interpret it as a percentage.

## 9. Multi-query retrieval

**Component:** `MultiQueryRetriever`  
**Model:** Gemini via `ChatGoogleGenerativeAI`

Multi-query retrieval asks an LLM to create alternative phrasings of the user's question, retrieves documents for those variants, and combines results.

Example original question: "How does the company protect information?"

Alternative searches might include:
- "What are the company's data security policies?"
- "How is access to confidential information controlled?"
- "What information governance measures are used?"

This can improve coverage when the user's wording differs from the source wording. It adds model calls, latency, and cost.

The project successfully retrieved five documents using this step.

## 10. Contextual compression

**Components:** `ContextualCompressionRetriever`, `LLMChainExtractor`  
**Model:** Gemini

A retrieved passage may contain more information than the question needs. Contextual compression attempts to extract the parts relevant to the query before passing context onward.

Compression can reduce irrelevant context, but it must be checked for omitted qualifications or distorted meaning.

The project returned one compressed document.

## 11. How the methods fit together

The combined script demonstrated:
- BM25 retrieval
- Vector retrieval
- Hybrid retrieval using RRF
- MMR retrieval
- CrossEncoder reranking
- Gemini multi-query retrieval
- Gemini-based contextual compression

These were demonstrated as methods in the script. Do not assume every method necessarily feeds the next in one serial pipeline. A production system should choose and evaluate the sequence that fits the task.

## 12. Project results

| Measure | Result |
|---|---:|
| Original PDF pages | 7 |
| Newly generated chunks | 35 |
| ChromaDB stored vectors | 35 |
| BM25 results | 5 |
| Vector results | 5 |
| Hybrid results | 5 |
| MMR results | 5 |
| Reranked results | 5 |
| Multi-query results | 5 |
| Compressed results | 1 |

The run ended with `PROJECT 3 COMPLETED SUCCESSFULLY!`.

## 13. Engineering considerations

- Evaluate hybrid retrieval against a baseline.
- RRF avoids direct raw-score mixing, but ranking quality still matters.
- MMR can improve diversity while trading away some top-result relevance.
- CrossEncoder reranking adds latency and compute; use it on a limited candidate set.
- Multi-query adds LLM calls, latency, and cost.
- Compression can omit important facts; validate its output.
- Preserve source metadata.
- Test answerable and unanswerable questions.

## 14. Interview questions

**What is BM25?** A keyword ranking algorithm using term frequency, inverse document frequency, and length normalization.

**What is vector search?** Retrieval based on similarity between query and document embeddings.

**Why hybrid search?** It combines exact keyword matching and semantic matching.

**What is RRF?** A rank-fusion method combining multiple ranked lists using reciprocal rank contributions.

**What is MMR?** A method balancing query relevance and diversity.

**What is a CrossEncoder?** A model that jointly processes a query and candidate document to produce a relevance score.

**Why rerank?** First-stage retrieval efficiently finds candidates; reranking can improve their order.

**Why can CrossEncoder scores be negative?** They are raw model scores, not necessarily probabilities.

**What is multi-query retrieval?** Generating alternative query phrasings and retrieving across them.

**What is contextual compression?** Reducing retrieved context to information relevant to the question.

## Key takeaway

Project 3 moves beyond basic semantic search. It demonstrates keyword retrieval, semantic retrieval, rank fusion, diversity, reranking, query expansion, and compression.
