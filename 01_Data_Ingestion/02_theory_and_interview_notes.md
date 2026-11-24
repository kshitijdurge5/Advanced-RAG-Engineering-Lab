# Project 1 — Data Ingestion

**Code:** `01_multiple_document_loaders.py`  
**Status:** Completed

## Purpose

Data ingestion collects information from different sources, extracts its contents, and prepares it for later processing. A RAG application may need to read PDFs, CSVs, Word files, HTML pages, and plain-text files. Loaders convert these formats into a common representation.

## Execution flow

1. Select a source file.
2. Use a loader designed for its format.
3. Extract text and metadata.
4. Return LangChain `Document` objects.
5. Combine documents for later chunking and indexing.

## Libraries and loaders

LangChain is a framework that provides reusable components for language-model applications. In this project, its document loaders and `Document` representation were used.

| Loader | Purpose |
|---|---|
| `TextLoader` | Reads `.txt` files |
| `CSVLoader` | Converts CSV rows into text documents |
| `BSHTMLLoader` | Extracts readable text from HTML |
| `PyPDFLoader` | Extracts text from PDF pages |
| `Docx2txtLoader` | Extracts text from Word `.docx` files |
| `DirectoryLoader` | Loads multiple matching files from a directory using a configured loader |

### What each loader does

- **TextLoader:** reads plain text, such as notes, policies, and documentation.
- **CSVLoader:** reads tabular data. In a row-oriented setup, each row becomes a document. It does not create a relational database or perform SQL analysis.
- **BSHTMLLoader:** extracts readable text from HTML markup.
- **PyPDFLoader:** extracts text from PDF pages and commonly returns page-level documents with page metadata. Scanned PDFs may require OCR; complex layouts and tables may not extract perfectly.
- **Docx2txtLoader:** extracts text from Word documents.
- **DirectoryLoader:** processes multiple matching files. Different file types may need separate loader configurations.

## LangChain Document

A `Document` commonly contains:

- `page_content`: the extracted text.
- `metadata`: information about the source or context, such as file path, page number, or row.

Example:

```python
Document(
    page_content="TechVision AI develops AI applications.",
    metadata={"source": "company_information.txt"}
)
```

`page_content` is the information; `metadata` describes where it came from. Metadata supports filtering, traceability, and citations later in RAG.

## Project output

The successful run reported:

| Source | Documents |
|---|---:|
| TXT | 1 |
| CSV | 5 |
| HTML | 1 |
| PDF | 7 |
| DOCX | 1 |
| **Total** | **15** |

The directory-loading demonstration also reported 15 documents. These counts reflect this project's supplied dataset and configuration.

## Why it matters in RAG

Ingestion provides the source content for later steps:

1. Clean or normalize text when necessary.
2. Split long documents into chunks.
3. Preserve metadata.
4. Generate embeddings.
5. Store chunks in a searchable index.

## Engineering considerations

- PDF extraction may not preserve layout, tables, or images.
- HTML can contain navigation and boilerplate that should be filtered.
- Encoding must be handled correctly.
- Large jobs should log failures and track processed files.
- Production systems should consider updates, duplicate detection, permissions, and source versions.

## Interview questions

**What is data ingestion?** Collecting data, extracting its contents, and preparing it for downstream processing.

**What is a LangChain Document?** A structured object commonly containing `page_content` and `metadata`.

**Why preserve metadata?** It identifies source, page, row, or other context and supports traceability and citations.

**What does DirectoryLoader do?** Loads multiple matching files from a directory using a configured loader.

**Does PyPDFLoader perform OCR?** Do not assume so; scanned PDFs generally need OCR.

**Why use different loaders?** File formats store information differently and need format-appropriate extraction.

## Key takeaway

Project 1 converts different file formats into a common collection of LangChain documents. Those documents become the input for chunking, embeddings, and retrieval.
