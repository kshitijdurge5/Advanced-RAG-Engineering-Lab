
# ============================================================
# PROJECT 1: COMPLETE DATA INGESTION
# ============================================================
# Libraries covered:
# 1. TextLoader       -> TXT files
# 2. CSVLoader        -> CSV files
# 3. BSHTMLLoader     -> HTML files
# 4. PyPDFLoader      -> PDF files
# 5. Docx2txtLoader   -> DOCX files
# 6. DirectoryLoader  -> Load multiple files from a folder
#
# All loaded content is converted into LangChain Documents.
# ============================================================


# STEP 1: IMPORT LIBRARIES

from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    BSHTMLLoader,
    PyPDFLoader,
    Docx2txtLoader,
    DirectoryLoader
)


# STEP 2: DEFINE DOCUMENT LOCATIONS

# __file__ represents the location of this Python file.
# .parent gives the 01_Data_Ingestion folder.
# .parent.parent gives the main project folder.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_FOLDER = PROJECT_ROOT / "documents"


# STEP 3: CHECK DOCUMENT FOLDER

if not DOCUMENTS_FOLDER.exists():
    raise FileNotFoundError(
        f"Documents folder not found: {DOCUMENTS_FOLDER}"
    )

print("\nPROJECT 1: DATA INGESTION")
print("=" * 60)


# STEP 4: TEXT LOADER

# TextLoader reads ordinary .txt files.

text_file = DOCUMENTS_FOLDER / "company_information.txt"

text_loader = TextLoader(
    str(text_file),
    encoding="utf-8"
)

text_documents = text_loader.load()

print("\n1. TEXT LOADER")
print("Total documents:", len(text_documents))
print("Content:", text_documents[0].page_content[:200])
print("Metadata:", text_documents[0].metadata)


# STEP 5: CSV LOADER

# CSVLoader converts each CSV row into a separate Document.

csv_file = DOCUMENTS_FOLDER / "employee_data.csv"

csv_loader = CSVLoader(
    str(csv_file),
    encoding="utf-8"
)

csv_documents = csv_loader.load()

print("\n2. CSV LOADER")
print("Total CSV rows:", len(csv_documents))
print("First row:", csv_documents[0].page_content)
print("Metadata:", csv_documents[0].metadata)


# STEP 6: HTML LOADER

# BSHTMLLoader extracts readable text from HTML.
# open_encoding handles the file's UTF-8 encoding.

html_file = DOCUMENTS_FOLDER / "nasa_climate_evidence.html"

html_loader = BSHTMLLoader(
    str(html_file),
    open_encoding="utf-8"
)

html_documents = html_loader.load()

print("\n3. HTML LOADER")
print("Total HTML documents:", len(html_documents))
print("Content:", html_documents[0].page_content[:200])
print("Metadata:", html_documents[0].metadata)


# STEP 7: PDF LOADER

# PyPDFLoader generally creates one Document per PDF page.

pdf_file = DOCUMENTS_FOLDER / "TechVision_AI_RAG_Learning_Dataset.pdf"

pdf_loader = PyPDFLoader(str(pdf_file))

pdf_documents = pdf_loader.load()

print("\n4. PDF LOADER")
print("Total PDF pages:", len(pdf_documents))
print("First page:", pdf_documents[0].page_content[:200])
print("Metadata:", pdf_documents[0].metadata)


# STEP 8: DOCX LOADER

# Docx2txtLoader extracts text from Word documents.

docx_file = DOCUMENTS_FOLDER / "company_report.docx"

docx_loader = Docx2txtLoader(str(docx_file))

docx_documents = docx_loader.load()

print("\n5. DOCX LOADER")
print("Total DOCX documents:", len(docx_documents))
print("Content:", docx_documents[0].page_content[:200])
print("Metadata:", docx_documents[0].metadata)


# STEP 9: COMBINE ALL DOCUMENTS

# Each loader returns a list of LangChain Document objects.
# We combine all lists into one collection.

all_documents = (
    text_documents
    + csv_documents
    + html_documents
    + pdf_documents
    + docx_documents
)

print("\n6. COMBINED DOCUMENT COLLECTION")
print("Total documents:", len(all_documents))
print("Document class:", type(all_documents[0]))
print("First document metadata:", all_documents[0].metadata)


# STEP 10: DIRECTORY LOADER

# DirectoryLoader loads files from a folder using a file pattern.
# Each loader below handles one file type.

print("\n7. DIRECTORY LOADER")
print("=" * 60)


# TXT files

txt_directory_loader = DirectoryLoader(
    str(DOCUMENTS_FOLDER),
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

directory_txt_documents = txt_directory_loader.load()


# CSV files

csv_directory_loader = DirectoryLoader(
    str(DOCUMENTS_FOLDER),
    glob="*.csv",
    loader_cls=CSVLoader,
    loader_kwargs={"encoding": "utf-8"}
)

directory_csv_documents = csv_directory_loader.load()


# HTML files

html_directory_loader = DirectoryLoader(
    str(DOCUMENTS_FOLDER),
    glob="*.html",
    loader_cls=BSHTMLLoader,
    loader_kwargs={"open_encoding": "utf-8"}
)

directory_html_documents = html_directory_loader.load()


# PDF files

pdf_directory_loader = DirectoryLoader(
    str(DOCUMENTS_FOLDER),
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

directory_pdf_documents = pdf_directory_loader.load()


# DOCX files

docx_directory_loader = DirectoryLoader(
    str(DOCUMENTS_FOLDER),
    glob="*.docx",
    loader_cls=Docx2txtLoader
)

directory_docx_documents = docx_directory_loader.load()


# STEP 11: COMBINE DIRECTORY LOADER RESULTS

directory_documents = (
    directory_txt_documents
    + directory_csv_documents
    + directory_html_documents
    + directory_pdf_documents
    + directory_docx_documents
)

print("TXT documents:", len(directory_txt_documents))
print("CSV documents:", len(directory_csv_documents))
print("HTML documents:", len(directory_html_documents))
print("PDF pages:", len(directory_pdf_documents))
print("DOCX documents:", len(directory_docx_documents))

print("\nTotal DirectoryLoader documents:", len(directory_documents))


# STEP 12: INSPECT A DOCUMENT

# Every LangChain Document has:
# page_content -> extracted text
# metadata     -> source information and other details

print("\n8. DOCUMENT STRUCTURE")
print("=" * 60)

sample_document = directory_documents[0]

print("Document type:", type(sample_document))
print("\nPage content:")
print(sample_document.page_content[:300])

print("\nMetadata:")
print(sample_document.metadata)


# STEP 13: FINAL SUMMARY

print("\n" + "=" * 60)
print("PROJECT 1 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("Individual loader documents:", len(all_documents))
print("DirectoryLoader documents:", len(directory_documents))
