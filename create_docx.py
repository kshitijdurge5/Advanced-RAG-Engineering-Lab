
# ==========================================
# STEP 22: CREATE A WORD DOCUMENT
# ==========================================

# Import Document from python-docx
from docx import Document

# Create a new Word document
doc = Document()

# Add the main heading
doc.add_heading("TechVision AI Company Report", level=1)

# Add a paragraph
doc.add_paragraph(
    "TechVision AI is a technology company based in Mumbai, India."
)

# Add another heading
doc.add_heading("Our AI Services", level=2)

# Add services as bullet points
doc.add_paragraph(
    "Generative AI Development",
    style="List Bullet"
)

doc.add_paragraph(
    "Machine Learning Solutions",
    style="List Bullet"
)

doc.add_paragraph(
    "AI Chatbot Development",
    style="List Bullet"
)

# Add another heading
doc.add_heading("Technology Stack", level=2)

# Add technology information
doc.add_paragraph(
    "Python, LangChain, Gemini, Groq, ChromaDB and FastAPI."
)

# Save the Word document
doc.save("documents/company_report.docx")

# Display success message
print("DOCX document created successfully!")
