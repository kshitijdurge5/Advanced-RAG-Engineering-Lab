
# ==========================================
# CREATE OUR FIRST SAMPLE DOCUMENT
# ==========================================

# Import os to work with folders and file paths
import os

# Create the documents folder if it does not exist
os.makedirs("documents", exist_ok=True)

# Store some sample company information
company_information = """
COMPANY NAME: TechVision AI

ABOUT THE COMPANY:
TechVision AI is a technology company that develops
Artificial Intelligence and Machine Learning solutions.

DEPARTMENTS:

1. Artificial Intelligence Department
This department develops Generative AI applications,
Large Language Models and AI-powered chatbots.

2. Machine Learning Department
This department develops predictive models,
classification systems and recommendation engines.

3. Data Science Department
This department performs data analysis,
data visualisation and statistical modelling.

4. Human Resources Department
This department manages employee recruitment,
training, salaries and performance reviews.

TECHNOLOGIES USED:

Python
Machine Learning
Deep Learning
LangChain
Gemini API
Groq API
ChromaDB
FastAPI

COMPANY LOCATION:
Mumbai, India.
"""

# Create the TXT file path
file_path = "documents/company_information.txt"

# Open the file in write mode
# encoding="utf-8" supports different text characters
with open(file_path, "w", encoding="utf-8") as file:

    # Write company information into the file
    file.write(company_information)

# Display success message
print("Sample document created successfully!")

# Display the location of the file
print("File location:", file_path)


# ==========================================
# CREATE SAMPLE EMPLOYEE CSV FILE
# ==========================================

# Import pandas to create a table
import pandas as pd

# Create employee information using a dictionary
employee_data = {

    # Employee names
    "Name": ["Rahul", "Priya", "Amit", "Sneha", "Karan"],

    # Employee departments
    "Department": [
        "AI",
        "Data Science",
        "Machine Learning",
        "HR",
        "AI"
    ],

    # Employee salaries
    "Salary": [60000, 55000, 65000, 40000, 70000],

    # Employee experience in years
    "Experience": [2, 3, 4, 1, 5]
}

# Convert dictionary into a pandas DataFrame
employee_df = pd.DataFrame(employee_data)

# Save the DataFrame as a CSV file
employee_df.to_csv(
    "documents/employee_data.csv",
    index=False
)

# Display the employee dataset
print("\nEMPLOYEE DATA:")
print(employee_df)

print("\nEmployee CSV file created successfully!")
