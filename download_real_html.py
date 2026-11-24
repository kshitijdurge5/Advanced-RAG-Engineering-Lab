
# ==========================================
# DOWNLOAD A REAL NASA HTML WEBPAGE
# ==========================================

# Import requests to access webpages
#requests is a Python library used to communicate with websites and APIs.
import requests

# Import os to manage file paths
import os

# NASA's official climate change webpage
url = "https://science.nasa.gov/climate-change/evidence/"

# Create the documents folder if it does not exist
# makedirs()	Creates directories (folders)
# "documents"	Folder name
# exist_ok=True	Don't raise an error if the folder already exists

os.makedirs("documents", exist_ok=True)



# Send a request to download the webpage
response = requests.get(
    url,
    timeout=30,#(This means Python will wait up to 30 seconds for the request before timing out.Why?Imagine the website is slow or not responding. We don't want our program to wait indefinitely.) 
    headers={"User-Agent": "Mozilla/5.0"}
)

# Check whether the webpage request was successful
#raise_for_status() checks the HTTP response.
#- If the response is successful, execution continues.
#- If the response indicates an HTTP error, Python raises an exception.
response.raise_for_status()

# Save the original HTML into our documents folder
file_path = "documents/nasa_climate_evidence.html"

with open(file_path, "w", encoding="utf-8") as file:

    # Save the HTML source code
    file.write(response.text)

# Display download information
print("NASA HTML downloaded successfully!")

print("HTTP Status:", response.status_code)

print("HTML File:", file_path)

print("HTML Size:", len(response.text), "characters")
