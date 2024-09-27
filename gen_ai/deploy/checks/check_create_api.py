import requests

# API endpoint
url = "http://127.0.0.1:8080/create_project"

# Replace with the actual user ID and project name
user_id = "user_123"
project_name = "Test Project 2"

# Path to the PDF file to upload
file_path = "/home/chertushkin/platform-gen-ai/peakpdf.pdf"

# Prepare the files parameter as a list of tuples (name, file, content_type)
with open(file_path, "rb") as f:
    files = [
        ('files', ('file1.pdf', open(file_path, 'rb'), 'application/pdf')),
        ('files', ('file2.pdf', open(file_path, 'rb'), 'application/pdf'))
    ]

    # Data to send with the request
    data = {"user_id": user_id, "project_name": project_name}

    # Make the POST request
    response = requests.post(url, data=data, files=files)

# Print the response from the server
print(response.text)

# import requests

# # API endpoint
# url = "http://127.0.0.1:8080/document/123"

# document_ids = ["doc_id_1", "doc_id_2", "doc_id_3"]

# # Correct payload format expected by FastAPI
# payload = {
#     "document_ids": document_ids
# }

# # Send the DELETE request with the correct JSON format
# response = requests.get(url)

# # Print the response from the server
# print(response.text)
