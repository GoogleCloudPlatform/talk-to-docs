import requests
import base64


url = "https://test-misha-deploy-wbkml5x37q-uc.a.run.app/create_project/" 

example_bytearray = bytearray(b"dummy file content")
encoded_file_object = base64.b64encode(example_bytearray).decode('utf-8')

dummy_data = {
    "project_name": "Test Project",
    "user_id": "test-user-123",
    "local_documents": [
        {
            "file_object": encoded_file_object,
            "file_title": "dummy_file.txt"
        }
    ],
    "external_documents": [
        {
            "document_url": "https://example.com/test-doc",
            "created_on": "2024-01-01T00:00:00",
            "document_name": "Test External Doc"
        }
    ]
}

response = requests.post(url, json=dummy_data)

print(response.status_code)
print(response.json())
