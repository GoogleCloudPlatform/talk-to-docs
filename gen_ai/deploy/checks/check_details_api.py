import requests

url = "http://127.0.0.1:8080/project_details/"

data = {"project_id": "8e6b286d-950a-4945-8734-3aecab7e71b6", "user_id": "user_123"}

response = requests.post(url, data=data)

print(response.json())
