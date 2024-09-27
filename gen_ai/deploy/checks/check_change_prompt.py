import requests

url = "http://127.0.0.1:8080/change_prompt/"

data = {
    "project_id": "8e6b286d-950a-4945-8734-3aecab7e71b6",
    "user_id": "user_123",
    "prompt_name": "aspect_based_summary_prompt",
    "prompt_value": "You a very good summarizer",
}

response = requests.post(url, data=data)

print(response.json())
