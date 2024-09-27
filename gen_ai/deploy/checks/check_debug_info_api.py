import requests

url = "http://127.0.0.1:8080/debug_response/"

data = {"prediction_id": "d5ea985d-5475-421d-94de-560a546668fa"}

response = requests.post(url, data=data)

print(response.json())
