import requests

API_URL = "https://api-inference.huggingface.co/models/TrieuNguyen/chest_xray_pneumonia"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("./examples/person19_virus_50.jpeg")

print(output)