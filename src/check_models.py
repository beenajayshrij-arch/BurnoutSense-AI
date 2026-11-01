import os
import requests

api_key = os.environ["GEMINI_API_KEY"]
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

resp = requests.get(url)
data = resp.json()

for model in data.get("models", []):
    name = model.get("name", "")
    methods = model.get("supportedGenerationMethods", [])
    if "flash" in name.lower() and "generateContent" in methods:
        print(name)