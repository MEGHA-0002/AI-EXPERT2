import requests
import json

API_KEY = "your_api_key"

url = "https://api.openai.com/v1/images/generations"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "gpt-image-1",
    "prompt": "A futuristic city at sunset, cyberpunk style, neon lights, highly detailed",
    "size": "1024x1024",
    "quality": "high",
    "background": "auto",
    "output_format": "png",
    "n": 1
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    print("Image generated successfully!")
    print(response.json())
else:
    print("Error:", response.text)