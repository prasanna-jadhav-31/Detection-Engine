import requests
import os

IMAGE_PATH = r"C:\Users\vaish\.gemini\antigravity\brain\94e2d2b4-3b76-4761-9099-5b83bc4a000c\media__1776180289336.jpg"
URL = "http://127.0.0.1:8000/analyze"

def test_analyze():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    with open(IMAGE_PATH, "rb") as f:
        files = {"file": ("cat.jpg", f, "image/jpeg")}
        try:
            print("Sending request to /analyze...")
            response = requests.post(URL, files=files)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Response JSON:")
                print(response.json())
            else:
                print("Error Detail:")
                print(response.text)
        except Exception as e:
            print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_analyze()
