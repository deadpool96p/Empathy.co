import requests
import os

BASE_URL = "http://localhost:8001"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.json()}")

def test_analyze_text_hindi():
    print("\nTesting Hindi Text Emotion...")
    text = "मुझे बहुत गुस्सा आ रहा है" # I am very angry
    data = {"text": text, "language": "hi"}
    response = requests.post(f"{BASE_URL}/analyze", data=data)
    print(f"Input: {text}")
    print(f"Response: {response.json().get('text')}")
    print(f"Final Emotion: {response.json().get('final_emotion')}")

def test_analyze_text_happy():
    print("\nTesting Happy Text Emotion...")
    text = "मैं आज बहुत खुश हूँ" # I am very happy today
    data = {"text": text, "language": "hi"}
    response = requests.post(f"{BASE_URL}/analyze", data=data)
    print(f"Input: {text}")
    print(f"Response: {response.json().get('text')}")
    print(f"Final Emotion: {response.json().get('final_emotion')}")

def test_file_size_limit():
    print("\nTesting File Size Limit...")
    # Create a large dummy file (51MB)
    dummy_file = "large_dummy.wav"
    with open(dummy_file, "wb") as f:
        f.write(b"\0" * (51 * 1024 * 1024))
    
    try:
        with open(dummy_file, "rb") as f:
            files = {"audio": f}
            response = requests.post(f"{BASE_URL}/analyze", files=files, data={"language": "en"})
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
    finally:
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

if __name__ == "__main__":
    try:
        test_health()
        test_analyze_text_hindi()
        test_analyze_text_happy()
        test_file_size_limit()
    except Exception as e:
        print(f"Verification failed: {e}")
