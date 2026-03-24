import requests
import os

BASE_URL = "http://localhost:8001"

def test_analyze_text(text, label):
    print(f"\nTesting {label} Text Emotion...")
    data = {"text": text, "language": "hi"}
    response = requests.post(f"{BASE_URL}/analyze", data=data)
    res_json = response.json()
    print(f"Input: {text}")
    print(f"Text Emotion: {res_json.get('text', {}).get('emotion')}")
    print(f"Final Emotion: {res_json.get('final_emotion')}")
    print(f"Final Confidence: {res_json.get('final_confidence')}")

if __name__ == "__main__":
    test_analyze_text("मुझे बहुत गुस्सा आ रहा है", "Angry")
    test_analyze_text("मैं आज बहुत खुश हूँ", "Happy")
