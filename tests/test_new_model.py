from transformers import pipeline

try:
    classifier = pipeline("text-classification", model="MilaNLProc/xlm-emo-t", top_k=None)
    result = classifier("मुझे बहुत गुस्सा आ रहा है") # Hindi: I am very angry
    print(f"Hindi result: {result}")
    result2 = classifier("मैं आज बहुत खुश हूँ") # Hindi: I am very happy today
    print(f"Hindi result 2: {result2}")
except Exception as e:
    print(f"Error loading model: {e}")
