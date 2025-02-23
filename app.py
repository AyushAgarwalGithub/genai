from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM
import torch


app = Flask(__name__)
CORS(app)

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1).squeeze().tolist()
    labels = ["Negative", "Neutral", "Positive"]
    sentiment = labels[scores.index(max(scores))]
    return {"label": sentiment, "score": max(scores)}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    user_text = data.get("text", "")
    
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
    
    sentiment, score = analyze_sentiment(user_text)
    
    response = {
        "sentiment": sentiment,
        "sentiment_score": round(score, 2)
    }
    
    return jsonify(response)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
