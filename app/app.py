"""
NLU A6: Naive RAG vs Contextual Retrieval — Flask Chatbot
Author  : Supanut Kompayak
ID      : st126055
Chapter : 5 — Logistic Regression
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rag import ask

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    result = ask(question, top_k=3)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
